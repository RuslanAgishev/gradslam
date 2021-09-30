#!/usr/bin/env python

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import os
from time import time
from gradslam.slam.pointfusion import PointFusion
from gradslam import Pointclouds, RGBDImages
from threading import RLock
# ROS
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import message_filters
import tf2_ros
from ros_numpy import msgify, numpify
from tf.transformations import quaternion_from_matrix


class GradslamROS:
    def __init__(self, odometry='gt', height: int = 240, width: int = 320):
        self.bridge = CvBridge()
        self.tf = tf2_ros.Buffer()
        self.tf_sub = tf2_ros.TransformListener(self.tf)
        self.world_frame = 'subt'
        self.robot_frame = 'X1_ground_truth'
        self.camera_frame = 'X1/base_link/front_realsense_optical'
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.slam = PointFusion(odom=odometry, dsratio=1, device=self.device)
        self.width, self.height = width, height
        self.pointclouds = Pointclouds(device=self.device)
        self.prev_frame = None
        self.route = Path()
        self.route.header.frame_id = self.world_frame
        self.route_pub = rospy.Publisher('~route', Path, queue_size=2)
        self.pc_pub = rospy.Publisher('~cloud', PointCloud2, queue_size=1)
        self.extrinsics_lock = RLock()
        self.map_step = 16
        self.max_depth_range = 10.0

        self.robot2camera = self.get_extrinsics()
        rospy.loginfo(f'Got extrinsics: {numpify(self.robot2camera.transform)}')

        # Subscribe to topics
        caminfo_sub = message_filters.Subscriber('/X1/front_rgbd/optical/camera_info', CameraInfo)
        rgb_sub = message_filters.Subscriber('/X1/front_rgbd/optical/image_raw', Image)
        depth_sub = message_filters.Subscriber('/X1/front_rgbd/depth/optical/image_raw', Image)

        # Synchronize the topics by time
        ats = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, caminfo_sub], queue_size=1, slop=0.1)
        ats.registerCallback(self.callback)

    def get_extrinsics(self):
        with self.extrinsics_lock:
            while not rospy.is_shutdown():
                try:
                    robot2camera = self.tf.lookup_transform(self.robot_frame, self.camera_frame,
                                                            rospy.Time.now(), rospy.Duration.from_sec(1.0))
                    return robot2camera
                except (tf2_ros.TransformException, tf2_ros.ExtrapolationException) as ex:
                    rospy.logwarn('Could not transform from robot %s to camera %s: %s.',
                                  self.robot_frame, self.camera_frame, ex)

    def callback(self, rgb_msg, depth_msg, caminfo_msg):
        t0 = time()
        try:
            world2robot = self.tf.lookup_transform(self.world_frame, self.robot_frame,
                                          rospy.Time.now(), rospy.Duration.from_sec(1.0))
        except (tf2_ros.TransformException, tf2_ros.ExtrapolationException) as ex:
            rospy.logwarn('Could not transform from world %s to robot %s: %s.',
                          self.world_frame, self.robot_frame, ex)
            return
        rospy.logdebug('Transformation search took: %.3f', time() - t0)
        T = numpify(world2robot.transform) @ numpify(self.robot2camera.transform)
        pose = torch.as_tensor(T, dtype=torch.float32).view(1, 1, 4, 4)

        try:
            # get rgb image
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, rgb_msg.encoding)
            rgb_image = np.asarray(rgb_image, dtype=np.float32)
            rgb_image = cv2.resize(rgb_image,
                                   (self.width, self.height),
                                   interpolation=cv2.INTER_LINEAR)
            # get depth image
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, depth_msg.encoding)
            # depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            depth_image = np.asarray(depth_image, dtype=np.float32)
            depth_image = np.clip(depth_image, 0.0, self.max_depth_range)
            depth_image = cv2.resize(depth_image,
                                     (self.width, self.height),
                                     interpolation=cv2.INTER_NEAREST)
        except CvBridgeError as e:
            rospy.logerr(e)
            return
        # get intrinsic params
        k = torch.as_tensor(caminfo_msg.K, dtype=torch.float32).view(3, 3)
        K = torch.eye(4)
        K[:3, :3] = k
        intrins = K.view(1, 1, 4, 4)

        assert rgb_image.shape[:2] == depth_image.shape
        w, h = rgb_image.shape[:2]
        rgb_image = torch.from_numpy(rgb_image).view(1, 1, w, h, 3)
        depth_image = torch.from_numpy(depth_image).view(1, 1, w, h, 1)

        # create gradslam input
        live_frame = RGBDImages(rgb_image, depth_image, intrins, pose).to(self.device)
        rospy.logdebug('Data preprocessing took: %.3f', time()-t0)

        # SLAM inference
        t0 = time()
        self.pointclouds, live_frame.poses = self.slam.step(self.pointclouds, live_frame, self.prev_frame)
        self.prev_frame = live_frame
        rospy.loginfo(f"Position: {live_frame.poses[..., :3, 3].squeeze()}")
        rospy.logdebug('SLAM inference took: %.3f', time() - t0)

        # publish odometry / path
        t0 = time()
        assert live_frame.poses.shape == (1, 1, 4, 4)
        pose = PoseStamped()
        pose.header.frame_id = self.world_frame
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = live_frame.poses[..., 0, 3]
        pose.pose.position.y = live_frame.poses[..., 1, 3]
        pose.pose.position.z = live_frame.poses[..., 2, 3]
        q = quaternion_from_matrix(live_frame.poses[0, 0].cpu().numpy())
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]

        self.route.poses.append(pose)
        self.route.header.stamp = rospy.Time.now()
        self.route_pub.publish(self.route)

        # publish point cloud
        n_pts = np.ceil(self.pointclouds.points_padded.shape[1] / self.map_step).astype(int)
        cloud = np.zeros((n_pts,), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                          ('r', 'f4'), ('g', 'f4'), ('b', 'f4')])
        for i, f in enumerate(['x', 'y', 'z']):
            cloud[f] = self.pointclouds.points_padded[..., i].squeeze().cpu().numpy()[::self.map_step]
        for i, f in enumerate(['r', 'g', 'b']):
            cloud[f] = self.pointclouds.colors_padded[..., i].squeeze().cpu().numpy()[::self.map_step] / 255.
        pc_msg = msgify(PointCloud2, cloud)
        pc_msg.header.stamp = rospy.Time.now()
        pc_msg.header.frame_id = self.world_frame
        self.pc_pub.publish(pc_msg)
        rospy.logdebug('Data publishing took: %.3f', time() - t0)


if __name__ == '__main__':
    rospy.init_node('gradslam_ros', log_level=rospy.INFO)
    odometry = rospy.get_param('~odometry')  # gt, icp, gradicp
    proc = GradslamROS(odometry=odometry)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
