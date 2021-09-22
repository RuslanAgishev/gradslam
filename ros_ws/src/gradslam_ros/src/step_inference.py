#!/usr/bin/env python

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import os
from time import time
from gradslam.slam.pointfusion import PointFusion
from gradslam import Pointclouds, RGBDImages
# ROS
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import message_filters
import tf2_ros
from ros_numpy import msgify, numpify
from tf.transformations import quaternion_from_matrix


class Processor:
    def __init__(self, height: int = 240, width: int = 320):
        self.bridge = CvBridge()
        self.tf = tf2_ros.Buffer()
        self.tf_sub = tf2_ros.TransformListener(self.tf)
        self.world_frame = 'subt'
        self.camera_frame = 'X1/base_link/front_realsense_optical'
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.slam = PointFusion(odom='gradicp', dsratio=1, device=self.device)
        self.width, self.height = width, height
        self.pointclouds = Pointclouds(device=self.device)
        self.initial_pose = torch.eye(4, dtype=torch.float32, device=self.device).view(1, 1, 4, 4)
        self.prev_frame = None
        self.initialized = False
        self.route = Path()
        self.route.header.frame_id = self.world_frame
        self.route_pub = rospy.Publisher('~route', Path, queue_size=2)

        # Subscribe to topics
        caminfo_sub = message_filters.Subscriber('/X1/front_rgbd/optical/camera_info', CameraInfo)
        rgb_sub = message_filters.Subscriber('/X1/front_rgbd/optical/image_raw', Image)
        depth_sub = message_filters.Subscriber('/X1/front_rgbd/depth/optical/image_raw', Image)

        # Synchronize the topics by time
        ats = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, caminfo_sub], queue_size=5, slop=0.1)
        ats.registerCallback(self.callback)

    def callback(self, rgb_msg, depth_msg, caminfo_msg):
        t0 = time()
        # try:
        #     tf = self.tf.lookup_transform(self.world_frame, self.camera_frame,
        #                                   rospy.Time.now(), rospy.Duration.from_sec(1.0))
        # except tf2_ros.TransformException as ex:
        #     rospy.logerr('Could not transform from world %s to camera %s: %s.',
        #                  self.world_frame, self.camera_frame, ex)
        #     return
        # T = numpify(tf.transform)
        # T = torch.as_tensor(T).view(1, 1, 4, 4)
        try:
            # get rgb image
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, rgb_msg.encoding)
            rgb_image = np.asarray(rgb_image, dtype=float)
            rgb_image = cv2.resize(
                rgb_image, (self.width, self.height), interpolation=cv2.INTER_LINEAR
            )
            # get depth image
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, depth_msg.encoding)
            depth_image = np.asarray(depth_image, dtype=np.int64)
            depth_image = cv2.resize(
                depth_image, (self.width, self.height), interpolation=cv2.INTER_NEAREST,
            )
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
        live_frame = RGBDImages(rgb_image, depth_image, intrins).to(self.device)
        rospy.logdebug('Data preprocessing took: %.3f', time()-t0)

        # SLAM inference
        t0 = time()
        if not self.initialized:
            live_frame.poses = self.initial_pose
            self.initialized = True
        self.pointclouds, live_frame.poses = self.slam.step(self.pointclouds, live_frame, self.prev_frame)
        self.prev_frame = live_frame
        rospy.loginfo(f"Pose: {live_frame.poses}")
        rospy.logdebug('SLAM inference took: %.3f', time() - t0)

        # publish odometry / path
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


if __name__ == '__main__':
    rospy.init_node('bag2data', log_level=rospy.DEBUG)
    ip = Processor()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
