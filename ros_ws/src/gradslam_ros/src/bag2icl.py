#!/usr/bin/env python

import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch import nn
import os
from time import time

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
import message_filters
import tf2_ros
from ros_numpy import msgify, numpify
import rospkg


class Processor:
    def __init__(self):
        self.bridge = CvBridge()
        self.tf = tf2_ros.Buffer()
        self.tf_sub = tf2_ros.TransformListener(self.tf)
        self.world_frame = 'subt'
        self.camera_frame = 'X1/base_link/front_realsense_optical'
        self.folder_name = 'test'  # 'explorer_x1_rgbd_traj/living_room_traj1_frei_png'
        self.rgb_path = os.path.join(rospkg.RosPack().get_path('gradslam_ros'),
                                     f'data/{self.folder_name}/rgb/')
        self.depth_path = os.path.join(rospkg.RosPack().get_path('gradslam_ros'),
                                       f'data/{self.folder_name}/depth/')
        self.tfs_path = os.path.join(rospkg.RosPack().get_path('gradslam_ros'),
                                     f'data/{self.folder_name}/livingRoom1n.gt.sim')
        self.assocs_path = os.path.join(rospkg.RosPack().get_path('gradslam_ros'),
                                        f'data/{self.folder_name}/associations.txt')
        self.image_n = 0
        self.save_data = rospy.get_param('~save_data', True)

        if not os.path.isdir(os.path.join(rospkg.RosPack().get_path('gradslam_ros'),
                                          f'data/{self.folder_name}')):
            os.makedirs(self.rgb_path)
            os.makedirs(self.depth_path)

        # Subscribe to topics
        # info_sub = message_filters.Subscriber('/X1/front_rgbd/optical/camera_info', CameraInfo)
        rgb_sub = message_filters.Subscriber('/X1/front_rgbd/optical/image_raw', Image)
        depth_sub = message_filters.Subscriber('/X1/front_rgbd/depth/optical/image_raw', Image)

        # Synchronize the topics by time
        ats = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=5, slop=0.1)
        ats.registerCallback(self.callback)

    def callback(self, rgb_msg, depth_msg):
        t0 = time()
        try:
            tf = self.tf.lookup_transform(self.world_frame, self.camera_frame,
                                          rospy.Time.now(), rospy.Duration.from_sec(1.0))
            rospy.logdebug('Found transform in %.3f sec', time()-t0)
        except tf2_ros.TransformException as ex:
            rospy.logerr('Could not transform from world %s to camera %s: %s.',
                         self.world_frame, self.camera_frame, ex)
            return
        try:
            # get rgb image
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, rgb_msg.encoding)
            # get depth image
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, depth_msg.encoding)
        except CvBridgeError as e:
            rospy.logerr(e)
            return
        T = numpify(tf.transform)
        assert rgb_image.shape[:2] == depth_image.shape
        assert T.shape == (4, 4)

        if self.save_data:
            # write images
            cv2.imwrite(self.rgb_path+str(self.image_n)+'.png', rgb_image)
            cv2.imwrite(self.depth_path + str(self.image_n) + '.png', depth_image.astype(np.uint16))

            # write associations
            with open(self.assocs_path, 'a') as f:
                f.write(str(self.image_n)+' depth/'+str(self.image_n)+'.png '+str(self.image_n)+' rgb/'+str(self.image_n)+'.png')
                f.write('\n')

            # write transformations
            with open(self.tfs_path, 'a') as f:
                for line in np.matrix(T[:3, :]):
                    np.savetxt(f, line, fmt='%.2f')
                f.write('\n')

            self.image_n += 1
            rospy.loginfo('Writing took: %.3f sec', time() - t0)


if __name__ == '__main__':
    rospy.init_node('bag2data', log_level=rospy.DEBUG)
    ip = Processor()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
