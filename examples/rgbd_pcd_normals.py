#!/usr/bin/env python

import open3d as o3d
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import imageio


path = "../ros_ws/src/gradslam_ros/data/explorer_x1_rgbd_traj/living_room_traj1_frei_png/"

ind = np.random.randint(575)
# ind = 12
color_img = imageio.imread(path+f"rgb/{ind}.png")
# depth_img = np.load(path+f"depth/{ind}.npy")

color_raw = o3d.io.read_image(path+f"rgb/{ind}.png")
depth_raw = o3d.io.read_image(path+f"depth/{ind}.png")
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('grayscale image')
plt.imshow(rgbd_image.color, cmap="gray")
plt.subplot(1, 2, 2)
plt.title('depth image')
plt.imshow(rgbd_image.depth, cmap="gray")
plt.show()

K = np.load(path+f"caminfo/{ind}.npy")
intrinsics = o3d.camera.PinholeCameraIntrinsic(width=color_img.shape[0], height=color_img.shape[1],
                                               fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])

# Create point cloud from RGBD image
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
# Flip it, otherwise the pointcloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
# o3d.visualization.draw_geometries([pcd])
print(pcd)

# Estimate Normals from Points
print("Downsample the point cloud with a voxel of 0.05")
downpcd = pcd.voxel_down_sample(voxel_size=0.05)
# o3d.visualization.draw_geometries([downpcd])
print(downpcd)


# Open3D can be used to estimate point cloud normals with `estimate_normals`,
# which locally fits a plane per 3D point to derive the normal.
downpcd.estimate_normals()
# o3d.visualization.draw_geometries([downpcd], point_show_normal=True)

# However, the estimated normals might not be consistently oriented.
# `orient_normals_consistent_tangent_plane` propagates the normal orientation using a minimum spanning tree.
downpcd.orient_normals_consistent_tangent_plane(100)
o3d.visualization.draw_geometries([downpcd], point_show_normal=True)
