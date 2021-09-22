from argparse import ArgumentParser, RawTextHelpFormatter

import open3d as o3d
import torch
from torch.utils.data import DataLoader

from gradslam.datasets.icl import ICL
from gradslam.datasets.tum import TUM
from gradslam.slam.pointfusion import PointFusion
from gradslam import Pointclouds, RGBDImages
from time import time
from tqdm import tqdm


dataset_path = './tutorials/TUM/'
seqlen = 100
odometry = 'gradicp'

if __name__ == "__main__":
    # select device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # load dataset
    t0 = time()
    dataset = TUM(dataset_path, seqlen=seqlen, height=60, width=80)
    loader = DataLoader(dataset=dataset, batch_size=16)
    colors, depths, intrinsics, poses, *_ = next(iter(loader))

    # create rgbdimages object
    rgbdimages = RGBDImages(colors, depths, intrinsics, poses, channels_first=False)
    print(f'Data preparation took {time()-t0:.3f} sec')

    # SLAM: whole sequence
    t0 = time()
    slam = PointFusion(odom=odometry, dsratio=1, device=device)
    pointclouds, recovered_poses = slam(rgbdimages)
    print(f'SLAM procesing took {time()-t0:.3f} sec')

    # SLAM: step by step
    # t0 = time()
    # slam = PointFusion(odom=odometry, dsratio=1, device=device)
    # pointclouds = Pointclouds(device=device)
    # batch_size, seq_len = rgbdimages.shape[:2]
    # initial_poses = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(batch_size, 1, 1, 1)
    # prev_frame = None
    # for s in tqdm(range(seq_len)):
    #     t1 = time()
    #     live_frame = rgbdimages[:, s].to(device)
    #     if s == 0 and live_frame.poses is None:
    #         live_frame.poses = initial_poses
    #     pointclouds, live_frame.poses = slam.step(pointclouds, live_frame, prev_frame)
    #     print(f'SLAM step took {(time() - t1):.3f} sec')
    #     prev_frame = live_frame
    # print(f'SLAM processing took {time() - t0:.3f} sec')

    # visualization
    o3d.visualization.draw_geometries([pointclouds.open3d(0)])
