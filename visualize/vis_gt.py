
import os
import numpy as np
import torch
import sys

import open3d as o3d
# o3d.visualization.webrtc_server.enable_webrtc()

def get_list(split, path, subsets, category_dict, data_percentage, max_imgs):
    data_list = []
    for subset in subsets:
        for cat in category_dict[subset]:
            list_fname = f"{path}/{subset}/lists/{cat}_{split}.list"
            if not os.path.exists(list_fname):
                continue
            lines = open(list_fname).read().splitlines()
            lines = lines[:round(data_percentage*len(lines))]
            for i, img_fname in enumerate(lines):
                if i >= max_imgs: break
                name = '.'.join(img_fname.split('.')[:-1])
                object_name = name.split('_')[-2]
                sample_id = name.split('_')[-1]
                data_list.append((subset, cat, object_name, sample_id))
    return data_list

def get_pointcloud(path, subset, category, object_name):
    fname = f"{category}/{category}_{object_name}"
    pc_fname = f"{path}/{subset}/pointclouds/{fname}.npy"
    pc = np.load(pc_fname)
    dpc = {"points":torch.from_numpy(pc).float()}
    return dpc
    
def get_camera(subset, category, object_name, sample_id):
    fname = f"{category}/{category}_{object_name}_{sample_id}"
    intr_p = f"{path}/{subset}/camera_data/intr/{fname}.npy"
    extr_p = f"{path}/{subset}/camera_data/extr/{fname}.npy"
    Rt = np.load(extr_p)
    K = torch.from_numpy(np.load(intr_p))
    return K, Rt

def get_gt_sdf(path, subset, category, object_name):
    fname = f"{category}/{category}_{object_name}"
    gt_fname = f"{path}/{subset}/gt_sdf/{fname}.npy"
    gt_dict = np.load(gt_fname, allow_pickle=True).item()
    gt_sample_points = torch.from_numpy(gt_dict['sample_pt']).float()
    gt_sample_sdf = torch.from_numpy(gt_dict['sample_sdf']).float() - 0.003
    return gt_sample_points, gt_sample_sdf

if __name__ == "__main__":

    o3d.visualization.webrtc_server.enable_webrtc()

    sys.path.append('/home/zubairirshad/ZeroShape')
    import utils.camera as camera

    path = '/home/zubairirshad/Downloads/train_data'
    category_dict = {}
    category_list = []

    max_imgs = 10
    data_percentage = 1

    subsets = ['objaverse_LVIS_tiny']
    for subset in subsets:
        subset_path = "{}/{}".format(path, subset)
        categories = [name[:-11] for name in os.listdir("{}/lists".format(subset_path)) if name.endswith("_train.list")]
        category_dict[subset] = categories
        category_list += [cat for cat in categories]

    lists_all = get_list('train', path, subsets, category_dict, data_percentage, max_imgs)

    print("lists_all", lists_all[:10])

    list = lists_all[:-10]

    # idx = np.random.choice(len(lists_all))
    idx = 0
    subset, category, object_name, sample_id = lists_all[idx]

    print("subset, category, object_name, sample_id", subset, category, object_name, sample_id)

    #Get camera pose

    # load camera
    K, Rt = get_camera(subset, category, object_name, sample_id)
    R = np.zeros((3,4))
    R[:3,:3] = Rt[:3,:3]
    t = Rt[:3,3]
    t = camera.pose(t=t)
    pose = camera.pose.compose([R, t])

    print("K", K)
    print("pose", pose)

    # load point cloud
    dpc = get_pointcloud(path, subset, category, object_name)

    # load gt sdf
    gt_sample_points, gt_sample_sdf = get_gt_sdf(path, subset, category, object_name)

    # print("dpc", dpc.shape)
    print("dpc", dpc['points'].shape)
    print("gt_sample_points", gt_sample_points.shape)
    print("gt_sample_sdf", gt_sample_sdf.shape)

    #visualize point cloud in open3d

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(dpc['points'].numpy())

    #visualize SDF with colored points <0 should be red and >0 should be blue

    colors = np.zeros((gt_sample_sdf.shape[0], 3))
    colors[gt_sample_sdf < 0] = [1, 0, 0]
    colors[gt_sample_sdf > 0] = [0, 0, 1]



    #transform gt sample points to world frame
    R_gt = pose[:, :3]
    # [B, 3, 1]
    T_gt = pose[:, 3:]
    # [B, 3, N]
    gt_sample_points_transposed = gt_sample_points.permute(1, 0).contiguous()
    # camera coordinates, [B, N, 3]
    gt_sample_points_cam = (R_gt @ gt_sample_points_transposed + T_gt).permute(1,0).contiguous()
    # normalize with seen std and mean, [B, N, 3]
    # var.gt_points_cam = (gt_sample_points_cam - seen_points_mean_gt.unsqueeze(1)) / seen_points_scale_gt.unsqueeze(-1).unsqueeze(-1)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(gt_sample_points_cam.numpy())
    pcd.colors = o3d.utility.Vector3dVector(colors)

    #draw coordiante frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])

    # o3d.visualization.draw_geometries([pcd])

    draw = o3d.visualization.EV.draw

    #no
    draw([ {'geometry': pcd, 'name': 'pcd'},{'geometry': coord_frame, 'name': 'coord_frame'}  ])
