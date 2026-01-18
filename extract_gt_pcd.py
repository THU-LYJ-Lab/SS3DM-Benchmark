import os
import sys
def set_env(depth: int):
    # Add project root to sys.path
    current_file_path = os.path.abspath(__file__)
    project_root_path = os.path.dirname(current_file_path)
    for _ in range(depth):
        project_root_path = os.path.dirname(project_root_path)
    if project_root_path not in sys.path:
        sys.path.append(project_root_path)
        print(f"Added {project_root_path} to sys.path")
set_env(2)

import numpy as np
import open3d as o3d
import sklearn.neighbors as skln
import argparse
from glob import glob
import os
import pickle
import trimesh

import logging

from utils.evaluate_utils import resize_cam_intrin, generate_rays 

import cv2

from tqdm import tqdm


def clean_invisible_faces(pose_all, intrinsic, vertices, triangles, number_of_points=2048):
    mesh = trimesh.Trimesh(vertices, triangles)
    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    
    reso_level = 4.0
    # import pdb; pdb.set_trace()
    target_img_size = (int(intrinsic[0,2]*2), int(intrinsic[1,2]*2))
    
    
    intrin = resize_cam_intrin(intrinsic, resolution_level=reso_level)
    if reso_level > 1.0:
        W = int(target_img_size[0] // reso_level)
        H = int(target_img_size[1] // reso_level)
        target_img_size = (W,H)
    
    all_indices = []
    # import pdb; pdb.set_trace()  
    for pose in pose_all:
        pose_np = pose
        # import pdb; pdb.set_trace()
        rays_o, rays_d = generate_rays(target_img_size, intrin, pose)
        rays_o = rays_o.reshape(-1,3)
        rays_d = rays_d.reshape(-1,3)

        idx_faces_hits = intersector.intersects_first(rays_o, rays_d)
        all_indices.append(idx_faces_hits)

    values = np.unique(np.array(all_indices)) 
    mask_faces = np.ones(len(mesh.faces))
    mask_faces[values[1:]] = 0
    logging.info(f'Surfaces/Kept: {len(mesh.faces)}/{len(values)}')
    
    ## create open3d triangle mesh
    mesh_o3d = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))

    mesh_o3d.remove_triangles_by_mask(mask_faces)
    print(f'Before cleaning: {len(mesh_o3d.vertices)}')
    # mesh_o3d.remove_triangles_by_mask(mask_faces)
    mesh_o3d.remove_unreferenced_vertices()
    print(f'After cleaning: {len(mesh_o3d.vertices)}')
    
    
    pointcloud_o3d = o3d.geometry.PointCloud()
    pointcloud_o3d.points = mesh_o3d.vertices
    pointcloud_o3d = pointcloud_o3d.voxel_down_sample(voxel_size=0.05)

    return pointcloud_o3d, mesh_o3d


def depth_to_points(depth, intrinsic_mat, pose, mask, h=1080, w=1920):
    grids = np.meshgrid(np.arange(w), np.arange(h))
    grids = np.stack(grids, axis=-1)
    grids = grids.reshape(-1, 2)
    grids = np.concatenate([grids, np.ones((grids.shape[0], 1))], axis=-1)
    grids = grids.astype(np.float32)
    grids = np.matmul(np.linalg.inv(intrinsic_mat), grids.T).T

    mask = mask.reshape(-1, 1)[:,0]
    camera_points = grids[mask] * depth.reshape(-1, 1)[mask]
    camera_points = camera_points[depth.reshape(-1)[mask.reshape(-1)] < 500, :]
    camera_points = np.concatenate([camera_points, np.ones((camera_points.shape[0], 1))], axis=-1)
    
    world_points = np.matmul(pose, camera_points.T).T
    
    return world_points[:, :3]
    
    
    
def depth_fusion(poses_all, depths_all, masks_all, intrinsic_mat, h=1080, w=1920):
    all_points_o3d = o3d.geometry.PointCloud()
    idx = 0
    for pose, depth, mask in tqdm(zip(poses_all, depths_all, masks_all)):
        pose_np = pose
        depth_np = depth
        points = depth_to_points(depth_np, intrinsic_mat, pose_np, mask, h=1080, w=1920)
        # points = points[mask.reshape(-1, 1)[:,0], :]
        
        # append the points to the all_points
        points_o3d = o3d.geometry.PointCloud()
        points_o3d.points = o3d.utility.Vector3dVector(points)
        points_o3d = points_o3d.voxel_down_sample(voxel_size=0.01)
        all_points_o3d += points_o3d
        
        # downsample the points with open3d
        # import pdb; pdb.set_trace()
        if idx%5==0:
            all_points_o3d = all_points_o3d.voxel_down_sample(voxel_size=0.5)
        
        idx += 1
        # all_points.append(points)
    
    # all_points_o3d = o3d.geometry.PointCloud()
    # all_points_o3d.points = o3d.utility.Vector3dVector(np.concatenate(all_points, axis=0))
    # import pdb; pdb.set_trace()
    all_points_o3d = all_points_o3d.voxel_down_sample(voxel_size=0.5)
    
    return all_points_o3d

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/data/huyb/cvpr-2024/data/ss3dm/DATA', help='input data')
    args = parser.parse_args()

    data_root = args.data_root

    for town_name in os.listdir(data_root):
    # for town_name in ['Town02']:
        if os.path.isdir(os.path.join(data_root, town_name)):
            town_dir = os.path.join(data_root, town_name)
            for seq_name in os.listdir(town_dir):
            # for seq_name in ['Town02_150']:
                if os.path.isdir(os.path.join(town_dir, seq_name)):
                    seq_dir = os.path.join(town_dir, seq_name)
                    
                    print("Processing {} ...".format(seq_dir))
                    
                    gt_path = os.path.join(town_dir, f'{town_name}_obj.obj')
                    meta_path = os.path.join(seq_dir, 'scenario.pt')
                    depth_gt_dir = os.path.join(seq_dir, 'depth_gts')
                    pcd_save_path = seq_dir
    
                    ## load the meta data with pickle
                    with open(meta_path, 'rb') as f:
                        meta_data = pickle.load(f)
                    
                    pose_all = []
                    depth_selected = []
                    pose_selected = []
                    mask_selected = []
                    
                    print("Loading the poses and depth images ...")  
                    for observer in meta_data['observers'].keys():
                        if meta_data['observers'][observer]['class_name'] == 'Camera':
                            
                            for i in range(meta_data['metas']['n_frames']):
                                pose_all.append(meta_data['observers'][observer]['data']['c2w'][i].astype(np.float32))
                                
                                intrinsic_mat = meta_data['observers'][observer]['data']['intr'][i].astype(np.float32)
                                
                                # import pdb; pdb.set_trace()
                                if len(pose_all)%50==0:
                                # if len(pose_all) < 2:
                                    image = cv2.imread(os.path.join(depth_gt_dir, observer, '%.8d.png' % (i)), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                                    image = image.astype(np.float32) / 65535 * 1000.0
                                    depth_selected.append(image)
                                    pose_selected.append(meta_data['observers'][observer]['data']['c2w'][i].astype(np.float32))
                                    
                                    this_mask_image = np.load(os.path.join(data_root, town_name, seq_name, 'masks', observer, '%.8d.npz' % (i)))['arr_0']
                                    valid_mask = this_mask_image!=11
                                    mask_selected.append(valid_mask)
                    
                    if not os.path.exists(os.path.join(pcd_save_path, 'gt_point_cloud_from_depth_fusion.ply')):
                    # if True:
                        ## select the points that are close to the depth-fusion results.
                        print("Fusing the depth images ...")
                        gt_point_cloud_from_depth_fusion = depth_fusion(pose_selected, depth_selected, mask_selected, intrinsic_mat)
                        
                        print("Saving the fused point cloud ...")
                        o3d.io.write_point_cloud(os.path.join(pcd_save_path, 'gt_point_cloud_from_depth_fusion.ply'), gt_point_cloud_from_depth_fusion)
                    else:
                        gt_point_cloud_from_depth_fusion = o3d.io.read_point_cloud(os.path.join(pcd_save_path, 'gt_point_cloud_from_depth_fusion.ply'))
                    
                    depth_fusion_tree = skln.KDTree(np.asarray(gt_point_cloud_from_depth_fusion.points))
                    
                    
                    if not os.path.exists(os.path.join(pcd_save_path, 'cleaned_gt_mesh.ply')):
                        print("Cleaning the ground truth mesh ...")
                        obj_path = gt_path
                        mesh_gt = o3d.io.read_triangle_mesh(obj_path)
                        
                        ##  scale the mesh_gt with 0.01, center is (0,0,0)
                        mesh_gt.scale(0.01, center=(0,0,0))
                            
                        vertices_gt, triangles_gt = np.asarray(mesh_gt.vertices), np.asarray(mesh_gt.triangles)
                        
                        _, mesh_o3d_gt = clean_invisible_faces(pose_all, intrinsic_mat, vertices_gt, triangles_gt)
                        
                        mesh_o3d_gt.compute_triangle_normals()
                        
                        print("Saving the cleaned ground truth mesh ...")
                        o3d.io.write_triangle_mesh(os.path.join(pcd_save_path, 'cleaned_gt_mesh.ply'), mesh_o3d_gt)
                    else:
                        mesh_o3d_gt = o3d.io.read_triangle_mesh(os.path.join(pcd_save_path, 'cleaned_gt_mesh.ply'))
                    
                    
                    print("Cleaning triangles which are far from the depth-fusion results ...")
                    # select triangles that are close to the depth-fusion results.
                    vertices_points = np.asarray(mesh_o3d_gt.vertices)
                    
                    # remove triangles that are far away from the depth-fusion results.
                    dist, _ = depth_fusion_tree.query(vertices_points)
                    mesh_o3d_gt.remove_vertices_by_mask(dist > 5)
                    
                    o3d.io.write_triangle_mesh(os.path.join(pcd_save_path, 'cleaned_gt_mesh.ply'), mesh_o3d_gt)
                    
                        
                    if not os.path.exists(os.path.join(pcd_save_path, 'cleaned_gt_point_cloud_dense.ply')):
                    # if True:
                        print("Sampling points on the cleaned ground truth mesh ...")
                        cleaned_point_cloud_gt = mesh_o3d_gt.sample_points_uniformly(204800*50, use_triangle_normal=True)
                        
                        print("Saving the sampled points on the cleaned ground truth mesh ...")
                        o3d.io.write_point_cloud(os.path.join(pcd_save_path, 'cleaned_gt_point_cloud_dense.ply'), cleaned_point_cloud_gt)
                    else:
                        cleaned_point_cloud_gt = o3d.io.read_point_cloud(os.path.join(pcd_save_path, 'cleaned_gt_point_cloud_dense.ply'))
                    
                    
                    
                    
                    if not os.path.exists(os.path.join(pcd_save_path, 'selected_gt_point_cloud_dense.ply')):     
                    # if True:                 
                        selected_point_cloud_gt = cleaned_point_cloud_gt
                        
                        print("Selecting the points that are close to the depth-fusion results ...")
                        dist, _ = depth_fusion_tree.query(np.asarray(selected_point_cloud_gt.points))
                        
                        selected_point_cloud_gt = selected_point_cloud_gt.select_by_index(np.where(dist < 1.5)[0])
                        
        
                        print("Saving the selected points on the cleaned ground truth mesh ...")
                        o3d.io.write_point_cloud(os.path.join(pcd_save_path, 'selected_gt_point_cloud_dense.ply'), selected_point_cloud_gt)
                        
                    else:
                        selected_point_cloud_gt = o3d.io.read_point_cloud(os.path.join(pcd_save_path, 'selected_gt_point_cloud_dense.ply'))
                    
                    
                    print("Orienting the normals towards the camera locations ...")
                    for pose in pose_all:
                        cam_loc = pose[:3, 3]
                        selected_point_cloud_gt.orient_normals_towards_camera_location(cam_loc)
                    o3d.io.write_point_cloud(os.path.join(pcd_save_path, 'selected_gt_point_cloud_dense.ply'), selected_point_cloud_gt)
                    
                    # resample the point cloud with voxel size 0.025
                    if not os.path.exists(os.path.join(pcd_save_path, 'selected_gt_point_cloud_dense_resampled.ply')):
                        selected_point_cloud_gt = selected_point_cloud_gt.voxel_down_sample(voxel_size=0.05)
                        o3d.io.write_point_cloud(os.path.join(pcd_save_path, 'selected_gt_point_cloud_dense_resampled.ply'), selected_point_cloud_gt)
                    else:
                        selected_point_cloud_gt = o3d.io.read_point_cloud(os.path.join(pcd_save_path, 'selected_gt_point_cloud_dense_resampled.ply'))
                    
                    # import pdb; pdb.set_trace()
