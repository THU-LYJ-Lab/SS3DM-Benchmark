import os
import sys
# def set_env(depth: int):
#     # Add project root to sys.path
#     current_file_path = os.path.abspath(__file__)
#     project_root_path = os.path.dirname(current_file_path)
#     for _ in range(depth):
#         project_root_path = os.path.dirname(project_root_path)
#     if project_root_path not in sys.path:
#         sys.path.append(project_root_path)
#         print(f"Added {project_root_path} to sys.path")
# set_env(2)

import numpy as np
import open3d as o3d

import argparse
import glob
from glob import glob
import os
from utils.evaluate_utils import read_meta_data, calculate_metrics, clean_invisible_faces

def get_seq_exp_dir(exp_root, method_name, town_name, seq_name):
    if method_name == 'streetsurf' or method_name == 'urban_nerf':
        seq_exp_dir = os.path.join(exp_root, seq_name)
    elif method_name == 'r3d3':
        seq_exp_dir = os.path.join(exp_root, town_name, seq_name)
    elif method_name == 'nerf_loam' or method_name == 'sugar':
        seq_exp_dir = os.path.join(exp_root, seq_name)
    
    if not os.path.exists(seq_exp_dir):
        print(seq_exp_dir   )
        raise(FileNotFoundError, "No exp dir for {}".format(seq_name))

    if method_name == 'streetsurf' or method_name == 'urban_nerf':
        config_list = os.listdir(seq_exp_dir)
    elif method_name == 'r3d3' or method_name == 'sugar':
        config_list = ['']
    elif method_name == 'nerf_loam':
        config_list = os.listdir(seq_exp_dir)
                        
    return seq_exp_dir, config_list

def get_pred_mesh_path(this_config_dir, method_name):
    if method_name == 'r3d3':
        pred_mesh_path = glob.glob(os.path.join(this_config_dir, 'mesh.ply*'))
        if not os.path.exists(os.path.join(this_config_dir, 'meshes')):
            os.makedirs(os.path.join(this_config_dir, 'meshes'))
    elif method_name == 'streetsurf' or method_name == 'urban_nerf':
        pred_mesh_path = glob.glob(os.path.join(this_config_dir, 'meshes', '*.ply'))
    elif method_name == 'nerf_loam':
        pred_mesh_path = glob.glob(os.path.join(this_config_dir, 'mesh', 'final_mesh_transformed.ply'))
        if not os.path.exists(os.path.join(this_config_dir, 'meshes')):
            os.makedirs(os.path.join(this_config_dir, 'meshes'))
    elif method_name == 'sugar':
        pred_mesh_path = glob.glob(os.path.join(this_config_dir, 'sugarfine_3Dgs7000_sdfestim02_sdfnorm02_level03_decim1000000_normalconsistency01_gaussperface1.obj'))
        
        if not os.path.exists(os.path.join(this_config_dir, 'meshes')):
            os.makedirs(os.path.join(this_config_dir, 'meshes'))
    
    if len(pred_mesh_path) == 0:
        raise(FileNotFoundError, "No mesh found for {}".format(this_config_dir))
    else:
        pred_mesh_path = pred_mesh_path[0]
        print("Processing {} ...".format(seq_name))
           
    return pred_mesh_path, os.path.join(this_config_dir, 'meshes')

def get_json_output_path(this_config_dir, args):
    if args.box:
        json_path = os.path.join(this_config_dir, 'metrics_dense_box.json')
    else:
        json_path = os.path.join(this_config_dir, 'metrics_dense.json')
        
    if args.resample:
        json_path = json_path.replace('.json', '_resampled.json')
        
    return json_path


def evaluate(this_config_dir, method_name, json_path, pose_all, intrinsic_mat, overwrite=False):
    pred_mesh_path, pcd_save_path = get_pred_mesh_path(this_config_dir, method_name)
       
    if (not overwrite) and os.path.exists(json_path):
        print("Metrics already exist for {}".format(this_config_dir))
        with open(json_path, 'r') as f:
            metrics = json.load(f)
        return metrics
        
    ## 1. clean invisible faces
    if not os.path.exists(os.path.join(pcd_save_path, 'cleaned_pred_mesh.ply')):
        mesh_pred = o3d.io.read_triangle_mesh(pred_mesh_path)
        
        vertices_pred, triangles_pred = np.asarray(mesh_pred.vertices), np.asarray(mesh_pred.triangles)
        
        _, mesh_o3d_pred = clean_invisible_faces(pose_all, intrinsic_mat, vertices_pred, triangles_pred)
        
        mesh_o3d_pred.compute_triangle_normals()
        
        o3d.io.write_triangle_mesh(os.path.join(pcd_save_path, 'cleaned_pred_mesh.ply'), mesh_o3d_pred)
    else:
        mesh_o3d_pred = o3d.io.read_triangle_mesh(os.path.join(pcd_save_path, 'cleaned_pred_mesh.ply'))
    
    ## 2. sample N points from the mesh, N=204800*15
    if not os.path.exists(os.path.join(pcd_save_path, 'cleaned_pred_point_cloud.ply')): 
        cleaned_point_cloud_pred = mesh_o3d_pred.sample_points_poisson_disk(204800*15, use_triangle_normal=True)
        
        o3d.io.write_point_cloud(os.path.join(pcd_save_path, 'cleaned_pred_point_cloud.ply'), cleaned_point_cloud_pred)
    else:
        cleaned_point_cloud_pred = o3d.io.read_point_cloud(os.path.join(pcd_save_path, 'cleaned_pred_point_cloud.ply'))

    ## 3. load the ground truth point cloud
    if args.resample:
        selected_point_cloud_gt = o3d.io.read_point_cloud(os.path.join(seq_dir, 'selected_gt_point_cloud_dense_resampled.ply'))
    else:
        selected_point_cloud_gt = o3d.io.read_point_cloud(os.path.join(seq_dir, 'selected_gt_point_cloud_dense.ply'))

    ## 4. orient normals towards camera locations
    for pose in pose_all:
        cam_loc = pose[:3, 3]
        cleaned_point_cloud_pred.orient_normals_towards_camera_location(cam_loc)
    
    ## 5. crop the point cloud
    if args.box:
        ## calculate the bbox of camera locations
        all_locations = []
        for pose in pose_all:
            all_locations.append(pose[:3, 3])
        all_locations = np.array(all_locations)
        min_loc = np.min(all_locations, axis=0)
        max_loc = np.max(all_locations, axis=0)
        
        ## enlarge the bbox, 50m in each direction
        min_loc -= 50//2
        max_loc += 50//2
        
        # import pdb; pdb.set_trace()
        
        if not os.path.exists(os.path.join(pcd_save_path, 'cleaned_pred_point_cloud_box.ply')): 
            cleaned_point_cloud_pred = cleaned_point_cloud_pred.crop(o3d.geometry.AxisAlignedBoundingBox(min_loc, max_loc))
            
            o3d.io.write_point_cloud(os.path.join(pcd_save_path, 'cleaned_pred_point_cloud_box.ply'), cleaned_point_cloud_pred) 
        else:
            cleaned_point_cloud_pred = o3d.io.read_point_cloud(os.path.join(pcd_save_path, 'cleaned_pred_point_cloud_box.ply'))
        
        if not os.path.exists(os.path.join(pcd_save_path, 'selected_gt_point_cloud_box.ply')):  
            selected_point_cloud_gt = selected_point_cloud_gt.crop(o3d.geometry.AxisAlignedBoundingBox(min_loc, max_loc))
            
            o3d.io.write_point_cloud(os.path.join(pcd_save_path, 'selected_gt_point_cloud_box.ply'), selected_point_cloud_gt)
        else:
            selected_point_cloud_gt = o3d.io.read_point_cloud(os.path.join(pcd_save_path, 'selected_gt_point_cloud_box.ply'))
        
    ## 6. resample the point cloud with voxel size 0.05   
    if args.resample:
        if not os.path.exists(os.path.join(pcd_save_path, 'cleaned_pred_point_cloud_resampled.ply')):
            ## resample the point cloud
            cleaned_point_cloud_pred = cleaned_point_cloud_pred.voxel_down_sample(voxel_size=0.05)
            
            o3d.io.write_point_cloud(os.path.join(pcd_save_path, 'cleaned_pred_point_cloud_resampled.ply'), cleaned_point_cloud_pred)
        else:
            cleaned_point_cloud_pred = o3d.io.read_point_cloud(os.path.join(pcd_save_path, 'cleaned_pred_point_cloud_resampled.ply'))
        
        if not os.path.exists(os.path.join(pcd_save_path, 'selected_gt_point_cloud_resampled.ply')):
            o3d.io.write_point_cloud(os.path.join(pcd_save_path, 'selected_gt_point_cloud_resampled.ply'), selected_point_cloud_gt)
    
    ## 7. calculate metrics
    metrics = calculate_metrics(selected_point_cloud_gt, cleaned_point_cloud_pred)
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/data/huyb/cvpr-2024/data/ss3dm/DATA', help='input data')
    parser.add_argument('--exp_dir', type=str, default='/data/huyb/cvpr-2024/neuralsim/logs/ss3dm', help='exp dir')
    parser.add_argument('--method', type=str, default='streetsurf', help='method name')
    parser.add_argument('--box', action='store_true', help='whether to use box')
    parser.add_argument('--resample', action='store_true', help='whether to resample the point cloud')
    args = parser.parse_args()
    
    data_root = args.data_root
    method_name = args.method
    town_list = os.listdir(data_root)
    town_list.sort()
    # import pdb; pdb.set_trace()
    for town_name in town_list:
        if os.path.isdir(os.path.join(data_root, town_name)):
            town_dir = os.path.join(data_root, town_name)
            for seq_name in os.listdir(town_dir):
                if os.path.isdir(os.path.join(town_dir, seq_name)):
                    seq_dir = os.path.join(town_dir, seq_name)
                    
                    pose_all, intrinsic_mat = read_meta_data(seq_dir)
                    
                
                    ## example: 
                    ## seq_exp_dir: neuralsim/logs/ss3dm/streetsurf/Town01_150
                    ## config_list: ['onlylidar_all_cameras', 'withmask_withlidar_withnormal_all_cameras', 'withmask_withnormal_all_cameras']
                    seq_exp_dir, config_list = get_seq_exp_dir(args.exp_dir, method_name, town_name, seq_name)

                    for config in config_list:
                        # import pdb; pdb.set_trace()
                        
                        this_config_dir = os.path.join(seq_exp_dir, config)
                        
                        json_path = get_json_output_path(this_config_dir, args)
                        
                        metrics = evaluate(this_config_dir=this_config_dir, method_name=method_name, json_path=json_path, pose_all=pose_all, intrinsic_mat=intrinsic_mat, overwrite=False)

                        print(metrics)
                        
                        # import pdb; pdb.set_trace()
                        
                        
                        import json
                        
                        with open(json_path, 'w') as f:
                            json.dump(metrics, f, indent=4)
                        
                        
                        # import pdb; pdb.set_trace()
    
    
    
