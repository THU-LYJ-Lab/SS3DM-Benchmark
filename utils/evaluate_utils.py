import os
import pickle
import logging
import torch
import numpy as np
import open3d as o3d
import trimesh
import sklearn.neighbors as skln

def read_meta_data(seq_dir):
    meta_path = os.path.join(seq_dir, 'scenario.pt')
    with open(meta_path, 'rb') as f:
        meta_data = pickle.load(f)

    pose_all = []

    for observer in meta_data['observers'].keys():
        if meta_data['observers'][observer]['class_name'] == 'Camera':
            
            for i in range(meta_data['metas']['n_frames']):
                pose_all.append(meta_data['observers'][observer]['data']['c2w'][i].astype(np.float32))
                
                intrinsic_mat = meta_data['observers'][observer]['data']['intr'][i].astype(np.float32)
    return pose_all, intrinsic_mat

def resize_cam_intrin(intrin, resolution_level):
    if resolution_level != 1.0:
        logging.info(f'Resize instrinsics, resolution_level: {resolution_level}')
        intrin[:2,:3] /= resolution_level
    return intrin

def generate_rays(img_size, intrin, pose = None, normalize_dir = True):
    '''Generate rays with specified size, intrin and pose.
    Args:
        intrin: 4*4
        pose: 4*4, (default: None, identity), camera to world
    Return:
        rays_o, rays_d: H*W*3, numpy array
    '''
    if pose is None:
        pose = np.identity(4)
    pose = torch.tensor(pose).cuda()
    intrin = torch.tensor(intrin).cuda()

    W,H = img_size
    tu = torch.linspace(0, W - 1, W).cuda()
    tv = torch.linspace(0, H - 1, H).cuda()
    pixels_v, pixels_u = torch.meshgrid(tv, tu)
    p = torch.stack([pixels_u, pixels_v, torch.ones_like(pixels_v )], dim=-1) # W, H, 3
    p = torch.matmul(torch.from_numpy(np.linalg.inv(intrin.cpu().numpy())).cuda()[None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
    if normalize_dir:
        # for volume rendering, depth along ray
        rays_v = p  / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)
    else:
        # for reprojection of depthmap, depth along z axis
        rays_v = p
    rays_v = torch.matmul(pose[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()
    rays_o = pose[None, None, :3, 3].expand(rays_v.shape) 
    return rays_o.cpu().numpy(), rays_v.cpu().numpy()

def calculate_IoU(pcd_gt_np, pcd_pred, threshold):
    # assign pcd to voxel grids
    pcd_gt_grid = np.floor(pcd_gt_np / threshold)
    pcd_pred_grid = np.floor(pcd_pred / threshold)

    # remove the repeated occupied voxels
    pcd_gt_grid = np.unique(pcd_gt_grid, axis=0)
    pcd_pred_grid = np.unique(pcd_pred_grid, axis=0)

    # compute intersection and union
    pcd_gt_grid_map = list(map(tuple, pcd_gt_grid))
    pcd_pred_grid_map = list(map(tuple, pcd_pred_grid))
    intersection = list(set(pcd_gt_grid_map).intersection(set(pcd_pred_grid_map)))
    union = list(set(pcd_gt_grid_map).union(set(pcd_pred_grid_map)))

    # compute IoU
    IoU = len(intersection) / len(union)
    
    return IoU

def calculate_metrics(pcd_gt_o3d, pcd_pred_o3d, max_dist=2, f_score_thresholds=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.20, 0.10, 0.05]):
    '''
    Both pcd_gt and pcd_pred are open3d.geometry.PointCloud.
    Suppose the point clouds contain normalized surface normals.
    
    return the accuracy, completeness, chamfer distance for distance < max_dist.
    
    also return the f-score with f_score_threshold.
    '''
    metrics = {}
    
    pcd_gt = np.asarray(pcd_gt_o3d.points)
    pcd_pred = np.asarray(pcd_pred_o3d.points)
    
    pcd_gt_tree = skln.KDTree(pcd_gt)
    pcd_pred_tree = skln.KDTree(pcd_pred)
    dist_gt, idx_gt = pcd_gt_tree.query(pcd_pred)
    dist_pred, idx_pred = pcd_pred_tree.query(pcd_gt)
    
    accuracy = np.mean(dist_gt[dist_gt < max_dist ])
    completeness = np.mean(dist_pred[dist_pred < max_dist ])
    chamfer_dist = accuracy + completeness
    
    # import pdb; pdb.set_trace()
    normals_gt = np.asarray(pcd_gt_o3d.normals)[idx_gt]
    normal_accuracy = 1.0 - np.mean(np.sum(np.multiply(normals_gt, np.asarray(pcd_pred_o3d.normals)[:,None,:]), axis=-1)[dist_gt < max_dist ])
    
    normals_pred = np.asarray(pcd_pred_o3d.normals)[idx_pred]
    normal_completeness = 1.0 - np.mean(np.sum(normals_pred * np.asarray(pcd_gt_o3d.normals)[:,None,:], axis=-1)[dist_pred < max_dist ])
    
    normal_chamfer_dist = normal_accuracy + normal_completeness
    
    for f_score_threshold in f_score_thresholds:
        precision = np.sum(dist_gt[dist_gt < max_dist ] < f_score_threshold) / len(dist_gt[dist_gt < max_dist ])
        recall = np.sum(dist_pred[dist_pred < max_dist ] < f_score_threshold) / len(dist_pred[dist_pred < max_dist ])
        f_score = 2 * precision * recall / (precision + recall)
        
        metrics[f'f_score({f_score_threshold:.2f}m)'] = {}
        metrics[f'f_score({f_score_threshold:.2f}m)']['precision'] = precision
        metrics[f'f_score({f_score_threshold:.2f}m)']['recall'] = recall
        metrics[f'f_score({f_score_threshold:.2f}m)']['f_score'] = f_score
        
        IoU = calculate_IoU(pcd_gt, pcd_pred, f_score_threshold)
        metrics[f'IoU({f_score_threshold:.2f}m)']['IoU'] = IoU
    
    metrics['chamfer_dist(m)'] = {}
    metrics['chamfer_dist(m)']['accuracy'] = accuracy
    metrics['chamfer_dist(m)']['completeness'] = completeness
    metrics['chamfer_dist(m)']['chamfer_dist'] = chamfer_dist
    
    metrics['normal_chamfer_dist(cosine)'] = {}
    metrics['normal_chamfer_dist(cosine)']['normal_accuracy'] = normal_accuracy
    metrics['normal_chamfer_dist(cosine)']['normal_completeness'] = normal_completeness
    metrics['normal_chamfer_dist(cosine)']['normal_chamfer_dist'] = normal_chamfer_dist
    
    return metrics
    
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

