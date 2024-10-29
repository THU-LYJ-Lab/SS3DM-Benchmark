# SS3DM-Benchmark

## Evaluation

### Evaluate the meshes

We provide the example evaluation scripts for the methods mentioned in our paper.

Evaluate meshes produced by StreetSurf.
```
python evaluate_mesh.py --exp_dir /data/huyb/cvpr-2024/neuralsim/logs/ss3dm/  --method streetsurf --box --resample
```

Evaluate meshes produced by UrbanNerf.
```
python evaluate_mesh.py --exp_dir /data/huyb/cvpr-2024/neuralsim/logs/ss3dm --method urban_nerf --box --resample
```

Evaluate meshes produced by SuGaR.
```
python evaluate_mesh.py --exp_dir /data/huyb/cvpr-2024/SuGaR/output/refined_mesh_flip --method sugar --box --resample
```

Evaluate meshes produced by NeRF-LOAM
```
python evaluate_mesh.py --exp_dir /data/huyb/cvpr-2024/NeRF-LOAM/logs/ss3dm --method nerf_loam --box --resample
```

Evaluate meshes produced by R3D3
```
python evaluate_mesh.py --exp_dir /data/huyb/cvpr-2024/r3d3/logs/ddad_tiny/eval_predictions --method r3d3 --box --resample
```

### Collect the metrics

We also include an example script to collect evaluation results and form a latex table
```
python collect_results.py --box --resample --plt_curve
```

## Run the existing methods

### StreetSurf

You can use this script to predict the meshes. [[neuralsim/code_single/tools/train_for_ss3dm.py]](https://github.com/AlbertHuyb/neuralsim/blob/main/code_single/tools/train_for_ss3dm.py)

### UrbanNeRF

You can use this script to predict the meshes. [[neuralsim/code_single/tools/train_for_ss3dm_urban_nerf.py]](https://github.com/AlbertHuyb/neuralsim/blob/main/code_single/tools/train_for_ss3dm_urban_nerf.py)

### SuGaR

You can use this script to train and extract mesh models. [[SuGaR/train_ss3dm.py]](https://github.com/AlbertHuyb/SuGaR/blob/main/train_ss3dm.py) 

The produced meshes should be flipped by this script. [[SuGaR/convert_mesh.py]](https://github.com/AlbertHuyb/SuGaR/blob/main/convert_mesh.py)

### NeRF-LOAM

You can use this script to train the models. [[NeRF-LOAM/demo/train_for_ss3dm_nerf_loam.py]](https://github.com/AlbertHuyb/NeRF-LOAM/blob/master/demo/train_for_ss3dm_nerf_loam.py) 

The produced meshes should be post-processed by this script.  [[NeRF-LOAM/demo/post_process_for_ss3dm_nerf_loam.py]](https://github.com/AlbertHuyb/NeRF-LOAM/blob/master/demo/post_process_for_ss3dm_nerf_loam.py)

### R3D3
You can use this script to predict the depth maps. [[r3d3/evaluate.sh]](https://github.com/AlbertHuyb/r3d3/blob/master/evaluate.sh) 

The produced depth maps should be post-processed by these scripts. [[r3d3/tools/fuse_depth_to_pointcloud.py]](https://github.com/AlbertHuyb/r3d3/blob/master/tools/fuse_depth_to_pointcloud.py), [[r3d3/tools/surface_extraction.py]](https://github.com/AlbertHuyb/r3d3/blob/master/tools/surface_extraction.py)

After the post-processing step, the results would be converted to predicted mesh surfaces.