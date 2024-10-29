# SS3DM-Benchmark

## Evaluation
1. Download the predicted meshes and sampled points.
2. 


Evaluate SuGaR meshes.
```
python neuralsim/code_single/tools/evaluate_for_ss3dm_dense.py --exp_dir /data/huyb/cvpr-2024/SuGaR/output/refined_mesh_flip --method sugar --box --resample
```
Evaluate streetsurf.
```
python neuralsim/code_single/tools/evaluate_for_ss3dm_dense.py --exp_dir /data/huyb/cvpr-2024/neuralsim/logs/ss3dm/  --method streetsurf --box --resample
```
Evaluate urban_nerf.
```
python neuralsim/code_single/tools/evaluate_for_ss3dm_dense.py --exp_dir /data/huyb/cvpr-2024/neuralsim/logs/ss3dm --method urban_nerf --box --resample
```
Evaluate R3D3
```
python neuralsim/code_single/tools/evaluate_for_ss3dm_dense.py --exp_dir /data/huyb/cvpr-2024/r3d3/logs/ddad_tiny/eval_predictions --method r3d3 --box --resample
```
Evaluate NeRF-LOAM
```
python neuralsim/code_single/tools/evaluate_for_ss3dm_dense.py --exp_dir /data/huyb/cvpr-2024/NeRF-LOAM/logs/ss3dm --method nerf_loam --box --resample
```

## Run the existing methods

### StreetSurf

You can use [this script](neuralsim/code_single/tools/train_for_ss3dm.py)

### UrbanNeRF

You can use [this script](neuralsim/code_single/tools/train_for_ss3dm_urban_nerf.py)

### SuGaR

You can use [this script](SuGaR/train_ss3dm.py)