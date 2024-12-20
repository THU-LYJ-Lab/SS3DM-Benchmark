# SS3DM-Benchmark

[[`HomePage`](https://ss3dm.top/)][`Paper`](https://arxiv.org/abs/2410.21739)][`Example Data`: [`GoogleDrive`](https://drive.google.com/drive/folders/1-5wfhVv4L6UMMqqED741I3HnEviOGsG7) [`HuggingFace`](https://huggingface.co/datasets/SS3DM/SS3DM-Benchmark-example/tree/main)]


> SS3DM: Benchmarking Street-View Surface Reconstruction with a Synthetic 3D Mesh Dataset <br>
[Yubin Hu*](https://github.com/AlbertHuyb), Kairui Wen*, Heng Zhou, [Xiaoyang Guo](https://xy-guo.github.io/), Yong-Jin Liu <br>
NeurIPS 2024 Track on Dataset and Benchmark

## News
🌟 [2024/11/07] We've uploaded the data to [GoogleDrive](https://drive.google.com/drive/folders/1-0xeAUYR6PicWK9eWmslVii3jGtyhL8Z) and [HuggingFace](https://huggingface.co/SS3DM)!

🥳 [2024/09/26] Our [paper](https://arxiv.org/abs/2410.21739) is accepted by NeurIPS 2024 Track on Dataset and Benchmark.

## Benchmark 

Benchmarking results on all sequences, including 14 short sequences, 8 middle sequences and 6 long sequences. 

| Method | IoU↑ | Prec.↑ | Recall↑ | F-score↑ | Acc↓ | Comp↓ | CD↓ | Acc_N↓ | Comp_N↓ | CD_N↓ | CD+CD_N↓ |
|--------|-------|---------|---------|-----------|-------|--------|------|--------|----------|--------|------------|
| R3D3   | 0.003 | 0.006   | 0.008   | 0.007     | 0.898 | 0.925  | 1.823| 0.717  | 0.712    | 1.429  | 3.252       |
| UrbanNeRF | 0.046 | 0.086   | 0.123   | 0.098     | 0.432 | 0.575  | 1.007| 0.442  | 0.557    | 0.999  | 2.006       |
| SuGaR  | 0.032 | 0.069   | 0.053   | 0.056     | 0.444 | 0.469  | 0.914| 0.650  | 0.662    | 1.312  | 2.226       |
| StreetSurf (RGB) | 0.044 | 0.078 | 0.067 | 0.069 | 0.372 | 0.490 | 0.862 | 0.517 | 0.616 | 1.133 | 1.995 |
| NeRF-LOAM | 0.072 | 0.107 | 0.139 | 0.116 | **0.151** | 0.400 | **0.551** | 0.687 | 0.724 | 1.411 | 1.962 |
| StreetSurf (LiDAR) | 0.107 | **0.206** | **0.245** | **0.215** | 0.246 | 0.367 | 0.613 | 0.506 | 0.582 | 1.088 | 1.701 |
| StreetSurf (Full) | **0.116** | 0.196 | 0.218 | 0.198 | 0.202 | **0.367** | 0.569 | **0.414** | **0.541** | **0.955** | **1.524** |

## Evaluation

You can download some example output meshes from [here](https://drive.google.com/drive/folders/1-0xeAUYR6PicWK9eWmslVii3jGtyhL8Z). You can download the zip files under `output-mesh-part` folder and unzip them to an `output-mesh` folder.

### Evaluate the meshes

We provide the example evaluation scripts for the methods mentioned in our paper.

Evaluate meshes produced by StreetSurf.
```
python evaluate_mesh.py --exp_dir output-mesh/streetsurf  --method streetsurf --box --resample
```

Evaluate meshes produced by UrbanNerf.
```
python evaluate_mesh.py --exp_dir output-mesh/urban_nerf --method urban_nerf --box --resample
```

Evaluate meshes produced by SuGaR.
```
python evaluate_mesh.py --exp_dir output-mesh/sugar --method sugar --box --resample
```

Evaluate meshes produced by NeRF-LOAM
```
python evaluate_mesh.py --exp_dir output-mesh/nerf_loam --method nerf_loam --box --resample
```

Evaluate meshes produced by R3D3
```
python evaluate_mesh.py --exp_dir output-mesh/r3d3 --method r3d3 --box --resample
```

### Collect the metrics

We also include an example script to collect evaluation results and form a latex table
```
python collect_results.py --box --resample --plt_curve
```

## Run the existing methods

### StreetSurf

You can use this script to predict the meshes. [[neuralsim/code_single/tools/train_for_ss3dm.py]](https://github.com/AlbertHuyb/neuralsim/blob/main/code_single/tools/train_for_ss3dm.py)

The results would be saved to `neuralsim/logs/ss3dm/streetsurf`.

### UrbanNeRF

You can use this script to predict the meshes. [[neuralsim/code_single/tools/train_for_ss3dm_urban_nerf.py]](https://github.com/AlbertHuyb/neuralsim/blob/main/code_single/tools/train_for_ss3dm_urban_nerf.py)


The results would be saved to `neuralsim/logs/ss3dm/urban_nerf`.

### SuGaR

You can use this script to train and extract mesh models. [[SuGaR/train_ss3dm.py]](https://github.com/AlbertHuyb/SuGaR/blob/main/train_ss3dm.py) 

The produced meshes should be flipped by this script. [[SuGaR/convert_mesh.py]](https://github.com/AlbertHuyb/SuGaR/blob/main/convert_mesh.py)

The results would be saved to `SuGaR/output/refined_mesh_flip`.

### NeRF-LOAM

You can use this script to train the models. [[NeRF-LOAM/demo/train_for_ss3dm_nerf_loam.py]](https://github.com/AlbertHuyb/NeRF-LOAM/blob/master/demo/train_for_ss3dm_nerf_loam.py) 

The produced meshes should be post-processed by this script.  [[NeRF-LOAM/demo/post_process_for_ss3dm_nerf_loam.py]](https://github.com/AlbertHuyb/NeRF-LOAM/blob/master/demo/post_process_for_ss3dm_nerf_loam.py)

The results would be saved to `NeRF-LOAM/logs/ss3dm`.

### R3D3
You can use this script to predict the depth maps. [[r3d3/evaluate.sh]](https://github.com/AlbertHuyb/r3d3/blob/master/evaluate.sh) 

The produced depth maps should be post-processed by these scripts. [[r3d3/tools/fuse_depth_to_pointcloud.py]](https://github.com/AlbertHuyb/r3d3/blob/master/tools/fuse_depth_to_pointcloud.py), [[r3d3/tools/surface_extraction.py]](https://github.com/AlbertHuyb/r3d3/blob/master/tools/surface_extraction.py)

After the post-processing step, the results would be converted to predicted mesh surfaces.

The results would be saved to `r3d3/logs/ddad_tiny/eval_predictions`. 
