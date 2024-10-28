import os
import json
import argparse
from matplotlib import pyplot as plt

def get_metrics_data(method_name):
    if method_name == 'streetsurf':
        exp_root = '/data/huyb/cvpr-2024/neuralsim/logs/ss3dm/streetsurf'
        
        submethod_list =  ['withmask_withnormal_all_cameras', 'onlylidar_all_cameras',  'withmask_withlidar_withnormal_all_cameras']
    elif method_name == 'urban_nerf':
        exp_root = '/data/huyb/cvpr-2024/neuralsim/logs/ss3dm/urban_nerf'
        
        submethod_list = ['withmask_withlidar_withnormal_all_cameras']
        
    elif method_name == 'r3d3':
        exp_root = '/data/huyb/cvpr-2024/r3d3/logs/ddad_tiny/eval_predictions'
        
        submethod_list = ['']
    
    elif method_name == 'nerf-loam':
        exp_root = '/data/huyb/cvpr-2024/NeRF-LOAM/logs/ss3dm'
        
        submethod_list = ['']
    
    elif method_name == 'sugar':
        exp_root = '/data/huyb/cvpr-2024/SuGaR/output/refined_mesh_flip'
        
        submethod_list = ['']

    all_metrics = {}
    # import pdb; pdb.set_trace()
    for method in submethod_list:
        all_metrics[method_name+'('+method+')'] = {}
        for scene in scene_list:
            if method_name == 'streetsurf':
                metric_file_path = os.path.join(exp_root, scene, method, 'metrics_dense.json')
            elif method_name == 'urban_nerf':
                metric_file_path = os.path.join(exp_root, scene, method, 'metrics_dense.json')
            elif method_name == 'r3d3':
                town_name = scene.split('_')[0]
                # import pdb; pdb.set_trace()
                metric_file_path = os.path.join(exp_root, town_name, scene, 'metrics_dense.json')
            elif method_name == 'nerf-loam':
                scene_dir = os.path.join(exp_root, scene)
                method = os.listdir(scene_dir)[0]
                
                metric_file_path = os.path.join(scene_dir, method, 'metrics_dense.json')
                
                method = ''
            elif method_name == 'sugar':
                scene_dir = os.path.join(exp_root, scene)
                
                metric_file_path = os.path.join(scene_dir, 'metrics_dense.json')
                
                method = ''
            
            if args.box:
                metric_file_path = metric_file_path.replace('.json', '_box.json')
            
            if args.resample:
                metric_file_path = metric_file_path.replace('.json', '_resampled.json')
            
            # import pdb; pdb.set_trace()
            if os.path.exists(metric_file_path):
                with open(metric_file_path, 'r') as f:
                    metrics = json.load(f)
                    all_metrics[method_name+'('+method+')'][scene] = metrics
                    
                    all_metric_name = metrics.keys()

    avg_metrics = {}
    for method in all_metrics.keys():
        avg_metrics[method] = {}
        
        avg_metrics[method]['Short'] = {}
        avg_metrics[method]['Medium'] = {}
        avg_metrics[method]['Long'] = {}
        avg_metrics[method]['All'] = {}
        
        for metric_name in all_metric_name:
            if 'f_score' in metric_name:
                metric_item_list = ['precision', 'recall', 'f_score']
            elif metric_name == 'chamfer_dist(m)':
                metric_item_list = ['accuracy', 'completeness', 'chamfer_dist']
            elif metric_name == 'normal_chamfer_dist(cosine)':
                metric_item_list = ['normal_accuracy', 'normal_completeness', 'normal_chamfer_dist']
            elif 'IoU' in metric_name:
                metric_item_list = ['IoU']
            
            avg_metrics[method]['Short'][metric_name] = {}
            avg_metrics[method]['Medium'][metric_name] = {}
            avg_metrics[method]['Long'][metric_name] = {}
            avg_metrics[method]['All'][metric_name] = {}
            
            for metric_item in metric_item_list:
                avg_metrics[method]['Short'][metric_name][metric_item] = 0.0
                avg_metrics[method]['Medium'][metric_name][metric_item] = 0.0
                avg_metrics[method]['Long'][metric_name][metric_item] = 0.0
                avg_metrics[method]['All'][metric_name][metric_item] = 0.0
            
            short_count = 0
            medium_count = 0
            long_count = 0
            
            for scene in scene_list:
                if scene not in all_metrics[method].keys():
                    print('Scene {} not in all_metrics for {}'.format(scene, method))
                    exit(0)
                    continue
                length = int(scene.split('_')[1])
                
                if length <= 300:
                    for metric_item in metric_item_list:   
                        avg_metrics[method]['Short'][metric_name][metric_item] += all_metrics[method][scene][metric_name][metric_item]
                    short_count += 1
                elif length <= 600:
                    for metric_item in metric_item_list:   
                        avg_metrics[method]['Medium'][metric_name][metric_item] += all_metrics[method][scene][metric_name][metric_item]
                    medium_count += 1
                elif length > 600:
                    for metric_item in metric_item_list:   
                        avg_metrics[method]['Long'][metric_name][metric_item] += all_metrics[method][scene][metric_name][metric_item]
                    long_count += 1
                
                for metric_item in metric_item_list:   
                    avg_metrics[method]['All'][metric_name][metric_item] += all_metrics[method][scene][metric_name][metric_item]
            
            for metric_item in metric_item_list:
                avg_metrics[method]['Short'][metric_name][metric_item] /= short_count
                avg_metrics[method]['Medium'][metric_name][metric_item] /= medium_count
                avg_metrics[method]['Long'][metric_name][metric_item] /= long_count
                avg_metrics[method]['All'][metric_name][metric_item] /= (short_count + medium_count + long_count)
            
            # print('All sequence counts: {}, {}, {}'.format(short_count, medium_count, long_count))

    # import pdb; pdb.set_trace()     
        
    
        


    return all_metrics, avg_metrics, submethod_list


parser = argparse.ArgumentParser()
parser.add_argument('--output_path', type=str, default='./output-metrics')
parser.add_argument('--box', action='store_true')
parser.add_argument('--resample', action='store_true')
parser.add_argument('--plt_curve', action='store_true')

args = parser.parse_args()

output_path = args.output_path

scene_list = []
data_root = '/data/huyb/cvpr-2024/data/ss3dm/DATA'
for town_name in os.listdir(data_root):
    if os.path.isdir(os.path.join(data_root, town_name)):
        town_dir = os.path.join(data_root, town_name)
        for scene in os.listdir(town_dir):
            if os.path.isdir(os.path.join(town_dir, scene)):
                scene_list.append(scene)
                
scene_list.sort()
print(scene_list)

output_file = os.path.join(output_path, 'collected_results.tex')


save_metric_name = ['IoU(0.10m)', 'f_score(0.05m)', 'chamfer_dist(m)', 'normal_chamfer_dist(cosine)']
metric_subitem_num = [1, 3, 3, 3]
        
if args.box:
    output_file = output_file.replace('.tex', '_box.tex')

if args.resample:
    output_file = output_file.replace('.tex', '_resampled.tex')
    
if not os.path.exists(output_path):
    os.makedirs(output_path)
    
lines_to_write = ''
lines_to_write += ('\\begin{tabular}{|' + 'c|' * (sum(metric_subitem_num) + 1) + '}\n')
lines_to_write += ('\\hline\n')
lines_to_write += ('method & part &')
lines_to_write += ('\\hline\n')
for metric_name in save_metric_name:
    lines_to_write += ('{} & '.format(metric_name))
lines_to_write += ('\\\\\n')
lines_to_write += ('\\hline\n')

for part in ['Short', 'Medium', 'Long', 'All']:       
    for method_name in ['r3d3', 'urban_nerf', 'sugar', 'streetsurf', 'nerf-loam']:
        all_metrics, avg_metrics, submethod_list = get_metrics_data(method_name)

        for method in avg_metrics.keys():
            lines_to_write += ('{} & {} &  '.format(part, method))

            for metric_name in save_metric_name:
                for metric_item in avg_metrics[method][part][metric_name].keys():
                    lines_to_write += ('{:.3f} & '.format(avg_metrics[method][part][metric_name][metric_item]))
            
            lines_to_write += ('{:.3f} & '.format(avg_metrics[method][part]['chamfer_dist(m)']['chamfer_dist'] + avg_metrics[method][part]['normal_chamfer_dist(cosine)']['normal_chamfer_dist']))
            lines_to_write += ('\\\\\n')
            lines_to_write += ('\\hline\n')
    
lines_to_write += ('\\end{tabular}\n')

with open(output_file, 'w') as f:
    f.write(lines_to_write)
    
if args.plt_curve:
    # Draw f-score curve
    for curve_term in ['f_score', 'IoU']:
        for part in ['Short', 'Medium', 'Long', 'All']:
            plt.figure(figsize=(6, 8))
            # set the width of lines
            plt.rcParams['lines.linewidth'] = 3
            plt.rcParams['lines.markersize'] = 10
            
            for method_name in ['r3d3', 'urban_nerf', 'streetsurf', 'nerf-loam', 'sugar']:
                all_metrics, avg_metrics, submethod_list = get_metrics_data(method_name)
                for method in avg_metrics.keys():
                    f_score = []
                    threshold = []
                    for metric_name in avg_metrics[method][part].keys():
                        if curve_term in metric_name:
                            f_score.append(avg_metrics[method][part][metric_name][curve_term])
                            this_threshold = float(metric_name.split('(')[1].split('m')[0])
                            threshold.append(this_threshold)
                    
                    if method == 'streetsurf(onlylidar_all_cameras)':
                        plt.plot(threshold, f_score, label='StreetSurf (LiDAR)', marker='o', color='lightcoral', linestyle='dashed')
                    elif method == 'streetsurf(withmask_withnormal_all_cameras)':
                        plt.plot(threshold, f_score, label='StreetSurf (RGB)', marker='o', color='indianred', linestyle='dashdot')
                    elif method == 'streetsurf(withmask_withlidar_withnormal_all_cameras)':
                        plt.plot(threshold, f_score, label='StreetSurf (Full)', marker='o', color='brown', linestyle='solid')
                    elif method == 'urban_nerf(withmask_withlidar_withnormal_all_cameras)':
                        plt.plot(threshold, f_score, label='UrbanNerf', marker='^')
                    elif method == 'r3d3()':
                        plt.plot(threshold, f_score, label='R3D3', marker='p')
                    elif method == 'nerf-loam()':
                        plt.plot(threshold, f_score, label='Nerf-LOAM', marker='d')
                    elif method == 'sugar()':
                        plt.plot(threshold, f_score, label='SuGaR', marker='*')

            # set legend position
            plt.legend(loc='upper left')
            plt.xlabel('Threshold')
            if curve_term == 'f_score':
                plt.ylabel('F-score')
                plt.title('F-score curve ({})'.format(part))
                
                ## set y axis range
                plt.ylim(0.0, 1.0)
                
            elif curve_term == 'IoU':
                plt.ylabel('IoU')
                plt.title('IoU curve ({})'.format(part))
                
                ## set y axis range
                plt.ylim(0.0, 0.6)
            
            
            
            figure_path = os.path.join(output_path, '{}_curve_{}.pdf'.format(curve_term, part))

            if args.box:
                figure_path = figure_path.replace('.pdf', '_box.pdf')
            
            if args.resample:
                figure_path = figure_path.replace('.pdf', '_resampled.pdf')
            
            
            plt.savefig(figure_path, bbox_inches='tight')
            plt.close()