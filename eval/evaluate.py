# standard
import os
import numpy as np
import pandas as pd
import json

# plotting
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import imageio

# evaluation
import motmetrics
from get_trackings import get_trackings



def read_trackings(results_dir):
    filenames = [x for x in os.listdir(results_dir) if x[-5:] == '.json']
    for filename in filenames:
        exp_info = json.loads(filename[:-5])
        with open(os.path.join(results_dir, filename), 'r') as f:
            trackings = json.load(f)
        gt_raw = f'eval/2DMOT2015/{exp_info["datasetTrainOrTest"]}/{exp_info["datasetName"]}/gt/gt.txt'
        gt_raw = pd.read_csv(gt_raw, header=None).values.astype(np.int32)
        ground_truth = [[] for _ in range(gt_raw[-1,0])]
        for row in gt_raw:
            ground_truth[row[0]-1].append(row[1:])
        ground_truth = list(map(np.array, ground_truth))
        yield exp_info, trackings, ground_truth
    

def fig_to_np_array(fig):
    fig.canvas.draw()
    arr = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    n_cols, n_rows = fig.canvas.get_width_height()
    return np.reshape(arr, (n_rows, n_cols, 3))

def make_gif(results_dir, exp_info, trackings):
    gifname = os.path.join(results_dir, exp_info["datasetName"] + '.gif')
    with imageio.get_writer(gifname, mode='I') as writer:
        for i, tracks in enumerate(trackings):
            fig,ax = plt.subplots(1)
            img_filename = f'eval/2DMOT2015/{exp_info["datasetTrainOrTest"]}/{exp_info["datasetName"]}/img1/{(("0"*6) + str(i+1))[-6:]}.jpg'
            ax.imshow(np.array(Image.open(img_filename), dtype=np.uint8), animated=True)
            for t in tracks:
                x1,y1, x2,y2 = t['bbox']
                ax.text((x1+x2)/2, y2, str(t['id']), bbox=dict(facecolor='white', alpha=0.75))
                rect = patches.Rectangle((x1,y1),x2-x1,y2-y1,linewidth=1,edgecolor='r',facecolor='none')
                ax.add_patch(rect)
            writer.append_data(fig_to_np_array(fig))
            plt.close(fig)



def make_all_gifs(results_dir):
    for exp_info, trackings, _ in read_trackings(results_dir):
        make_gif(results_dir, exp_info, trackings)



def evaluate_experiment(results_dir):
    accumulators = []
    experiment_names = []
    for exp_info, trackingsOverTime, ground_truth in read_trackings(results_dir):
        acc = motmetrics.MOTAccumulator(auto_id=True)
        accumulators.append(acc)
        experiment_names.append(exp_info['datasetName'])
        for tracks, gt in zip(trackingsOverTime, ground_truth):
            if tracks:
                x1,y1, x2,y2 = np.array([t['bbox'] for t in tracks]).T
            else:
                x1,y1, x2,y2 = [np.array([])] * 4
            pred_bboxes = np.array([x1,y1,x2-x1,y2-y1]).T
            iou = motmetrics.distances.iou_matrix(gt[:, 1:5] if len(gt) else np.empty((0,0)), pred_bboxes, max_iou=1)
            acc.update(gt[:,0] if len(gt) else np.empty((0)), [t['id'] for t in tracks], iou)
    
    mh = motmetrics.metrics.create()
    summary = mh.compute_many(accumulators,
        metrics=motmetrics.metrics.motchallenge_metrics,
        names = experiment_names,
        generate_overall=True)
    strsummary = motmetrics.io.render_summary(
        summary,  
        formatters=mh.formatters, 
        namemap=motmetrics.io.motchallenge_metric_names
    )
    strsummary = "MOTA and MOTP are wrong. See py-motmetrics README. Don't use these.\n" + strsummary
    return strsummary



def evaluate(tracker_weight_params = {}, train=True, n_datasets=1, gif=True):
    results_dir = get_trackings(tracker_weight_params = tracker_weight_params, train=train, n_datasets=n_datasets)
    metrics = evaluate_experiment(results_dir)
    if len(tracker_weight_params):
        with open(os.path.join(results_dir, 'tracker_weight_params.json'), 'w') as f:
            f.write(json.dumps(tracker_weight_params))
    with open(os.path.join(results_dir, 'metrics.txt'), 'w') as f:
        f.write(metrics)
    if gif:
        make_all_gifs(results_dir)
    print(metrics)
    return metrics


if __name__ == "__main__":
    evaluate(n_datasets=100, train=True, gif=True)
