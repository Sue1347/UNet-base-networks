import matplotlib.pyplot as plt
import numpy as np
from medpy.metric.binary import dc, hd, assd
from sklearn.metrics import jaccard_score

import torch
import torch.nn.functional as F
from tqdm import tqdm




def plot_img_and_mask(img, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()

# import matplotlib.pyplot as plt

def show_prediction(inputs, preds, gts, idx=0):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(inputs[idx][0].cpu(), cmap='gray')
    plt.title("Input")
    
    plt.subplot(1, 3, 2)
    plt.imshow(preds[idx].cpu(), cmap='jet')
    plt.title("Prediction")

    plt.subplot(1, 3, 3)
    plt.imshow(gts[idx].cpu(), cmap='jet')
    plt.title("Ground Truth")

    plt.show()

def calculate_metrics_for_2D_volume(mask, groundtruth, cls_num):
    # matrix: [[dc, hd, assd, iou, f1], ...], from the first label to the last and the average.
    list_results = np.zeros((cls_num-1, 4)) # 5 metrics

    # Calculate metrics for different labels
    for i in range(cls_num-1):
        # print(f"Metrics for class {i} ...")
        # Ensure binary masks
        msk = (mask == (i+1)).astype(np.uint8)
        gt = (groundtruth == (i+1) ).astype(np.uint8)

        intersection = np.logical_and(gt, msk).sum()
        union = np.logical_or(gt, msk).sum()

        try:
            list_results[i, 0] = dc(msk, gt)
        except:
            list_results[i, 0] = np.nan

        try: 
            list_results[i, 1] = intersection / union
        except:
            list_results[i, 1] = np.nan

        try:
            list_results[i, 2] = hd(msk, gt)
        except:
            list_results[i, 2] = np.nan

        try:
            list_results[i, 3] = assd(msk, gt)
        except:
            list_results[i, 3] = np.nan

        # try:
        #     list_results[i, 4] = jaccard_score(gt.flatten(), msk.flatten()) # IoU
        # except:
        #     list_results[i, 4] = np.nan
    
        # TP = np.sum((msk == 1) & (gt == 1))
        # TN = np.sum((msk == 0) & (gt == 0))
        # FP = np.sum((msk == 1) & (gt == 0))
        # FN = np.sum((msk == 0) & (gt == 1))
        # print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
        
    # list_results[cls_num] = np.mean(list_results[:cls_num-1],axis=0)
    # print(f"Average: DICE: {list_results[cls_num-1, 0]:.4f}, Hausdorff: {list_results[cls_num-1, 1]:.4f}",
    #       f"ASSD: {list_results[cls_num-1, 2]:.4f}, IoU: {list_results[cls_num-1, 3]:.4f}, f1_score: {list_results[cls_num-1, 4]:.4f}")

    return list_results