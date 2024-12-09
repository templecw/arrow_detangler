# diagram_detector/visualize_samples.py

import os
import torch
from torch.utils.data import ConcatDataset
from datasets.gen_coco_dataset import GenCOCODataset
from utils.transforms import get_transform
from datasets.unified_label_mapping import generalized_categories, unified_label_mapping
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def visualize_sample(img, target, label_to_category_name):

    # Convert image tensor to NumPy array
    img_np = img.cpu().numpy().transpose((1, 2, 0))

    # Create a figure and axis
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img_np)

    boxes = target['boxes'].cpu().numpy()
    labels = target['labels'].cpu().numpy()

    for idx in range(len(boxes)):
        box = boxes[idx]
        label = labels[idx]
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin

        # Create a rectangle patch
        rect = patches.Rectangle(
            (xmin, ymin),
            width,
            height,
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

        # Get the category name
        category_name = label_to_category_name.get(label, 'Unknown')

        # Add label text
        ax.text(
            xmin,
            ymin - 5,
            f"{category_name}",
            color='white',
            fontsize=12,
            backgroundcolor='red'
        )

    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    ################### Uncomment if you want to use hdBPMN #############################
    # hdBPMN_root_dir = 'hdBPMN'  
    # hdBPMN_train_dir = os.path.join(hdBPMN_root_dir, 'train')
    # hdBPMN_train_annotation_file = os.path.join(hdBPMN_root_dir, 'train.json')

    FA_root_dir = 'FA' 
    FA_train_dir = os.path.join(FA_root_dir, 'train')
    FA_train_annotation_file = os.path.join(FA_root_dir, 'train.json')

    FC_B_root_dir = 'FC_B'
    FC_B_train_dir = os.path.join(FC_B_root_dir, 'train')
    FC_B_train_annotation_file = os.path.join(FC_B_root_dir, 'train.json')

    # Initialize datasets with unified labels
    ################### Uncomment if you want to use hdBPMN #############################
    # dataset_hdBPMN = GenCOCODataset(
    #     root=hdBPMN_train_dir,
    #     annotation_file=hdBPMN_train_annotation_file,
    #     transforms=get_transform(train=False),
    #     unified_label_mapping=unified_label_mapping
    # )

    dataset_FA = GenCOCODataset(
        root=FA_train_dir,
        annotation_file=FA_train_annotation_file,
        transforms=get_transform(train=False),
        unified_label_mapping=unified_label_mapping
    )

    dataset_FC_B = GenCOCODataset(
        root=FC_B_train_dir,
        annotation_file=FC_B_train_annotation_file,
        transforms=get_transform(train=False),
        unified_label_mapping=unified_label_mapping
    )

    # Combine datasets
    ################### Uncomment if you want to use hdBPMN #############################
    # dataset = ConcatDataset([dataset_hdBPMN, dataset_FA, dataset_FC_B])
    dataset = ConcatDataset([dataset_FA, dataset_FC_B])
    # Build label to category name mapping
    label_to_category_name = {v: k for k, v in generalized_categories.items()}

    # Visualize a few samples
    for idx in range(0,5):
        img, target = dataset[idx]
        visualize_sample(img, target, label_to_category_name)

if __name__ == '__main__':
    main()
