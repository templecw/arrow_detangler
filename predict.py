# diagram_detector/predict.py

import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import argparse

from models.faster_rcnn_model import get_model
from utils.transforms import get_transform
from datasets.unified_label_mapping import generalized_categories

def visualize_predictions(img, predictions, label_to_category_name, conf_threshold=0.7):

    # Convert image tensor to NumPy array
    img_np = img.cpu().numpy().transpose((1, 2, 0))

    # Create a figure and axis
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img_np)

    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()

    indices = np.where(scores >= conf_threshold)[0]

    for idx in indices:
        box = boxes[idx]
        label = labels[idx]
        score = scores[idx]
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin

        # Create a rectangle patch
        rect = patches.Rectangle(
            (xmin, ymin),
            width,
            height,
            linewidth=2,
            edgecolor='g',
            facecolor='none'
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

        # Get the category name
        category_name = label_to_category_name.get(label, 'Unknown')

        # Add label text with confidence score
        ax.text(
            xmin,
            ymin - 5,
            f"{category_name}: {score:.2f}",
            color='white',
            fontsize=12,
            backgroundcolor='green'
        )

    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Process an image and determine arrow-based relationships between diagram elements.")
    parser.add_argument('-f', '--file', required=False, help='Path to the image to be processed')
    args = parser.parse_args()
    if args.file:
        image_path = args.file
    else:
        image_path = 'FC_B/test/writer017_fc_001.png' 

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Build label to category name mapping
    label_to_category_name = {v: k for k, v in generalized_categories.items()}

    # Number of classes (including background)
    num_classes = len(generalized_categories) + 1  # hdBPMN classes + background

    # Load the trained model
    model = get_model(num_classes)
    model.load_state_dict(torch.load('models/saved_models/gen_detector.pth'))
    model.to(device)
    model.eval()

    # Load the image
    img = Image.open(image_path).convert("RGB")
    img_tensor = get_transform(train=False)(img).to(device)

    with torch.no_grad():
        predictions = model([img_tensor])[0]

    # Visualize predictions
    visualize_predictions(img_tensor.cpu(), predictions, label_to_category_name, conf_threshold=0.7)

if __name__ == '__main__':
    main()
