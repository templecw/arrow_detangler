# diagram_detector/evaluate.py

import torch
from torch.utils.data import DataLoader, ConcatDataset
import os
import json

from datasets.gen_coco_dataset import GenCOCODataset
from models.faster_rcnn_model import get_model
from utils.transforms import get_transform
from datasets.unified_label_mapping import unified_label_mapping, generalized_categories

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def collate_fn(batch):
    return tuple(zip(*batch))

def merge_coco_annotations(datasets, merge_file_name='./results/merged_annotations.json'):
    """
    Merges COCO annotations from multiple datasets into a single COCO object,
    mapping all categories to the generalized categories defined in generalized_categories.

    Args:
        datasets: List of dataset instances (e.g., hdBPMN, FA, FC_B test datasets).

    Returns:
        merged_coco: COCO object with merged annotations.
        image_id_mapping: Mapping from (dataset, old_image_id) to new_image_id.
    """
    merged_dataset = {
        'images': [],
        'annotations': [],
        'categories': []
    }

    # Use the generalized categories as the final categories
    # generalized_categories = {'circle': 1, 'rectangle': 2, 'diamond': 3, 'oval': 4, 'arrow': 5, 'text': 6}
    merged_dataset['categories'] = [
        {'id': cat_id, 'name': cat_name}
        for cat_name, cat_id in generalized_categories.items()
    ]

    annotation_id = 1  # Unique annotation ID across all datasets
    image_id_mapping = {}  # Map (dataset, old_image_id) to new_image_id
    image_id_offset = 0

    for dataset in datasets:
        coco = dataset.coco
        # Retrieve dataset images and annotations
        dataset_images = coco.dataset['images']
        dataset_annotations = coco.dataset['annotations']

        # Map image IDs to ensure uniqueness
        max_image_id = 0
        for img in dataset_images:
            old_image_id = img['id']
            new_image_id = old_image_id + image_id_offset
            image_id_mapping[(dataset, old_image_id)] = new_image_id

            img_copy = img.copy()
            img_copy['id'] = new_image_id
            merged_dataset['images'].append(img_copy)
            if new_image_id > max_image_id:
                max_image_id = new_image_id

        # Update annotations
        for ann in dataset_annotations:
            ann_copy = ann.copy()
            ann_copy['id'] = annotation_id
            annotation_id += 1

            # Update image_id
            old_image_id = ann['image_id']
            ann_copy['image_id'] = image_id_mapping[(dataset, old_image_id)]

            # Map the dataset-specific category to a unified label (generalized category)
            original_cat_id = ann['category_id']
            category_name = dataset.category_id_to_name[original_cat_id]

            # Check if category_name is in unified_label_mapping
            if category_name not in unified_label_mapping:
                # If not mapped, skip this annotation
                continue

            unified_label = unified_label_mapping[category_name]
            # unified_label corresponds directly to the category_id in generalized_categories
            ann_copy['category_id'] = unified_label

            merged_dataset['annotations'].append(ann_copy)

        image_id_offset = max_image_id + 1

    # Write merged annotations to a file
    with open(merge_file_name, 'w') as f:
        json.dump(merged_dataset, f)

    merged_coco = COCO(merge_file_name)
    return merged_coco, image_id_mapping

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ################### Uncomment if you want to use hdBPMN #############################
    # hdBPMN_root_dir = 'hdBPMN'  # Adjust as necessary
    # hdBPMN_test_dir = os.path.join(hdBPMN_root_dir, 'test')
    # hdBPMN_test_annotation_file = os.path.join(hdBPMN_root_dir, 'test.json')

    FA_root_dir = 'FA'  # Adjust as necessary
    FA_test_dir = os.path.join(FA_root_dir, 'test')
    FA_test_annotation_file = os.path.join(FA_root_dir, 'test.json')

    FC_B_root_dir = 'FC_B'  # Adjust as necessary
    FC_B_test_dir = os.path.join(FC_B_root_dir, 'test')
    FC_B_test_annotation_file = os.path.join(FC_B_root_dir, 'test.json')

    # Load the test datasets
    ################### Uncomment if you want to use hdBPMN #############################
    # test_dataset_hdBPMN = GenCOCODataset(
    #     root=hdBPMN_test_dir,
    #     annotation_file=hdBPMN_test_annotation_file,
    #     transforms=get_transform(train=False),
    #     unified_label_mapping=unified_label_mapping
    # )

    test_dataset_FA = GenCOCODataset(
        root=FA_test_dir,
        annotation_file=FA_test_annotation_file,
        transforms=get_transform(train=False),
        unified_label_mapping=unified_label_mapping
    )

    test_dataset_FC_B = GenCOCODataset(
        root=FC_B_test_dir,
        annotation_file=FC_B_test_annotation_file,
        transforms=get_transform(train=False),
        unified_label_mapping=unified_label_mapping
    )
    ################### Uncomment if you want to use hdBPMN #############################
    # test_datasets = [test_dataset_hdBPMN, test_dataset_FA, test_dataset_FC_B]
    test_datasets = [test_dataset_FA, test_dataset_FC_B]
    test_dataset = ConcatDataset(test_datasets)

    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Number of classes (including background)
    num_classes = len(generalized_categories) + 1  # circle, rectangle, diamond, oval, arrow, text + background

    # Load the trained model
    model = get_model(num_classes)
    model.load_state_dict(torch.load('models/saved_models/gen_detector.pth', map_location=device))
    model.to(device)
    model.eval()

    # Merge COCO annotations from all test datasets into a single COCO object
    coco_gt, image_id_mapping = merge_coco_annotations(test_datasets)

    # Create a reverse mapping from dataset indices to determine which dataset an image belongs to
    dataset_lengths = [len(ds) for ds in test_datasets]
    dataset_cumulative_lengths = [sum(dataset_lengths[:i+1]) for i in range(len(dataset_lengths))]

    # We'll need to map labels back to category names (only if needed for debugging)
    # But since unified_label_mapping maps category_name->unified_label,
    # and generalized_categories maps category_name->id,
    # unified_label IS the category_id used by the model, so no extra mapping is needed here.

    coco_results = []

    for idx, (images, targets) in enumerate(test_data_loader):
        images = list(img.to(device) for img in images)
        image_ids = [t["image_id"].item() for t in targets]

        # Determine which dataset this index belongs to
        dataset_idx = next(i for i, cum_len in enumerate(dataset_cumulative_lengths) if idx < cum_len)
        dataset = test_datasets[dataset_idx]
        dataset_offset = 0 if dataset_idx == 0 else dataset_cumulative_lengths[dataset_idx - 1]

        with torch.no_grad():
            outputs = model(images)

        for image_id, output in zip(image_ids, outputs):
            boxes = output['boxes'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            labels = output['labels'].cpu().numpy()

            if boxes.shape[0] == 0:
                continue

            boxes[:, 2:] -= boxes[:, :2]  # Convert [xmin, ymin, xmax, ymax] -> [xmin, ymin, w, h]

            # Map image_id to merged_image_id
            merged_image_id = image_id_mapping[(dataset, image_id)]

            for box, score, label in zip(boxes, scores, labels):
                # Here, label is already a unified label which equals the generalized category ID
                # Because the model is trained on generalized categories and unified_label_mapping ensures consistency.
                # So we can directly use 'label' as category_id.
                category_id = label

                coco_result = {
                    'image_id': merged_image_id,
                    'category_id': int(category_id),
                    'bbox': box.tolist(),
                    'score': float(score),
                }
                coco_results.append(coco_result)

    # Save results in COCO format
    with open('./results/results.json', 'w') as f:
        json.dump(coco_results, f, indent=4)

    # Load results and evaluate
    coco_dt = coco_gt.loadRes('./results/results.json')
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == '__main__':
    main()