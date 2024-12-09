# diagram_detector/train.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import os

from datasets.gen_coco_dataset import GenCOCODataset
from models.faster_rcnn_model import get_model
from utils.transforms import get_transform
from datasets.unified_label_mapping import unified_label_mapping, generalized_categories
from evaluate import merge_coco_annotations

from pycocotools.cocoeval import COCOeval
import json

def collate_fn(batch):
    return tuple(zip(*batch))

############## Currently inoperable due to the nuances of evaluating a combined dataset. ########################
# def validate(model, data_loader, device, epoch, coco_gt, image_id_mapping, d_c_lengths):
#     model.eval()
#     # coco_gt = data_loader.dataset.coco
#     coco_results = []

#     for idx, (images, targets) in enumerate(data_loader):
#         images = list(img.to(device) for img in images)
#         image_ids = [t["image_id"].item() for t in targets]

#         with torch.no_grad():
#             outputs = model(images)

#         for image_id, output in zip(image_ids, outputs):
#             boxes = output['boxes'].cpu().numpy()
#             scores = output['scores'].cpu().numpy()
#             labels = output['labels'].cpu().numpy()

#             if boxes.shape[0] == 0:
#                 continue

#             boxes[:, 2:] -= boxes[:, :2]  # Convert to [x, y, width, height]

#             for box, score, label in zip(boxes, scores, labels):
#                 # Map labels back to original category IDs for evaluation
#                 category_id = label  # Assuming labels correspond to unified labels
#                 coco_result = {
#                     'image_id': image_id,
#                     'category_id': category_id,
#                     'bbox': box.tolist(),
#                     'score': float(score),
#                 }
#                 coco_results.append(coco_result)

#     # Save results in COCO format
#     results_file = f'results_epoch_{epoch}.json'
#     with open(results_file, 'w') as f:
#         json.dump(coco_results, f, indent=4)

#     # Load results and evaluate
#     coco_dt = coco_gt.loadRes(results_file)
#     coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
#     coco_eval.evaluate()
#     coco_eval.accumulate()
#     coco_eval.summarize()

def main():
    # Check if GPU is available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ################### Uncomment if you want to use hdBPMN #############################
    # hdBPMN_root_dir = 'hdBPMN'  
    # hdBPMN_train_dir = os.path.join(hdBPMN_root_dir, 'train')
    # hdBPMN_train_annotation_file = os.path.join(hdBPMN_root_dir, 'train.json')
    # hdBPMN_val_dir = os.path.join(hdBPMN_root_dir, 'val')
    # hdBPMN_val_annotation_file = os.path.join(hdBPMN_root_dir, 'val.json')

    FA_root_dir = 'FA'  
    FA_train_dir = os.path.join(FA_root_dir, 'train')
    FA_train_annotation_file = os.path.join(FA_root_dir, 'train.json')
    # FA_val_dir = os.path.join(FA_root_dir, 'val')
    # FA_val_annotation_file = os.path.join(FA_root_dir, 'val.json')

    FC_B_root_dir = 'FC_B'  
    FC_B_train_dir = os.path.join(FC_B_root_dir, 'train')
    FC_B_train_annotation_file = os.path.join(FC_B_root_dir, 'train.json')
    # FC_B_val_dir = os.path.join(FC_B_root_dir, 'val')
    # FC_B_val_annotation_file = os.path.join(FC_B_root_dir, 'val.json')

################### Uncomment if you want to use hdBPMN #############################
    # Initialize datasets with unified labels
    # train_dataset_hdBPMN = GenCOCODataset(
    #     root=hdBPMN_train_dir,
    #     annotation_file=hdBPMN_train_annotation_file,
    #     transforms=get_transform(train=True),
    #     unified_label_mapping=unified_label_mapping
    # )

    train_dataset_FA = GenCOCODataset(
        root=FA_train_dir,
        annotation_file=FA_train_annotation_file,
        transforms=get_transform(train=True),
        unified_label_mapping=unified_label_mapping
    )

    train_dataset_FC_B = GenCOCODataset(
        root=FC_B_train_dir,
        annotation_file=FC_B_train_annotation_file,
        transforms=get_transform(train=True),
        unified_label_mapping=unified_label_mapping
    )

    # val_dataset_hdBPMN = GenCOCODataset(
    #     root=hdBPMN_val_dir,
    #     annotation_file=hdBPMN_val_annotation_file,
    #     transforms=get_transform(train=False),
    #     unified_label_mapping=unified_label_mapping
    # )

    # val_dataset_FA = GenCOCODataset(
    #     root=FA_val_dir,
    #     annotation_file=FA_val_annotation_file,
    #     transforms=get_transform(train=False),
    #     unified_label_mapping=unified_label_mapping
    # )

    # val_dataset_FC_B = GenCOCODataset(
    #     root=FC_B_val_dir,
    #     annotation_file=FC_B_val_annotation_file,
    #     transforms=get_transform(train=False),
    #     unified_label_mapping=unified_label_mapping
    # )

    # Combine datasets
    ################### Uncomment if you want to use hdBPMN #############################
    # train_dataset = ConcatDataset([train_dataset_hdBPMN, train_dataset_FA, train_dataset_FC_B])
    train_dataset = ConcatDataset([train_dataset_FA, train_dataset_FC_B])
    
    # Data loaders
    train_data_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)


    ####### validation stuff #####################
    # val_dataset = ConcatDataset([val_dataset_hdBPMN, val_dataset_FA, val_dataset_FC_B])

    # val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    # coco_gt, image_id_mapping = merge_coco_annotations(val_dataset)
    # dataset_lengths = [len(ds) for ds in val_dataset]
    # dataset_cumulative_lengths = [sum(dataset_lengths[:i+1]) for i in range(len(dataset_lengths))]
    #################################################

    # Number of classes (including background)
    num_classes = len(generalized_categories) + 1  # Unique labels + background

    # Get the model using our helper function
    model = get_model(num_classes)
    model.to(device)

    # Construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        i = 0
        epoch_loss = 0.0
        for images, targets in train_data_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass and optimization
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

            if i % 50 == 0:
                print(f"Epoch: {epoch}, Iteration: {i}, Loss: {losses.item():.4f}")
            i += 1

        # Compute average loss over the epoch
        avg_epoch_loss = epoch_loss / len(train_data_loader)
        print(f"Epoch {epoch} training loss: {avg_epoch_loss:.4f}")

        # Update the learning rate
        lr_scheduler.step()


        # Evaluate on the validation dataset
        # print(f"Evaluating on validation dataset at epoch {epoch}...")
        # validate(model, val_data_loader, device, epoch, coco_gt, image_id_mapping, dataset_cumulative_lengths)

    # Save the model
    if not os.path.exists('models/saved_models'):
        os.makedirs('models/saved_models')
    torch.save(model.state_dict(), 'models/saved_models/gen_detector.pth')

if __name__ == '__main__':
    main()
