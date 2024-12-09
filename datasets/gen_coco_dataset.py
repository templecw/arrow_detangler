# diagram_detector/datasets/gen_coco_dataset.py

import os
import torch
import torchvision
from PIL import Image

class GenCOCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation_file, transforms=None, unified_label_mapping=None):
        from pycocotools.coco import COCO

        self.root = root
        self.coco = COCO(annotation_file)
        self.transforms = transforms
        self.unified_label_mapping = unified_label_mapping

        # Load image ids
        self.ids = list(sorted(self.coco.imgs.keys()))

        # Original category IDs to names
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.category_id_to_name = {cat['id']: cat['name'] for cat in self.categories}

    def __getitem__(self, index):
        # Get image id
        img_id = self.ids[index]
        # Load image
        coco = self.coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        img_info = coco.loadImgs(img_id)[0]
        path = img_info['file_name']

        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            x_min, y_min, width, height = ann['bbox']
            x_max = x_min + width
            y_max = y_min + height
            boxes.append([x_min, y_min, x_max, y_max])

            category_id = ann['category_id']
            category_name = self.category_id_to_name[category_id]

            # Map to unified label
            label = self.unified_label_mapping.get(category_name)
            if label is None:
                continue  # Skip if not in the unified mapping

            labels.append(label)
            areas.append(ann['area'])
            iscrowd.append(ann.get('iscrowd', 0))

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([img_id])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = areas
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)
