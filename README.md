# Diagram Detection and Relationship Extraction

This project is designed to detect diagrammatic elements (shapes, text, arrows) in images and determine the relationships between these elements based on arrow connections. It supports multiple datasets by merging their COCO annotations into a unified set of generalized categories (e.g., circle, rectangle, diamond, oval, arrow, text).

## Features

- **Object Detection**: Uses a Faster R-CNN model (or similar) trained on unified categories (circle, rectangle, diamond, oval, arrow, text).
- **Unified Label Mapping**: Maps dataset-specific categories from multiple datasets to a common set of generalized categories.
- **COCO Evaluation**: Merges and evaluates multiple datasets with a unified COCO annotation set.
- **Post-processing**: Identifies arrow endpoints and determines which shapes are connected by arrows, building a graph-like relationship structure.

## Directory Structure
```
diagram_detector/
├── datasets/
│   ├── __init__.py
│   ├── gen_coco_dataset.py
│   └── unified_label_mapping.py
├── models/
│   ├── __init__.py
│   └── faster_rcnn_model.py
├── utils/
│   ├── __init__.py
│   └── transforms.py
├── train.py
├── evaluate.py
├── predict.py
├── post_processing.py
├── visualize_samples.py
├── requirements.txt
└── README.md
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```
or ensure the following packages:
```
torch>=1.9.0
torchvision>=0.10.0
opencv-python
Pillow
numpy
pycocotools
matplotlib
scikit-image
argparse
```

### 2. Prepare the Datasets
**NOTE**: This is only required if you plan to use ```train.py```, ```evaluate.py```, or ```visualize_samples.py```. These folders contain thousands of images. For the sake of saving space, ```hdBPMN``` dataset is not included here. To use ```hdBPMN```, follow the provided link, download the dataset, and rename the folder to ```hdBPMN```. 

If you want to use this dataset, you'll need to find and uncomment the lines in the code marked with ```################### Uncomment if you want to use hdBPMN #############################```. Here's what you would uncomment in ```train.py```:

```python
################### Uncomment if you want to use hdBPMN #############################
70    # hdBPMN_root_dir = 'hdBPMN'  
71    # hdBPMN_train_dir = os.path.join(hdBPMN_root_dir, 'train')
72    # hdBPMN_train_annotation_file = os.path.join(hdBPMN_root_dir, 'train.json')
    ...

    ################### Uncomment if you want to use hdBPMN #############################
    # Initialize datasets with unified labels
90    # train_dataset_hdBPMN = GenCOCODataset(
91    #     root=hdBPMN_train_dir,
92    #     annotation_file=hdBPMN_train_annotation_file,
93    #     transforms=get_transform(train=True),
94    #     unified_label_mapping=unified_label_mapping
95    # )
    ...

    ################### Uncomment if you want to use hdBPMN #############################
134   # train_dataset = ConcatDataset([train_dataset_hdBPMN, train_dataset_FA, train_dataset_FC_B])
135   train_dataset = ConcatDataset([train_dataset_FA, train_dataset_FC_B])
    #### and comment the above line out ####
```

The datasets will need to be extracted from their zip files, located in ```diagram_detector```. 

Each dataset (e.g., hdBPMN, FA, FC_B) will follow the COCO format with:
```
dataset_name/
    train/
        img1.png
        img2.png
        ...
    val/
        img3.png
        img4.png
        ...
    test/
        img5.png
        img6.png
        ...
    train.json
    val.json
    test.json
```

These files were provided by _____ 

**NOTE:** The model created by the program is too large for git, so train will need to be run. The provided datasets will be sufficient for training a model capable of recognition with online notes. 

### 3. Unified Label Mapping
```unified_label_mapping.py``` maps all dataset-specific categories to generalized categories. Adjust this if you introduce new datasets or categories.

### 4. Training the Model
Simply run:
```bash
python train.py
```
Uses ```train.json``` and optionally a ```val.json``` **(validation during training not currently operable)** set to train.
The model will be saved to ```models/saved_models/gen_detector.pth``` by default.
### 5. Evaluating the Model
Simply run:
```bash
python evaluate.py
```
Merges and evaluates multiple test sets based on unified categories.
Produces COCO metrics to assess model performance.
### 6. Running Inference
Simply run:
```bash
python predict.py -f path/to/image 
```
Loads a single image and runs the model to detect objects.
Visualizes and/or prints out predictions.
### 7. Post-processing (Relationships)
Simply run:
```bash
python post_processing.py -f path/to/image
```
- Uses model outputs to find relationships between shapes based on arrows.
- Extracts arrow endpoints (via heuristic or advanced methods).
- Associates endpoints with nearest shapes and constructs a graph of connections.