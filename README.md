# Traffic_Light_and_Road_Sign_Detection 

This repository contains two Jupyter Notebooks designed for custom object detection using PyTorch and YOLOv5. The code includes custom dataset parsing from XML annotations, creating PyTorch `Dataset` and `DataLoader` classes, and setting up training pipelines.

## 📄 Contents

- `custom_dataset_loader.ipynb`: Implements a PyTorch `Dataset` class to handle object detection data with Pascal VOC-style XML annotations. Includes bounding box parsing, label encoding, and dataloader generation.
- `yolov5_training_pipeline.ipynb`: Contains the setup for training a YOLOv5 model on a custom dataset. Includes repository cloning, dataset registration, and preparation steps.

## 📦 Requirements

Install the necessary packages:

```bash
pip install torch torchvision numpy pandas matplotlib Pillow xmltodict opencv-python
``` 

## 📂 Dataset Structure
Expected folder structure:
```
dataset/
├── images/
│   ├── image1.jpg
│   └── ...
└── annotations/
    ├── image1.xml
    └── ...
Each XML file should follow the Pascal VOC format.
```

## 🚀 Training Setup
To create dataloaders for training, validation, and testing:

```
from dataset import get_dls
train_dl, val_dl, test_dl, class_names = get_dls(root='dataset', transformations=your_transforms, bs=8)
```
You can then integrate the dataset with your object detection model or use it with YOLOv5.

## 🔍 YOLOv5 Integration
The second notebook handles:

Cloning the YOLOv5 repository.

Preparing your custom dataset in YOLO format.

Training the model using train.py.

## 📊 Visualization & Evaluation
Use matplotlib or OpenCV to visualize bounding boxes and model predictions.
