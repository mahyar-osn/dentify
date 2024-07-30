# Dentify

Dentify is a Python package designed for the CoTreatAI Challenge. It facilitates data processing, model training, and prediction for dental x-ray images, specifically focusing on identifying and numbering teeth using the YOLOv8 model instance segmentation method.

## Table of Contents

- [Installation](#installation)
- [Directory Structure](#directory-structure)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Model Training](#model-training)
  - [Prediction](#prediction)
  - [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Hyperparameters](#hyperparameters)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

To install the required dependencies, run:

```bash
pip install -e .
```

## Directory Structure

The directory structure of the Dentify package is as follows:
```dentify/
├── configs/
│   └── train_configs.yaml
├── scripts/
│   ├── draw_annotations.py
├── src/
│   ├── dentify/
│   │   ├── __init__.py
│   │   ├── coco.py
│   │   │── training_yaml.py
├── .gitignore
├── evaluate.py
├── train.py
├── README.md
├── predict.py
├── pyroject.toml
├── setup.py
├── requirements.txt
```

- configs/: Contains configuration files for training.
- scripts/draw_annotations.py: Contains a script to draw and label images with annotations.
- src/: Source code for the Dentify package which includes utility scripts for data conversion and configuration generation.
- train.py: Script for training the model.
- predict.py: Script for making predictions using the trained model.
- pyroject.toml: TOML file for describing the project and dependencies and use for installation.
- requirements.txt: List of dependencies required for the package.

## Usage
### Data Preparation
Before training the model, the data needs to be prepared. This includes converting COCO annotations to 
YOLO format and generating training splits.
To prepare the data, run:

```bash
python train.py --prepare_data
```
Running the above command, will parse the coco `annotations.json` and create the following files required for training
within the same directory:
- data.yaml: a YAML file specifying the paths to images/labels as well as class names used for training.
- autosplit_train.txt: a TXT file specifying the paths to train split images.
- autosplit_val.txt: a TXT file specifying the paths to validation split images.
- autosplit_test.txt: a TXT file specifying the paths to test split images.

as well as a new directory called `labels` which contains the same number of TXT files as image files with corresponding
file names. Each TXT file in this directory contains class integers as well as the mask polygon integers used for
training.

## Model Training
To train the model, use the following command:
```bash
python train.py --configs_path configs/train_configs.yaml
```

This command will:
1. Load the configuration from configs/train_configs.yaml.
2. Prepare the dataset if --prepare_data is specified.
3. Train the YOLOv8 model using the specified hyperparameters.

Once training starts, in your terminal you should see something similar to:

```bash
Image sizes 640 train, 640 val
Using 8 dataloader workers
Logging results to runs/segment/cotreat_challenge_exp1
Starting training for 100 epochs...

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size

  0%|          | 0/56 [00:00<?, ?it/s]
      1/100      3.51G      1.218       4.65      4.425      1.446        267        640:   0%|          | 0/56 [00:00<?, ?it/s]
      1/100      3.51G      1.218       4.65      4.425      1.446        267        640:   2%|▏         | 1/56 [00:00<00:30,  1.82it/s]
      1/100      3.51G      1.206      4.641      4.413      1.437        261        640:   2%|▏         | 1/56 [00:00<00:30,  1.82it/s]
      1/100      3.51G      1.206      4.641      4.413      1.437        261        640:   4%|▎         | 2/56 [00:00<00:16,  3.21it/s]
      1/100      3.51G      1.194      4.492      4.423      1.423        246        640:   4%|▎         | 2/56 [00:00<00:16,  3.21it/s]
```
The results will be saved to `./runs/segment/<experiment_name>`.

## Prediction
To make predictions using the trained model, run:

```bash
python predict.py --image_path path/to/image.jpg --model_path path/to/model.pt
```
This results in the prediction result image to be saved to `runs/segment/predic/image.jpg`

## Configuration
The training configuration is specified in `configs/train_configs.yaml`. Here is an example configuration:

```bash
experiment_name: "<your_experiment_name>"
model: "yolov8n-seg" # or any other yolov8 segmentation models
data:
  coco_path: "path/to/annotations.json"
hyperparams:
  epochs: 100
  imgsz: 614
  batch: 16
  device: 0
  patience: 20
  lr0: 0.00269
  optimizer: AdamW
  weight_decay: 0.00015
augment:
  hsv_h: 0.01148
  hsv_s: 0.53554
  hsv_v: 0.13636
  degrees: 0.0
  translate: 0.12431
  scale: 0.07643
  shear: 0.0
  perspective: 0.0
  flipud: 0.0
  fliplr: 0.08631
  mosaic: 0.42551
  mixup: 0.0
  copy_paste: 0.0
  ```

## Hyperparameters
### Training Hyperparameters
- epochs: Number of training epochs.
- imgsz: Image size for training.
- batch: Batch size for training.
- device: Device to use for training (e.g., 0 for the first GPU).
- patience: Number of epochs to wait for improvement before early stopping.
- lr0: Initial learning rate.
- optimizer: Optimizer to use (e.g., AdamW).
- weight_decay: Weight decay for regularization. 

### Data Augmentation Hyperparameters
- hsv_h: Hue shift.
- hsv_s: Saturation shift.
- hsv_v: Value shift.
- degrees: Rotation degrees.
- translate: Translation factor.
- scale: Scaling factor.
- shear: Shearing factor.
- perspective: Perspective transformation factor.
- flipud: Vertical flip probability.
- fliplr: Horizontal flip probability.
- mosaic: Mosaic augmentation probability.
- mixup: MixUp augmentation probability.
- copy_paste: Copy-Paste augmentation probability.