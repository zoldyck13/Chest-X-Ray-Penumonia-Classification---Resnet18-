# Chest X-Ray Penumonia Classification

This project implements a deep learning pipeline for classifying chest X-ray images into **NORMAL** or **PENUMONIA** using **PyTorch** and **ResNet-18**. It also includes **Grad-CAM** visualization for model interperability.



___

## Project Structure

```
.
├── data
│   ├── processed
│   │   ├── test/
│   │   └── train/
│   │   └── val/
│
│   
├── runs
│   ├── gradcam_output/
│   ├── penuumonia_resnet18/
│   
├── src
│   ├── data/
│   ├── models/
│   ├── evalute.py
│   ├── export.py
│   ├── inference.py
│   ├── model_final.pt
│   ├── train.py
│
└── README.md

```
___
##  Data Preparation

#### Run the data preparation script to split and apply augmentations:

```bash
python3 src/data/prepare_data.py
```


![prepare_data](/assets/data_prepare.png)

This **prepare_data.p** suppose you have the two class images folders on the **data/raw/..** path:

```
├── data/raw/
    ├── NORMAL
    └── PNEUMONIA
```

I use 70% as spilt ratio for the training images, and you can change this by just modifiy this part here on **src/datda/prepare_data.py**:

```py

SPLIT_RATIO = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}
```
You can change the raw data on the **src/data/prepare_data.py**: 

```py
RAW_DIR = Path("../../data/raw/chest_xray")
PROCESSED_DIR = Path("../../data/processed/chest_xray")
```
___
## Data Loaders


The `loaders.py` script builds PyTorch DataLoaders for training, validation, and testing:

```py
from src.data.loaders import build_loaders

train_loader, val_loader, test_loader, class_to_idx = build_loaders()

```
* Applies transformations and normalization for ResNet-18.
* Supports batch size and number of workers configuration.
___
## Model


* ResNet-18 is used, optionally pretrained on ImageNet.
* Final fully connected layer is replaced to output 2 classes.

```py
from src.models.resnet import build_resnet18

model = build_resnet18(num_classes=2, pretrained=True)


```

___
## Training


Train the model using:

```py
python3 src/train.py

```

* Tracks training & validation losses.
* Saves the best model to `runs/<experiment_name>/best_resnet18.pt`.

![training_resnet.png](/assets/training_resnet.png)

___
## Evaluation

Evaluate the model on the test set and generate Grad-CAM visualizations:
```py
python3 src/evaluate.py --model runs/<experiment_name>/best_resnet18.pt
```

* Prints class probabilities.
* Saves Grad-CAM overlays for visualizing model focus.

___
## Inference on New Images

Perform prediction and visualize Grad-CAM for a single image:
```py
python3 src/inference.py \
    --model runs/<experiment_name>/best_resnet18.pt \
    --image data/raw/chest_xray/NORMAL/NORMAL1.jpeg \
    --out result.png

```

* Outputs predicted class and probabilities.
* Saves Grad-CAM overlay + heatmap in `result.png`.

![inference](/assets/inference.png)


![gradcam_result.png](/assets/gradcam_result.png)


___
## Requirements
* +Python 3.8
* PyTorch 2.x

* torchvision
* PIL / Pillow

* OpenCV

* Matplotlib

* tqdm


```zsh
pip install torch torchvision pillow opencv-python matplotlib tqdm
```

___
## Notes

* Ensure `data/raw/chest_xray/` contains all images before running prepare_data.py.

* Grad-CAM is implemented on the last convolutional layer of ResNet-18 (`layer4`).

* You can adjust image size, batch size, and augmentations in `loaders.py` and `prepare_data.py`.
___


### Author
- [Ali m. Saad |  Computer Engineer | Deep Learning Enthusiast](https://github.com/zoldyck13)