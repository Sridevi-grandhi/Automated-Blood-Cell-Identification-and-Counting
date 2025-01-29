# Blood Cell Detection Using YOLO NAS

This project aims to perform object detection and counting of blood cells using a YOLO NAS model. The notebook preprocesses blood cell image data, formats it for YOLO, trains a YOLO NAS model, and performs inference on test data.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Dependencies](#dependencies)
4. [Setup Instructions](#setup-instructions)
5. [Notebook Workflow](#notebook-workflow)
6. [Results](#results)
7. [References](#references)

---

## Overview

The notebook includes the following steps:

- **Data Preprocessing:** Extract and process data labels in YOLO format from XML files.
- **Dataset Splitting:** Divide the dataset into training, validation, and test sets.
- **Model Training:** Train the YOLO NAS model for blood cell detection using the `super-gradients` library.
- **Inference:** Predict and visualize blood cell detections on unseen test images.

---

## Features

- **Custom Dataset Support:** Works with the BCCD dataset for blood cell images.
- **YOLO NAS Architecture:** Implements YOLO NAS, optimized for object detection tasks.
- **Visualization Tools:** Generates plots to visualize bounding boxes and predictions.
- **Data Augmentation:** Includes image augmentations to improve model robustness.

---

## Dependencies

The project requires the following libraries:

- Python 3.7 or higher
- Jupyter Notebook
- `super-gradients`
- `numpy`
- `pandas`
- `matplotlib`
- `sklearn`
- `cv2`
- `torch`
- `glob`
- Other utilities for dataset management and visualization.

---

## Setup Instructions

1. Clone the repository

2. Install the required Python packages:

   ```bash
   pip install super-gradients numpy pandas matplotlib scikit-learn opencv-python
   ```

3. Prepare the data:

   - Download the BCCD dataset and ensure its structure matches the YOLO requirements (Sample data is available in the repository)
   - Run the data preprocessing cells in the notebook.

4. Train the model:

   - Execute the training cells to train the YOLO NAS model using the `super-gradients` library.

5. Perform inference:
   - Run inference cells to test the model on unseen data and visualize the results.

---

## Notebook Workflow

1. **Data Preprocessing**

   - Convert annotations from XML to YOLO format.
   - Prepare dataframes to structure image-label mappings.
   - Divide the dataset into training, validation, and test splits.

2. **Training the YOLO NAS Model**

   - Configure training parameters (epochs, batch size, learning rate).
   - Fine-tune YOLO NAS on the training set.
   - Save model checkpoints for best performance.

3. **Inference and Visualization**
   - Load the trained model checkpoint.
   - Predict bounding boxes and class labels for test images.
   - Visualize predictions on sample test images.

---

## Results

- The notebook generates metrics like mAP (mean Average Precision) and loss values for model performance evaluation.
- Predictions on test data include bounding boxes with labels for each blood cell type.

---

## References

- [BCCD Dataset GitHub Repository](https://github.com/Shenggan/BCCD_Dataset)
- [YOLO NAS Implementation](https://github.com/Deci-AI/super-gradients)

---

## Future Improvements

- Extend support to other object detection tasks.
- Fine-tune the model with additional augmentation strategies for better generalization.
- Explore alternative YOLO architectures or backbone networks.

## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── bcc            <- Data from third party sources.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    |
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
