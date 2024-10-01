# Code Explanation: Yap1 Expression Prediction in Cervical Cancer

This document provides a detailed explanation of the main components and workflow of the Yap1 Expression Prediction project.

## Data Preparation and Preprocessing

1. **Loading Ground Truth Data**:
   - The script loads the ground truth data from an Excel file containing Yap1 expression information.
   - It processes the case numbers and creates a binary label for Yap1 expression.

2. **Image Data Processing**:
   - The script walks through the `Preprocessed_Tiles` directory to find image files.
   - It matches each image with its corresponding Yap1 expression label.

3. **Data Augmentation**:
   - Various image transformations are applied to augment the training data, including resizing, flipping, rotation, and color jittering.

## Model Architecture

The project uses a modified ResNet101 architecture:
- The base model is a pretrained ResNet101.
- The final fully connected layer is replaced with a custom layer for binary classification.
- Dropout is added for regularization.

## Training Process

1. **Data Splitting**:
   - The dataset is split into training (70%), validation (10%), and test (30%) sets.

2. **Model Training**:
   - The model is trained using Binary Cross-Entropy loss and the AdamW optimizer.
   - Learning rate scheduling is implemented using ReduceLROnPlateau.
   - Early stopping is used to prevent overfitting.

3. **Hyperparameter Tuning**:
   - The script includes a hyperparameter tuning process, exploring different learning rates and batch sizes.

## Evaluation

The model is evaluated using several metrics:
- Area Under the Receiver Operating Characteristic (AUROC)
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Visualization

1. **Learning Curves**:
   - Training and validation losses are plotted to visualize the learning process.

2. **UMAP Visualization**:
   - UMAP is used to visualize the learned embeddings in a 2D space.

3. **Sample Image Annotation**:
   - The script includes functionality to annotate sample images, highlighting regions of interest.

## Logging and Saving

- TensorBoard is used for logging training progress and metrics.
- The best model is saved based on validation performance.

This code provides a comprehensive pipeline for training and evaluating a deep learning model for Yap1 expression prediction in cervical cancer histopathological images.
