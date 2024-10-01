# Yap1 Expression Prediction in Cervical Cancer using Deep Learning

## Project Overview
This project aims to predict gene expression in cancer types (cervical or head and neck in this case) using deep learning techniques. It utilizes a modified ResNet101 architecture to analyze histopathological images and predict gene expression levels.

## Features
- Data preprocessing and augmentation
- Modified ResNet101 model implementation
- Hyperparameter tuning
- Model evaluation with various metrics (AUR[Uploading config-yaml.txt…]()
OC, Precision, Recall, F1 Score)
- Visualization of results and model embeddings

## Requirements
- Python 3.7+
- PyTorch
- torchvision
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- UMAP

## Installation
1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```


## Usage
1. Prepare your data:
   - Place your histopathological images in the `Preprocessed_Tiles` directory
   - Ensure your ground truth data is in an Excel file named `GROUND TRUTH.xlsx`
2. Clarify the gene that you are interested in. Set the threshold in your database using mean and std dev. Focus on a single cancer type or multiple.
3. Run the Jupyter notebook `Gene_Expression_Prediction.ipynb`

## Results
The model achieves state-of-the-art performance in predicting expressions lying above or below the specified threshold, with detailed metrics available in the notebook.

## Data Structure
DesiredGene(Replace with your gene)-expression-prediction/

yap1-expression-prediction/
│
├── data/
│   └── Preprocessed_Tiles/
│
├── models/
│   └── best_model.pth
│
├── notebooks/
│   └── Yap1_Expression_Prediction.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── tests/
│   ├── __init__.py
│   ├── test_data_processing.py
│   ├── test_model.py
│   └── test_evaluate.py
│
├── utils/
│   └── visualization.py
│
├── .gitignore
├── config.yaml
├── README.md
├── requirements.txt
└── setup.py
└── setup.py
