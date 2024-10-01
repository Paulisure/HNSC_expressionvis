# Yap1 Expression Prediction in Cervical Cancer using Deep Learning

## Project Overview
This project aims to predict gene expression in cancer types (cervical or head and neck in this case) using deep learning techniques. It utilizes a modified ResNet101 architecture to analyze histopathological images and predict gene expression levels.

![image](https://github.com/user-attachments/assets/6c9de619-2f32-4dba-a264-a3bdc9770be5)


## Data Preparation
An instance of the image preparation is shown below. Out of 791 samples available in the NCI’s database for TCGA-HNSC, 757 of these were tumor samples that also contained gene expression profiles. Using these 757 samples, the WSIs were processed into tiles using OpenSlide, and resized to 1024 x 1024 pixels. 
   
![image](https://github.com/user-attachments/assets/b8c34c2e-aec0-4f84-b50b-d655eb672b3c)

## About this project:
The cutoff for YAP1 overexpression was determined after outliers were identified within the TCGA HNSC and normal control datasets by analyzing the distribution of YAP1 expression levels across all samples. Outliers were removed from the dataset using the Tukey method from the datasets. Following outlier removal, the cutoff for YAP1 overexpression was determined using the mean plus one standard deviation to establish a baseline cutoff value. Additionally, the biological context and significance of YAP1 expression cutoff was considered when determining this signal since potential benefit from YAP targeting therapies may occur on the lower end of the YAP-dependent cancer threshold. These values created the ground truth referenced for labeling tile images in ResNet. The metadata associated with the cases used can be found in the Supplemental Table. 
The models were constructed and trained using the Python library PyTorch. We utilized a pre-trained ResNet101 as the base model, adapted by appending fully connected layers configured with 512, 256, and 1 output units respectively, incorporating ReLU activations for non-linearity and dropout layers between each fully connected layer to prevent overfitting. Specifically, we used a dropout rate of 0.5 to deactivate neuron connections randomly during training phases. 
The final output layer uses a sigmoid activation function to yield the probability of binary classes. The loss was computed using Binary Cross-Entropy Loss (BCEWithLogitsLoss). For optimization, we used the AdamW optimizer with an initial learning rate of 0.001. To adjust the learning rate efficiently, ReduceLROnPlateau was used with a factor of 0.4 when the validation loss plateaus, enhancing the ability of convergence. 
Training involved shuffling the data with random transformations including horizontal flip, vertical flip, color jitter, and rotation up to 20 degrees. 
The model was set to train for up to 70 epochs, but with an early stopping mechanism if the validation loss did not improve for 35 consecutive epochs.
The dataset comprised 757 samples which were converted into 1514 1024x1024 tiles. These tiles were then stratified into a training set of 1060 tiles (70%), an evaluation set of 151 tiles (10%) and a testing cohort of 303 tiles (20%), based on predefined criteria. After training, the model achieved an Area Under the Receiver Operating Characteristic (AUROC) of 0.9383, showcasing high accuracy in distinguishing between the classes. The model demonstrated a Precision (Positive Predictive Value) of 0.8780 and Recall (Sensitivity) of 0.8727, with a Specificity of 0.8925 and an F1 Score of 0.8754. The Negative Predictive Value (NPV) stood at 0.9125.
The trained model was leveraged to predict outcomes for the testing cohort, providing insights into the prognostic value of the model in practical scenarios.


![image](https://github.com/user-attachments/assets/37e74bfc-f970-4253-b3ff-b3d38bf6a201)


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

![image](https://github.com/user-attachments/assets/cebb4f29-633a-4537-9c7f-ddf75364842c)

