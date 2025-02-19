# Churn Prediction Model

This project contains EDA & machine learning model for predicting customer churn on kaggle dataset "Customer Churn Dataset" by Muhammad Shahid Azeem.

## Dataset

[Customer Churn Dataset](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset)

## Skills shown

- Usage of pandas, sklearn, matplotlib, seaborn and joblib
- Exploratory data analysis (EDA)
- Feature extraction, feature engineering
- Data visualisation
- Component reduction using Principal Component Analysis (PCA)
- Creation of ML models and testing them:
  - Logistic Regression
  - SVM
  - SVM with GridSearch
- Saving and loading of ML model
- Inference

## Requirements

- Python 3.11.7
- Dependencies listed in `requirements.txt`

## Setup

1. Ensure you have Python 3.11.7 installed. You can download it from [python.org](https://www.python.org/downloads/).
2. Clone this repository
3. Create & activate a new virtual environment
4. Install the required packages: `pip install -r requirements.txt`
5. Run the Jupyter notebook: `jupyter notebook churn_prediction_model.ipynb`

## Files

- `churn_prediction_model.ipynb`: Main Jupyter notebook containing the model
- `inference.py`: Script for running predictions using the trained model
- `dataset/`: Directory containing the dataset (if applicable)

## Model Files

Various joblib files are present which contain different versions of the trained model and preprocessing objects:

- `pca_reduce2.joblib`: PCA model for dimensionality reduction
- `scaler2.joblib`: Feature scaler
- `svm_churn_model2.joblib`: SVM model for churn prediction

## Usage

Jupyter notebook cells are run using SHIFT + Enter. Please note that running SVM with GridSearch will take a long time because it goes through every option that is stated in the param_grid. Which means it would run SVM training 48 times (in this case) in order to find best combination of hyperparameters. 

Inference should be simply run in CMD and filled with given information. If not otherwise remarked number should be entered.

# Churn-Prediction-Model