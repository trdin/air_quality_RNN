# Air Quality Prediction Readme

This repository contains code for predicting air quality using machine learning models. The code is written in Python and utilizes various libraries for data preprocessing, feature engineering, and model training.

## Prerequisites

- Python 3.x
- Libraries: pandas, numpy, scikit-learn, tensorflow, matplotlib, seaborn

## Getting Started

1. Install the required dependencies:

```bash
   pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
```
2. Clone the repository:
```bash
    git clone https://github.com/trdin/air_quality_RNN.git
    cd air-quality-prediction
```
3. Download the dataset (RV2_UPP_IIR_SIPIA.csv) and place it in the root folder.
4. Run the Jupyter notebook:
```bash
    jupyter notebook
```

## Code Structure

### Data Loading and Preprocessing:

- Reads the dataset from "RV2_UPP_IIR_SIPIA.csv".
- Converts the 'Date' column to datetime and sorts the data.
- Handles missing values in the dataset.

### Exploratory Data Analysis (EDA):

- Displays histograms and statistical information for numerical columns.
- Utilizes RandomForestRegressor to impute missing values.

### Feature Engineering:

- Applies log transformation to right-skewed data.
- Creates new features based on interactions and extracts time-related features.

### Information Gain Analysis:

- Calculates information gain for feature selection.
- Selects features with information gain above a threshold.

### Model Training:

- Splits the data into training and testing sets.
- Normalizes and preprocesses the data for model training.
- Builds and trains SimpleRNN, GRU, and LSTM models.

### Model Evaluation:

- Calculates and prints metrics such as MAE, MSE, and EVS for each model.
- Visualizes predictions against actual values for evaluation.


