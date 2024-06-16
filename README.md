# CO2 Emissions Prediction Project

This project aims to predict the CO2 emissions of different vehicles based on their features using various regression models. The dataset used contains information about various car models, including make, model, vehicle class, engine size, cylinders, transmission, fuel type, and fuel consumption. The goal is to create a reliable model that can predict the CO2 emissions for a given set of vehicle features.

![CO2 Emissions](images/co2_emissions_banner.jpg)

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Data Preprocessing](#data-preprocessing)
5. [Feature Selection](#feature-selection)
6. [Model Training and Evaluation](#model-training-and-evaluation)
7. [Conclusion](#conclusion)
8. [Installation](#installation)
9. [Usage](#usage)
10. [Contributing](#contributing)
11. [License](#license)

## Project Overview

The primary objective of this project is to build a machine learning model to predict CO2 emissions from vehicles based on their characteristics. Multiple regression algorithms were used, including Linear Regression, Lasso Regression, Ridge Regression, Decision Tree Regression, Random Forest Regression, and Support Vector Regression (SVR). The project is divided into the following phases:

- Data Collection and Exploration
- Data Preprocessing
- Feature Engineering and Selection
- Model Training and Evaluation
- Model Comparison

## Dataset

The dataset used in this project contains the following columns:
- **Make**: The manufacturer of the vehicle
- **Model**: The model of the vehicle
- **Vehicle Class**: The class of the vehicle (e.g., SUV, Sedan)
- **Engine Size (L)**: The size of the vehicle's engine in liters
- **Cylinders**: The number of cylinders in the vehicle's engine
- **Transmission**: The type of transmission (e.g., Automatic, Manual)
- **Fuel Type**: The type of fuel used by the vehicle (e.g., Regular gasoline, Diesel)
- **Fuel Consumption City (L/100 km)**: Fuel consumption in city driving
- **Fuel Consumption Hwy (L/100 km)**: Fuel consumption on highways
- **Fuel Consumption Comb (L/100 km)**: Combined fuel consumption
- **Fuel Consumption Comb (mpg)**: Combined fuel consumption in miles per gallon
- **CO2 Emissions (g/km)**: The target variable representing CO2 emissions in grams per kilometer

## Exploratory Data Analysis (EDA)

EDA was conducted to understand the distribution of features and their relationships with the target variable, CO2 emissions. Various visualizations were used to identify patterns and correlations.

![Correlation Heatmap](images/correlation_heatmap.png)
*Figure 1: Correlation Heatmap*

## Data Preprocessing

Data preprocessing steps included:
- Renaming columns for better readability
- Handling missing values
- Encoding categorical variables
- Standardizing numerical features

## Feature Selection

Feature selection was performed using the Chi-Square test to identify the most relevant features for predicting CO2 emissions.

![Feature Importance](images/feature_importance.png)
*Figure 2: Feature Importance*

## Model Training and Evaluation

The following regression models were trained and evaluated:
- **Linear Regression**
- **Lasso Regression**
- **Ridge Regression**
- **Decision Tree Regression**
- **Random Forest Regression**
- **Support Vector Regression (SVR)**

Model performance was assessed using metrics such as RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), and R2 Score.

![Model Comparison](images/model_comparison.png)
*Figure 3: Model Comparison*

## Conclusion

The Random Forest Regression model provided the best performance in terms of RMSE, MAE, and R2 Score. This model can be used to predict CO2 emissions for a given set of vehicle features with high accuracy.

## Installation

To run this project, you need to have Python installed along with the following libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install the required libraries using the following command:

```bash
pip install -r requirements.txt
