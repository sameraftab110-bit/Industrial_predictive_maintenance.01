# Industrial Predictive Maintenance & Real-Time Machine Health Monitoring System

## Overview

An end-to-end industrial machine health monitoring and predictive maintenance system that uses machine sensor data, feature engineering, supervised machine learning, and Streamlit to predict machine failure risk.

## Project Objective

The system analyzes industrial machine parameters such as:

- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear
- Temperature difference
- RPM type
- Torque category
- Tool wear type
- Total failure count
- High-risk flag

The trained machine learning model predicts whether a machine is likely to experience failure.

## System Architecture

Machine Sensor Data
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Machine Learning
        ↓
Failure Prediction
        ↓
Streamlit Dashboard
        ↓
Machine Health Monitoring

## Machine Learning

Model:

Logistic Regression

Class imbalance:

Balanced class weights

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Cross-validation

Final F1 Score:

0.9565

## Dashboard

The Streamlit dashboard provides:

- Continuous machine-data monitoring
- Automatic machine failure prediction
- Healthy machine count
- Failure-risk count
- Machine monitoring table
- Failure probability
- Manual machine-health prediction
- Sensor parameter visualization

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- Matplotlib
- Seaborn
- Git
- GitHub

## Project Structure

```text
Data/
dashboards/
models/
notebooks/
reports/
src/
tests/
README.md
requirements.txt

