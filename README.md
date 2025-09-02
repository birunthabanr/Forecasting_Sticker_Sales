# 🏪 Sticker Sales Forecasting - Kaggle Playground Series S5E1

A machine learning solution for time series forecasting that predicts sticker sales across multiple stores and countries. This project was developed for Kaggle's Playground Series Season 5 Episode 1 competition, featuring an ensemble stacking model and interactive web application.

## 🎯 Overview

This project implements a comprehensive forecasting system using ensemble machine learning techniques to predict sales patterns with high accuracy. The solution combines multiple gradient boosting algorithms in a stacking framework and provides an intuitive web interface for real-time predictions.

### Key Features
- 🤖 **Ensemble Stacking Model** combining XGBoost, LightGBM, and CatBoost
- 🔧 **Automated Feature Engineering** with 10+ temporal and cyclic features
- 🌐 **Interactive Web Application** built with Streamlit
- 📊 **Advanced Time Series Analysis** with seasonal pattern recognition

## 🏆 Competition Results

- **Competition**: Kaggle Playground Series S5E1 - Forecasting Sticker Sales
- **Participants**: 2,722 teams globally
- **Challenge**: Predict multi-year sales data across different stores and countries
- **Approach**: Ensemble stacking with advanced feature engineering

## 📁 Project Structure
sticker-sales-forecasting/
│
├── data/                     # Data directory
│   ├── train.csv             # Training dataset
│   ├── test.csv              # Test dataset
│   └── sample_submission.csv # Submission format
│
├── notebooks/                # Jupyter notebooks
│   ├── 01_data_exploration.ipynb   # Data analysis and EDA
│   ├── 02_feature_engineering.ipynb # Feature creation
│   └── 03_model_training.ipynb     # Model development
│
├── models/                   # Trained models
│   └── stacking_model.joblib # Final ensemble model
│
├── app/                      # Streamlit application
│   └── streamlit_app.py      # Web application
│
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation