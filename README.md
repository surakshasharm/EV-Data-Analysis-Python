# 🔌 Electric Vehicle Data Analysis

This repository presents an exploratory data analysis (EDA) project focusing on electric vehicle (EV) registrations in Washington State, using the Electric_Vehicle_Population_DataSet. The goal is to uncover trends, patterns, and environmental impact insights related to Battery Electric Vehicles (BEVs) and Plug-in Hybrid Electric Vehicles (PHEVs).

## 📁 Repository Structure

├── dataAnalysis.py                 # Main Python script for data analysis

├── Electric_Vehicle_Population_DataSet.csv  # Dataset used for analysis

├── plottings/                      # Folder containing saved visualizations

└── README.md                       # Project description and instructions

## 📊 Key Features
### Data Cleaning & Exploration:

Removal of missing values

Summary statistics and data structure overvie0w

### Visualizations:

BEV vs. PHEV proportions

Yearly registration trends

Top manufacturers and models

Regional distributions (by County and City)

Annual CO₂ savings estimation

Correlation heatmap of numerical features

### Statistical Analysis:

T-test to compare model years of BEVs vs PHEVs

Z-score based outlier detection of vehicle makes

## 📈 Sample Visualizations

Saved plots can be found in the plottings/ directory. These include:

EV Type Distribution Pie Chart

EV Registrations Over Model Years

Top Makes and Models

Regional Heatmaps and Bar Charts

Estimated CO₂ Savings

## 📦 Libraries Used

pandas

numpy

matplotlib

seaborn

scipy

## 📂 Dataset

Filename: Electric_Vehicle_Population_DataSet.csv

Source: [Data.gov - Electric Vehicle Population Data](https://catalog.data.gov/dataset/electric-vehicle-population-data)

Attributes Used: Electric Vehicle Type, Make, Model, County, City, Model Year






