# House Price Prediction

This project implements a full **machine learning pipeline** to predict house prices using engineered features. It combines robust preprocessing, advanced feature engineering, and multiple regression models to provide accurate price estimates.

---

## Project Structure

The project is organized for clarity and modularity:

- **`DataSets/`**  
  Contains original datasets:  
  - Training data (`train.csv`)  
  - Test data (`test.csv`) 
  - Test data integrated within the main script and to download it within application to test the final model (`data_test.csv`)  

- **`configs/`**  
  Centralizes project settings, feature definitions, and hyperparameters for models.

- **`intern_src/pipelines_package/`**  
  Contains modular code for pipelines:  
  - `models_pipeline.py`: builds preprocessing + model pipelines  
  - `predict.py`: functions for making predictions  

- **`intern_src/utils_package/`**  
  Utility functions for data ingestion, preprocessing, transformations, and saving/loading models.

- **`models/`**  
  Stores the best trained model pipeline (e.g., `.pkl`).

- **`app.py`**  
  FastAPI application for serving predictions via web interface or CSV upload.

- **`salehouse_prediction.py`**  
  Script to generate predictions from saved models on test data.

- **`README.md`**  
  Overview of the project and usage instructions.

- **`requirements.txt`**  
  Python dependencies needed for the project.

---

## Installation

### Requirements

- Python 3.10+

### Setup

```bash
git clone <repository-url>
cd mlrg-project
pip install -r requirements.txt
```
---

## Running the Project

### 1. Training the Models

To train and save the best model pipeline:

```bash
python3 intern_src/pipelines_package/train_model.py
```
### 2. Batch Prediction

```bash
python3 salehouse_prediction.py
```

### 3. Running the FastAPI Web Application

```bash
uvicorn app:app --reload
```