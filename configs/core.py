import os
from pathlib import Path

# Define the MLflow tracking URI to use ml_project directory
MLFLOW_TRACKING_DIR = os.path.join(os.getcwd(), "mlflow_artifacts")

PACKAGE_ROOT = Path(__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DATASET_DIR = PACKAGE_ROOT / "DataSets"

# Package Overview
# Data Files
TRAINING_DATA_FILE = "train.csv"
TEST_DATA_FILE = "test.csv"
MLFLOW_ARTIFACTS = ""

# Variables
# The variable we are attempting to predict (sale price)
TARGET = "SalePrice"

PIPELINE_NAME = "ML_regression_model"
PIPELINE_SAVE_NAME = "regression_model_output_v_1"

# Will cause syntax errors since they begin with numbers
VARIABLES_TO_RENAME = {
    "1stFlrSF": "FirstFlrSF",
    "2ndFlrSF": "SecondFlrSF",
    "3SsnPorch": "ThreeSsnPorch",
}

FEATURES = [
    "MSSubClass",
    "MSZoning",
    "LotFrontage",
    "LotShape",
    "LandContour",
    "LotConfig",
    "Neighborhood",
    "OverallQual",
    "OverallCond",
    "YearRemodAdd",
    "YearBuilt",
    "GarageYrBlt",
    "RoofStyle",
    "Exterior1st",
    "ExterQual",
    "Foundation",
    "BsmtQual",
    "BsmtExposure",
    "BsmtFinType1",
    "HeatingQC",
    "CentralAir",
    "ThreeSsnPorch",  # renamed
    "FirstFlrSF",  # renamed
    "SecondFlrSF",  # renamed
    "GrLivArea",
    "BsmtFullBath",
    "HalfBath",
    "KitchenQual",
    "TotRmsAbvGrd",
    "Functional",
    "Fireplaces",
    "FireplaceQu",
    "GarageFinish",
    "GarageCars",
    "GarageArea",
    "PavedDrive",
    "WoodDeckSF",
    "ScreenPorch",
    "SaleCondition",
    # this one is only to calculate temporal variable:
    "YrSold",
]

# set train/test split
TEST_SIZE = 0.2

# to set the random seed
#RANDOM_STATE = 0

#ALPHA = [0.1, 0.01, 0.001, 0.0001]

# categorical variables
CATEGORICAL_VARS = [
    "MSZoning",
    "Street",
    "Alley",
    "LotShape",
    "LandContour",
    "Utilities",
    "LotConfig",
    "LandSlope",
    "Neighborhood",
    "Condition1",
    "Condition2",
    "BldgType",
    "HouseStyle",
    "RoofStyle",
    "RoofMatl",
    "Exterior1st",
    "Exterior2nd",
    "MasVnrType",
    "ExterQual",
    "ExterCond",
    "Foundation",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "Heating",
    "HeatingQC",
    "CentralAir",
    "Electrical",
    "KitchenQual",
    "Functional",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PavedDrive",
    "PoolQC",
    "Fence",
    "MiscFeature",
    "SaleType",
    "SaleCondition",
    "MSSubClass",
]

# categorical variables with NA in train set
CATEGORICAL_FREQUENT_CATEGORY = [
    "BsmtQual",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "GarageFinish",
    "BsmtCond",
    "Electrical",
    "GarageType",
    "GarageQual",
    "GarageCond",
]

CATEGORICAL_FREQUENT_MISSING = [
    "FireplaceQu",
    "Alley",
    "MasVnrType",
    "PoolQC",
    "Fence",
    "MiscFeature",
]

NUMERICAL_MISSING_VARS = ["LotFrontage", "MasVnrArea", "GarageYrBlt"]

TEMPORAL_VARS = ["YearRemodAdd", "YearBuilt", "GarageYrBlt"]

REF_VARS = "YrSold"

# variables to log transform
CONTINUOUS_VARS_LOG = ["LotFrontage", "FirstFlrSF", "GrLivArea", "LotArea"]

CONTINUOUS_VARS_YEO = [
    "MasVnrArea",
    "BsmtFinSF1",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "SecondFlrSF",
    "GarageArea",
    "WoodDeckSF",
    "OpenPorchSF",
]

BINARIZE_VARS = [
    "BsmtFinSF2",
    "LowQualFinSF",
    "EnclosedPorch",
    "ThreeSsnPorch",
    "MiscVal",
    "ScreenPorch",
]
