from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DATASET_DIR = PACKAGE_ROOT / "DataSets"
MLFLOW_TRACKING_DIR = PACKAGE_ROOT / "mlflow_artifacts"

# Package Overview
# Data Files
TRAINING_DATA_FILE = "train.csv"
TEST_DATA_FILE = "test.csv"
MLFLOW_ARTIFACTS = ""

# Variables
# The variable we are attempting to predict (sale price)
TARGET = "SalePrice"

PIPELINE_NAME = "best_model"
PIPELINE_SAVE_NAME = "best_model_output_v_1"

# Will cause syntax errors since they begin with numbers
VARIABLES_TO_RENAME = {
    "1stFlrSF": "FirstFlrSF",
    "2ndFlrSF": "SecondFlrSF",
    "3SsnPorch": "ThreeSsnPorch",
}

# Liste des features attendues
FEATURES = [
    "Id",
    "MSSubClass",
    "MSZoning",
    "LotFrontage",
    "LotArea",
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
    "OverallQual",
    "OverallCond",
    "YearBuilt",
    "YearRemodAdd",
    "RoofStyle",
    "RoofMatl",
    "Exterior1st",
    "Exterior2nd",
    "MasVnrType",
    "MasVnrArea",
    "ExterQual",
    "ExterCond",
    "Foundation",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinSF1",
    "BsmtFinType2",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "Heating",
    "HeatingQC",
    "CentralAir",
    "Electrical",
    "FirstFlrSF",
    "SecondFlrSF",
    "LowQualFinSF",
    "GrLivArea",
    "BsmtFullBath",
    "BsmtHalfBath",
    "FullBath",
    "HalfBath",
    "BedroomAbvGr",
    "KitchenAbvGr",
    "KitchenQual",
    "TotRmsAbvGrd",
    "Functional",
    "Fireplaces",
    "FireplaceQu",
    "GarageType",
    "GarageYrBlt",
    "GarageFinish",
    "GarageCars",
    "GarageArea",
    "GarageQual",
    "GarageCond",
    "PavedDrive",
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "ThreeSsnPorch",
    "ScreenPorch",
    "PoolArea",
    "PoolQC",
    "Fence",
    "MiscFeature",
    "MiscVal",
    "MoSold",
    "YrSold",
    "SaleType",
    "SaleCondition",
]

# set train/test split
TEST_SIZE = 0.2

# to set the random seed
# RANDOM_STATE = 0

# categorical variables
CATEGORICAL_VARS = [
    "MSSubClass",
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
"""""" """""" """"""
# categorical variables to encode
# CATEGORICAL_VARS = ['MSSubClass',  'MSZoning',  'LotShape',  'LandContour',
#                    'LotConfig', 'Neighborhood', 'RoofStyle', 'Exterior1st',
#                   'Foundation', 'CentralAir', 'Functional', 'PavedDrive',
#                    'SaleCondition']


# variables to map
QUAL_VARS = ["ExterQual", "BsmtQual", "HeatingQC", "KitchenQual", "FireplaceQu"]

EXPOSURE_VARS = ["BsmtExposure"]

FINISH_VARS = ["BsmtFinType1"]

GARAGE_VARS = ["GarageFinish"]

FENCE_VARS = ["Fence"]


# variable mappings
QUAL_MAPPINGS = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5, "Missing": 0, "NA": 0}

EXPOSURE_MAPPINGS = {"No": 1, "Mn": 2, "Av": 3, "Gd": 4}

FINISH_MAPPINGS = {
    "Missing": 0,
    "NA": 0,
    "Unf": 1,
    "LwQ": 2,
    "Rec": 3,
    "BLQ": 4,
    "ALQ": 5,
    "GLQ": 6,
}

GARAGE_MAPPINGS = {"Missing": 0, "NA": 0, "Unf": 1, "RFn": 2, "Fin": 3}
FENCE_MAPPINGS = {"Missing": 0, "NA": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4}


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
