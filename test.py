from configs import core, models_params
from intern_src.utils_package.data_ingestion import load_dataset
from intern_src.utils_package.data_transformation import logarithm_transformer
from sklearn.model_selection import train_test_split


def show_nan_columns(df):
    nan_cols = df.isna().sum()
    nan_cols = nan_cols[nan_cols > 0]

    if nan_cols.empty:
        print(f" Pas de NaN")
    else:
        print(f"\nColonnes avec NaN):\n{nan_cols.to_string()}")


# 1. Chargement des données
data = load_dataset(file_name=core.TRAINING_DATA_FILE)
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(["Id", core.TARGET], axis=1),
    data[core.TARGET],
    test_size=core.TEST_SIZE,
    random_state=models_params.RANDOM_STATE,
)
y_train = logarithm_transformer(data=y_train)
y_test = logarithm_transformer(data=y_test)
import numpy as np

print("+++++++++++++++++++++++++++++++++++++++++++++++++")
print(X_train.columns)

    #print(f"Le type de données de la colonne '{c}' est : {X_train[c].dtype}")
#print(f"check the NaN of the X_train: \n {X_train.isnull().sum()}")
#show_nan_columns(df=X_train)
#print("****************************")

