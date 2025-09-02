# From scikit learn
import xgboost as xgb
from feature_engine.encoding import OrdinalEncoder, RareLabelEncoder

# From feature_engine
from feature_engine.imputation import (
    AddMissingIndicator,
    CategoricalImputer,
    MeanMedianImputer,
)
from feature_engine.selection import DropFeatures
from feature_engine.transformation import LogTransformer, YeoJohnsonTransformer
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, MinMaxScaler

# From config_dir file
from configs import core, models_params

# From internal librairies
from intern_src.utils_package.data_transformation import TemporalTransformer


def create_base_pipeline():
    """Creates a generated pipeline"""
    model_pipeline = Pipeline(
        [
            # categorical missing values
            (
                "missing_imputation",
                CategoricalImputer(
                    imputation_method="missing",
                    variables=core.CATEGORICAL_FREQUENT_MISSING,
                ),
            ),
            (
                "frequent_imputation",
                CategoricalImputer(
                    imputation_method="frequent",
                    variables=core.CATEGORICAL_FREQUENT_CATEGORY,
                ),
            ),
            # numerical missing values
            (
                "missing indicator",
                AddMissingIndicator(variables=core.NUMERICAL_MISSING_VARS),
            ),
            (
                "mean imputation",
                MeanMedianImputer(
                    imputation_method="mean", variables=core.NUMERICAL_MISSING_VARS
                ),
            ),
            # temporal variables
            (
                "temporal transformer",
                TemporalTransformer(
                    variables=core.TEMPORAL_VARS, reference_variable=core.REF_VARS
                ),
            ),
            ("drop feature", DropFeatures(features_to_drop=core.REF_VARS)),
            ("log transformation", LogTransformer(variables=core.CONTINUOUS_VARS_LOG)),
            (
                "Yeo_Johnson transformation",
                YeoJohnsonTransformer(variables=core.CONTINUOUS_VARS_YEO),
            ),
            (
                "binary transformation",
                SklearnTransformerWrapper(
                    transformer=Binarizer(threshold=0), variables=core.BINARIZE_VARS
                ),
            ),
            # encoder the categorical variables
            (
                "rare labels",
                RareLabelEncoder(
                    tol=0.01, n_categories=1, variables=core.CATEGORICAL_VARS
                ),
            ),
            (
                "ordinal encoder",
                OrdinalEncoder(
                    encoding_method="ordered", variables=core.CATEGORICAL_VARS
                ),
            ),
        ]
    )
    return model_pipeline


def create_lasso_pipeline(alpha=0.001, max_iter=100, tol=1e-2):
    """Crée un pipeline complet avec Lasso
    et des hyperparamètres configurables."""
    base_pipeline = create_base_pipeline()

    return Pipeline(
        [
            *base_pipeline.steps,
            ("scaler", MinMaxScaler()),
            (
                "estimator",
                Lasso(
                    alpha=alpha,
                    max_iter=max_iter,
                    tol=tol,
                    random_state=models_params.RANDOM_STATE,
                ),
            ),
        ]
    )


def create_xgb_pipeline(n_estimators, learning_rate, max_depth):
    """Crée un pipeline complet avec XGBoost avec des hyperparamètres donnés."""
    base_pipeline = create_base_pipeline()

    return Pipeline(
        [
            *base_pipeline.steps,
            ("scaler", MinMaxScaler()),
            (
                "estimator",
                xgb.XGBRegressor(
                    objective="reg:squarederror",
                    random_state=models_params.RANDOM_STATE,
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                ),
            ),
        ]
    )


def create_mlp_pipeline(hidden_layer, activation):
    base_pipeline = create_base_pipeline()

    return Pipeline(
        [
            *base_pipeline.steps,
            ("scaler", MinMaxScaler()),
            (
                "estimator",
                MLPRegressor(
                    random_state=models_params.RANDOM_STATE,
                    hidden_layer_sizes=hidden_layer,
                    activation=activation,
                ),
            ),
        ]
    )
