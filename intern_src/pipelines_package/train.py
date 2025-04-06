from itertools import product

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from configs import core, models_params
from intern_src.pipelines_package.models_pipeline import (
    create_lasso_pipeline, create_mlp_pipeline, create_xgb_pipeline)
from intern_src.utils_package.data_ingestion import load_dataset, save_pipeline
from intern_src.utils_package.data_transformation import logarithm_transformer


def run_training() -> None:
    """Train the model with multiple
    alpha values from the configuration.
    """
    # read training data
    data = load_dataset(file_name=core.TRAINING_DATA_FILE)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(["Id", core.TARGET], axis=1),
        data[core.TARGET],
        test_size=core.TEST_SIZE,
        random_state=models_params.RANDOM_STATE,
    )
    y_train = logarithm_transformer(data=y_train)
    y_test = logarithm_transformer(data=y_test)

    # Training Lasso
    best_rmse = float("inf")
    best_model = None

    for alpha, max_iter in product(
        models_params.LASSO_PARAMS["alpha"], models_params.LASSO_PARAMS["max_iter"]
    ):
        model_pipeline = create_lasso_pipeline(alpha=alpha, max_iter=max_iter)
        model_pipeline.fit(X_train, y_train)
        predictions = model_pipeline.predict(X_test)
        rmse = mean_squared_error(y_test, predictions) ** 0.5

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model_pipeline

    # Training XGBoost
    for n_estimators, learning_rate, max_depth in product(
        models_params.XGB_PARAMS["n_estimators"],
        models_params.XGB_PARAMS["learning_rate"],
        models_params.XGB_PARAMS["max_depth"],
    ):

        xgb_pipeline = create_xgb_pipeline(n_estimators, learning_rate, max_depth)
        xgb_pipeline.fit(X_train, y_train)
        predictions = xgb_pipeline.predict(X_test)
        rmse = mean_squared_error(y_test, predictions) ** 0.5

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = xgb_pipeline

    # Training MLP
    for hidden_layer, activation in product(
        models_params.MLP_PARAMS["hidden_layer_sizes"],
        models_params.MLP_PARAMS["activation"],
    ):

        mlp_pipeline = create_mlp_pipeline(
            hidden_layer=hidden_layer, activation=activation
        )
        mlp_pipeline.fit(X_train, y_train)
        predictions = mlp_pipeline.predict(X_test)
        rmse = mean_squared_error(y_test, predictions) ** 0.5

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = mlp_pipeline

    print(f"The best model is: {best_model}")
    print(f"the best rmse is: {best_rmse}")
    # Sauvegarder le modèle optimisé
    save_pipeline(pipeline_to_persist=best_model)


if __name__ == "__main__":
    run_training()
