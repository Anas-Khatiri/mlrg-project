import typing as t

import numpy as np
import pandas as pd

from intern_src.utils_package.data_ingestion import load_pipeline
from intern_src.utils_package.data_prediction import validate_inputs
from configs import core

pipeline_file_name = f"{core.PIPELINE_SAVE_NAME}.pkl"
model_pipeline = load_pipeline(file_name=pipeline_file_name)


def make_prediction(
    *,
    input_data: t.Union[pd.DataFrame, dict],
) -> dict:
    """Make a prediction using a saved model pipeline."""

    data = pd.DataFrame(input_data)
    validated_data, errors = validate_inputs(input_data=data)
    results = {"predictions": None, "errors": errors}
    validated_data = validated_data.drop(["Id"], axis=1)
    if not errors:
        predictions = model_pipeline.predict(X=validated_data[validated_data.columns])
        results = {
            "predictions": [float(np.exp(pred)) for pred in predictions],
            "errors": errors,
        }

    return results
