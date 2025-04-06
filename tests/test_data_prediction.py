import pandas as pd

from configs import core
from intern_src.utils_package.data_prediction import (drop_missing_inputs,
                                                      validate_inputs)
from tests.testconfig import sample_input_data


def test_drop_missing_inputs_removes_na(sample_input_data):
    # Injecter une valeur manquante dans une variable
    # non autorisée à contenir des NaN
    var_with_na = [
        col
        for col in sample_input_data.columns
        if col
        not in (
            core.CATEGORICAL_FREQUENT_MISSING
            + core.CATEGORICAL_FREQUENT_CATEGORY
            + core.NUMERICAL_MISSING_VARS
        )
    ][0]

    sample_input_data.loc[0, var_with_na] = None

    cleaned_data = drop_missing_inputs(input_data=sample_input_data)

    # On s'assure que la ligne avec NaN
    # sur une variable critique a été supprimée
    assert cleaned_data[var_with_na].isnull().sum() == 0
    assert len(cleaned_data) < len(sample_input_data)


def test_validate_inputs_returns_data_and_no_errors(sample_input_data):
    validated_data, errors = validate_inputs(input_data=sample_input_data)

    # On attend qu'aucune erreur de validation ne soit levée
    assert isinstance(validated_data, pd.DataFrame)
    assert errors is None
