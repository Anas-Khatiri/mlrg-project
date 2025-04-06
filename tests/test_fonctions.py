import numpy as np
import pandas as pd
import pytest

from configs import core
from intern_src.utils_package.data_transformation import (
    TemporalTransformer, logarithm_transformer)
from tests.testconfig import sample_continuous_log_data, sample_input_data


def test_temporal_transformer_initialization():
    transformer = TemporalTransformer(
        variables=["var1", "var2"], reference_variable="ref"
    )
    assert transformer.variables == ["var1", "var2"]
    assert transformer.reference_variable == "ref"


def test_temporal_transformer_invalid_variables():
    with pytest.raises(ValueError, match="variables should be a list"):
        TemporalTransformer(variables="var1", reference_variable="ref")


def test_temporal_transformer_transform(sample_input_data):
    """
    Teste que la transformation temporelle est
    correctement appliquée en soustrayant
    chaque variable temporelle de la variable de référence.
    """
    # Assure que toutes les colonnes nécessaires sont là
    required_columns = core.TEMPORAL_VARS + [core.REF_VARS]
    for col in required_columns:
        assert (
            col in sample_input_data.columns
        ), f"{col} manquant dans les données d'entrée"

    transformer = TemporalTransformer(
        variables=core.TEMPORAL_VARS, reference_variable=core.REF_VARS
    )

    transformed = transformer.fit_transform(sample_input_data)

    # Vérifie la transformation pour chaque variable temporelle
    for var in core.TEMPORAL_VARS:
        expected = sample_input_data[core.REF_VARS] - sample_input_data[var]
        pd.testing.assert_series_equal(
            transformed[var], expected, check_names=False, check_dtype=False
        )


def test_logarithm_transformer(sample_continuous_log_data):
    """
    Teste que la transformation logarithmique
    est correctement appliquée
    sur les variables CONTINUOUS_VARS_LOG.
    """
    transformed = logarithm_transformer(sample_continuous_log_data)

    # Vérifie que chaque colonne a été transformée correctement
    for col in core.CONTINUOUS_VARS_LOG:
        expected = np.log(sample_continuous_log_data[col])
        pd.testing.assert_series_equal(
            transformed[col], expected, check_names=False, check_dtype=False
        )


def test_logarithm_transformer_negative_values():
    with pytest.raises(ValueError):
        logarithm_transformer(np.array([-1, 2, 3]))


"""""" """""" """""" """""" """""" """""" """""" """""" """""" """"
def test_mapper_initialization():
    mapper = Mapper(variables=['category_col'], mappings={'A': 1, 'B': 2})
    assert mapper.variables == ['category_col']
    assert mapper.mappings == {'A': 1, 'B': 2}


def test_mapper_transform(sample_categorical_data):
    mapper = Mapper(variables=['category_col'], mappings={'A': 1, 'B': 2})
    transformed = mapper.fit_transform(sample_categorical_data)

    assert transformed['category_col'].notnull().all(), "Des valeurs manquantes sont apparues après transformation."
    assert transformed[
               'category_col'].dtype == 'int64' or 'float64', "Le mapping ne convertit pas correctement les valeurs."


def test_mapper_transform_unknown_value(sample_categorical_data):
    mapper = Mapper(variables=['category_col'], mappings={'A': 1, 'B': 2})
    transformed = mapper.fit_transform(sample_categorical_data)

    assert transformed.isna().sum().sum() > 0, "Les valeurs inconnues ne sont pas transformées en NaN."
""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""


def test_feature_match(sample_input_data):
    """
    Vérifie que les features définies dans
    core.FEATURES correspondent exactement
    aux colonnes du dataset chargé.
    """
    expected_features = core.FEATURES  # Liste des features définies dans core
    actual_features = sample_input_data.columns.tolist()  # Colonnes du dataset

    # Vérifier que toutes les features attendues sont présentes
    missing_features = [
        feature for feature in expected_features
        if feature not in actual_features
    ]
    assert not missing_features, f"Colonnes manquantes : {missing_features}"

    # Vérifier qu'il n'y a pas de colonnes supplémentaires
    extra_features = [
        feature for feature in actual_features
        if feature not in expected_features
    ]
    assert not extra_features, \
        f"Colonnes en trop dans les données : {extra_features}"

    # Vérifier que l'ordre des colonnes est respecté
    assert (
        actual_features == expected_features
    ), (f"L'ordre des colonnes ne correspond pas "
        f": {actual_features} != {expected_features}")
