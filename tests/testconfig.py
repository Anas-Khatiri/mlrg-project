import pytest

from configs import core
from intern_src.utils_package.data_ingestion import load_dataset


@pytest.fixture()
def sample_input_data():
    data = load_dataset(file_name=core.TEST_DATA_FILE)
    return data


@pytest.fixture()
def sample_continuous_log_data(sample_input_data):
    return sample_input_data[core.CONTINUOUS_VARS_LOG]
