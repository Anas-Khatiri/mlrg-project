from configs import core
from intern_src.pipelines_package.predict import make_prediction
from intern_src.utils_package.data_ingestion import load_dataset


def sale_price_predict():
    test_data = load_dataset(file_name=core.TEST_DATA_FILE)
    result = make_prediction(input_data=test_data)
    predictions = result.get("predictions")
    print(f"House Price predictions: ")
    print(predictions)


if __name__ == "__main__":
    sale_price_predict()
