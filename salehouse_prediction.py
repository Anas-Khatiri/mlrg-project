from intern_src.utils_package.data_ingestion import load_dataset
from configs import core
from intern_src.pipelines_package.predict import make_prediction


def sale_price_predict():
    # save_file = 'result_predictions.csv'
    test_data = load_dataset(file_name=core.TEST_DATA_FILE)
    result = make_prediction(input_data=test_data)
    predictions = result.get("predictions")
    print(f"House Price predictions: ")
    print(predictions)
    # predictions_df = pd.DataFrame(predictions)
    # predictions_df.to_csv(save_file)


if __name__ == "__main__":
    sale_price_predict()
