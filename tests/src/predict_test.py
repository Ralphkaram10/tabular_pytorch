import pytest
import yaml
from src.predict import predict
import src.predict
from src.dataloader.dataloader import load_test_data
from src.common.loading import load_trained_model,load_pickle
from src.common.keywords import (
            MODEL_PATH_KEY,
            PREDICT_INPUT_DICT_KEY,
            X_KEY,
            Y_KEY,
            X_TEST_KEY,
            Y_TEST_KEY,
            BATCH_X_KEY,
            INPUT_SCALER_KEY,
            MODEL_KEY,
            TARGET_SCALER_KEY,
            BATCH_Y_PRED_KEY,
            TEST_LOADER_KEY
    )

@pytest.fixture
def predict_main():
    main_output_dict=src.predict.main()
    return main_output_dict

def test_predict_input_content(predict_main):
    predict_input_dict = predict_main[PREDICT_INPUT_DICT_KEY]
    assert set(predict_input_dict.keys())==set([BATCH_X_KEY,INPUT_SCALER_KEY,MODEL_KEY,TARGET_SCALER_KEY])

