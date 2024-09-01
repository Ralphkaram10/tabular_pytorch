import pytest
import src.kpi
from src.common.keywords import(
        MSE_KEY,
        RMSE_KEY,
        R2_KEY
        )

@pytest.fixture
def kpi_main():
    evaluate_output_dict=src.kpi.main()
    return evaluate_output_dict

def test_main_output_content(kpi_main):
    assert set(kpi_main.keys())==set([MSE_KEY,RMSE_KEY,R2_KEY])
