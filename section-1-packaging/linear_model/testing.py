import numpy as np
from config.core import config
from pipeline import price_pipe
from processing.data_manager import load_dataset
from predict import make_prediction


results=make_prediction(input_data=load_dataset(file_name=config.app_config.test_data_file))
print(results)
#make_prediction(input_data=sample_input_data)
