import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import os

def data_processing(data_path):
    data = pd.read_csv(data_path)

    preprocessed_data = data.dropna()

    return preprocessed_data
def apply_preprocessing(param_file_path):
    with open(param_file_path) as file:
        parma_yaml = yaml.safe_load(file)
    
    train_data_path = os.path.join(parma_yaml['split']['dir'],parma_yaml['split']['train_file'])
    final_train_data = data_processing(train_data_path)

    test_data_path = os.path.join(parma_yaml['split']['dir'],parma_yaml['split']['test_file'])
    final_test_data = data_processing(test_data_path)

    os.makedirs(parma_yaml['process']['dir'],exist_ok= True)
    processed_train_data_path = os.path.join(parma_yaml['process']['dir'],parma_yaml['process']['train_file'])
    processed_test_data_path = os.path.join(parma_yaml['process']['dir'],parma_yaml['process']['test_file'])
    final_train_data.to_csv(processed_train_data_path,index=False)
    final_test_data.to_csv(processed_test_data_path, index=False)
    


if __name__ =="__main__":
    path = 'F:\DVC\DVC_Project_1\params.yaml'
    apply_preprocessing(param_file_path=path)