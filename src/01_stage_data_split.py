import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os
import yaml
import numpy as np

def data_split(param_yaml_path):
    with open(param_yaml_path) as yaml_file:
        param_yaml = yaml.safe_load(yaml_file)
    local_data = param_yaml['data_source']["local_path"]

    data = pd.read_csv(local_data)
    data['quality'] = np.where(data['quality']>6.5,1,0)
    random_state = param_yaml["base"]["random_state"]
    split_ratio = param_yaml['split']["split_ratio"]

    train, test = train_test_split(data,test_size=split_ratio,random_state=random_state,)

    os.makedirs(param_yaml['split']['dir'],exist_ok= True)

    train_data_path = os.path.join(param_yaml['split']['dir'], param_yaml['split']['train_file'])
    train.to_csv(train_data_path,index= False)

    test_data_path = os.path.join(param_yaml['split']['dir'], param_yaml['split']['test_file'])
    test.to_csv(test_data_path,index= False)

if __name__ == "__main__":
    path = 'F:\DVC\DVC_Project_1\params.yaml'
    data_split(path)