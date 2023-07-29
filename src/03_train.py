import os,sys
import pickle
import pandas as pd
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier

def training(param_yaml_path):
    with open(param_yaml_path) as file:
        param_yaml = yaml.safe_load(file)

    final_train_data_path = os.path.join(param_yaml['process']['dir'],param_yaml['process']['train_file'])
    final_test_data_path = os.path.join(param_yaml['process']['dir'],param_yaml['process']['test_file'])

    random_state = param_yaml['base']['random_state']

    target = [param_yaml['base']['target_col']]

    train = pd.read_csv(final_train_data_path)
    test = pd.read_csv(final_test_data_path)

    y_train = train[target]
    y_test = test[target]

    x_train = train.drop(target,axis=1)
    x_test = train.drop(target,axis=1)

    random_state = param_yaml['base']['random_state']
    n_est = param_yaml['train']['n_set']

    rfc = RandomForestClassifier(random_state=random_state, n_estimators=n_est)
    rfc.fit(x_train,y_train.values.ravel())

    model_dir = param_yaml['model_dir']
    os.makedirs(model_dir,exist_ok= True)
    model_name = 'model.pkl'
    model_path = os.path.join(model_dir,model_name) 
    with open(model_path,'wb') as f:
        pickle.dump(rfc, f)

if __name__ == "__main__":
    training(param_yaml_path='F:\DVC\DVC_Project_1\params.yaml') 

