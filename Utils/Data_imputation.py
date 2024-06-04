import pandas as pd
import miceforest as mf
from miceforest import mean_match_shap, mean_match_default
import numpy as np
from datetime import datetime
from pathlib import Path
import os
import yaml
from loguru import logger



def impute_data(train, test, config, save=True):
    dataset_path = f'../Datasets/Imputed/'
    now = datetime.now() # current date and time
    exist = os.path.exists(f"{dataset_path}/{config['directory_name']}")
    folder_name = f"{dataset_path}/{now.strftime('%Y-%m-%d_%H-%M-%S')}/" if exist \
                    else f"{dataset_path}/{config['directory_name']}/"
    if save:
        os.makedirs(folder_name)
        config_path = f"{folder_name}setup.yaml"
        
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f)

    

    scheme_quality = mean_match_shap.copy()
    scheme_quality.set_mean_match_candidates(config['set_mean_match_candidates'])
    scheme_defaul = mean_match_default.copy()
    scheme_defaul.set_mean_match_candidates(config['set_mean_match_candidates'])

    kernel = mf.ImputationKernel(
        train,
        save_all_iterations=config['save_all_iterations'],
        random_state=np.random.RandomState(config['random_state']),
        datasets=config['num_datasets'],
        mean_match_scheme = scheme_quality if config['quality'] else scheme_defaul,
        save_models=1
    )

    kernel.mice(iterations=config['num_iterations'],
        verbose=True,
        device=config['device'],
        num_threads = config['num_threads'],
    )
    
    if save:
        filename = folder_name + 'kernel.pkl'
        kernel.save_kernel(filename)

    #Geh the train and test datasets
    kernel.compile_candidate_preds()
    test_imputed = kernel.impute_new_data(test, verbose=True)

    train_list = []
    test_list = []
    for i in range(config['num_datasets']):
        train = kernel.complete_data(i)
        test = test_imputed.complete_data(i)
        train_list.append(train)
        test_list.append(test)

    return train_list, test_list, folder_name


def store_csv(test_list, train_list, folder_name, config):

    for i in range(config['num_datasets']):
        logger.info(f"Storing dataset {i}")
        subset_name = f"{folder_name}dataset_{i}/"
        os.mkdir(subset_name)  
        train = train_list[i]
        test = test_list[i]
        
        train_path = f"{subset_name}train.csv"
        test_path = f"{subset_name}test.csv"
        
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
