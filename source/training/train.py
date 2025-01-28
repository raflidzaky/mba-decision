import os
import pandas as pd
from train_utils import ConfigLoader, DataLoader, ModelLoader, ModelTrainer

def run_train_pipeline(config_file, data_file):
    # Prepare the configuration and data configuration
    config_loader = ConfigLoader(config_file=config_file)
    data_loader   = DataLoader(file_path=data_file, 
                               config_loader=config_loader)
    
    # Load the dataset, X, and y
    dataset = data_loader.load_data()
    X = data_loader.load_feature(dataset=dataset)
    y = data_loader.load_target(dataset=dataset)

    X_train, X_test, y_train, y_test = data_loader.train_split(X=X, 
                                                               y=y)
    
    model_loader  = ModelLoader(config_loader=config_loader)
    model_trainer = ModelTrainer(model_loader=model_loader)

    model = model_trainer.train_model(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f'Model Accuracy: {accuracy:.4f}')