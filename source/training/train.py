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
    # Load model configuration from YAML
    model_loader  = ModelLoader(config_loader=config_loader)

    # Load training configuration and train the model
    model_trainer = ModelTrainer(model_loader=model_loader)
    model = model_trainer.train_model(X_train, y_train)

    # Serialize the model
    model_trainer.save_model(model=model, model_path='trained_detree1.pkl')

    # Check the model
    accuracy = model.score(X_test, y_test)
    print(f'Model Accuracy: {accuracy:.4f}')

if __name__ == "__main__":
    config_file = 'C:/Users/Rafli/decision-pursue-mba/config.yaml'
    data_file = 'C:/Users/Rafli/decision-pursue-mba/source/training/data/mba_decision_dataset.csv' 
    
    # Run the pipeline
    trained_model = run_train_pipeline(config_file, data_file)