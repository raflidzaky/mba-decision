# Def class to pull config from YAML
import yaml
import pandas as pd
import os
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Create config loader from config yaml
class ConfigLoader:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self):
        """Load the configuration from the YAML file."""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Configuration file {self.config_file} not found.")
        
        with open(self.config_file, 'r') as file:
            return yaml.safe_load(file)

    def get(self, key, default=None):
        """Get the configuration value for a given key, with an optional default."""
        return self.config.get(key, default)

# Def the feature
class FeatureConfig(BaseModel):
    entrepreneurial_interest: float
    networking_importance: float

# Def the data loader
class DataLoader:
    def __init__(self, file_path, config_loader):
        self.file_path = file_path
        self.config_loader = config_loader
    
    def load_data(self):
        dataset = pd.read_csv(self.file_path)
        return dataset
    
    def load_target(self, dataset):
        target = self.config_loader.get('target', {})
        target_column = list(target.values())
        data_Y = np.array(dataset[target_column]).reshape(-1, 1)
        return data_Y

    def load_feature(self, dataset):
        # Parse the feature needed
        features = self.config_loader.get('feature', {})
        
        # Filter the dataset to only contain feature columns
        feature_columns = list(features.values()) 

        # And pass it here
        data_X = np.array(dataset[feature_columns])
        return data_X
    
    def train_split(self, X, y):
        test_size = self.config_loader.get('data_split', {}).get('test_size')  # Default to 0.1 if not found
        random_state = self.config_loader.get('data_split', {}).get('random_state') 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                            random_state=random_state)
        return X_train, X_test, y_train, y_test

# Def the model selection (model name and params)
class ModelLoader:
    def __init__(self, config_loader):
        self.config_loader = config_loader
    
    def load_model(self):
        # Load model type and parameters from the YAML config
        model_config = self.config_loader.get('model', {})
        model_type = model_config.get('type')  # Default to DecisionTreeClassifier if not found
        params = model_config.get('params', {})

        # Instantiate the model based on the model type
        if model_type == "DecisionTreeClassifier":
            model = DecisionTreeClassifier(**params)
        else:
            raise ValueError(f"Model type '{model_type}' is not supported.")
        return model
    
# Def the model training
class ModelTrainer:
    def __init__(self, model_loader):
        self.config_loader = model_loader
    
    def train_model(self, X_train, y_train):
        model = self.model_loader.load_model()
        model.fit(X_train, y_train)
        return model