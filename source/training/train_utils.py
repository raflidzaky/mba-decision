# Def class to pull config from YAML
import yaml
import pandas as pd
import os
from pydantic import BaseModel

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
    
    def load_feature(self, dataset):
        # Parse the feature needed
        features = self.config_loader.get('feature', {})
        
        # Filter the dataset to only contain feature columns
        feature_columns = list(features.values()) 

        # And pass it here
        data_fixture = dataset[feature_columns]
        return data_fixture
    
# Def the splitting

# Def the model selection (model name and params)

# Def the model training