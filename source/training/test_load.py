from train_utils import ConfigLoader, DataLoader, ModelLoader, ModelTrainer
import numpy as np
import pytest

# TODO: Apply fixture decorators on each test case
# Since the data is not from a static data (rather a "dynamic" one like model loader)
# Fixture making sure to set up the data and "freeze it"
# Thus, inject the data to each test case

def test_X_type(X):
    # Test that X is a NumPy array
    assert isinstance(X, np.ndarray), f"Expected X to be a numpy.ndarray, but got {type(X)}"

def test_X_element_type(X):
    # Test that each element inside X is of type numpy.ndarray
    for i in X:
        for j in i:  # Iterate over each element in X (inner elements)
            assert isinstance(j, float), f"Expected element in X to be np.int64, but got {type(j)}"

def test_y_type(y):
    # Test that y is a NumPy array
    assert isinstance(y, np.ndarray), f"Expected y to be a numpy.ndarray, but got {type(y)}"
    
def test_split(X, X_train, X_test, y, y_train, y_test):
    # This func input is X/y train and test
    # The end goal is to make sure whether it really splits out the data AND
    # it splits accordingly (correct size)

    # Define test case for train dataset
    count_x_train = len(X_train)
    count_x       = len(X)
    total_x       = count_x_train / count_x

    count_y_train = len(y_train)
    count_y       = len(y)
    total_y       = count_y_train / count_y

    assert total_x == pytest.approx(0.9)
    assert total_y == pytest.approx(0.9)

    # Define test case for test dataset
    count_x_test = len(X_test)
    total_xs     = count_x_test / count_x

    count_y_test = len(y_test)
    total_ys     = count_y_test / count_y

    assert total_xs == pytest.approx(0.1)
    assert total_ys == pytest.approx(0.1)

def test_fitting(model, X_train, y_train):
    # Fit the model
    model.fit(X_train, y_train)

def test_prediction(model):
    dummy=[[0.9, 0.9]]
    prediction = model.predict(dummy)
    assert prediction == "Yes"

# Initialize config path
config_file = 'C:/Users/Rafli/decision-pursue-mba/config.yaml'
data_file = 'C:/Users/Rafli/decision-pursue-mba/source/training/data/mba_decision_dataset.csv' 

# Start to config data and model
config_loader = ConfigLoader(config_file=config_file)
model_loader  = ModelLoader(config_loader=config_loader)
data_loader   = DataLoader(file_path=data_file, 
                           config_loader=config_loader)

dataset = data_loader.load_data()
model   = model_loader.load_model()
X = data_loader.load_feature(dataset=dataset)
y = data_loader.load_target(dataset=dataset)

X_train, X_test, y_train, y_test = data_loader.train_split(X=X, 
                                                           y=y)
    
test_X_type(X=X)
test_X_element_type(X=X)
test_y_type(y=y)
test_split(X=X, X_train=X_train, X_test=X_test,
           y=y, y_train=y_train, y_test=y_test)
test_fitting(model=model, 
             X_train=X_train, 
             y_train=y_train)
test_prediction(model=model)