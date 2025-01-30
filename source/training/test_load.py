from train_utils import ConfigLoader, DataLoader, ModelLoader, ModelTrainer
import numpy as np
import pytest

# Fixtures for data and model setup
# The core ideas of fixture:
# 1. Fixture making sure to set up the data and "freeze it". Thus inject the data to every test case
# 2. However, since ML testing must be interdependent (making sure the flow logic is correctly done)
# The data has to be set up once for ALL OVER the test
# Thus, we need "scope" parameters in the decorators
# 3. If the scope is being deleted, for each every tests, pytest will load the data. 
# This will make the test operation heavy. Unless we want to test with fresh data for every test cases.

@pytest.fixture(scope='module')
def config_loader():
    config_file = 'C:/Users/Rafli/decision-pursue-mba/config.yaml'
    return ConfigLoader(config_file=config_file)

@pytest.fixture(scope='module')
def model_loader(config_loader):
    return ModelLoader(config_loader=config_loader)

@pytest.fixture(scope='module')
def data_loader(config_loader):
    data_file = 'C:/Users/Rafli/decision-pursue-mba/source/training/data/mba_decision_dataset.csv'
    return DataLoader(file_path=data_file, config_loader=config_loader)

@pytest.fixture(scope='module')
def dataset(data_loader):
    return data_loader.load_data()

@pytest.fixture(scope='module')
def X(data_loader, dataset):
    return data_loader.load_feature(dataset=dataset)

@pytest.fixture(scope='module')
def y(data_loader, dataset):
    return data_loader.load_target(dataset=dataset)

@pytest.fixture(scope='module')
def X_train(X, y, data_loader):
    X_train, X_test, y_train, y_test = data_loader.train_split(X=X, y=y)
    return X_train

@pytest.fixture(scope='module')
def X_test(X, y, data_loader):
    X_train, X_test, y_train, y_test = data_loader.train_split(X=X, y=y)
    return X_test

@pytest.fixture(scope='module')
def y_train(X, y, data_loader):
    X_train, X_test, y_train, y_test = data_loader.train_split(X=X, y=y)
    return y_train

@pytest.fixture(scope='module')
def y_test(X, y, data_loader):
    X_train, X_test, y_train, y_test = data_loader.train_split(X=X, y=y)
    return y_test

@pytest.fixture(scope='module')
def model(model_loader):
    return model_loader.load_model()

# Define the test cases
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