import pandas as pd


def read_train(path_train):
    train = pd.read_csv(path_train)
    return train


def read_test(path_test):
    test = pd.read_csv(path_test)
    return test
