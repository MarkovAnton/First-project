from sklearn.model_selection import train_test_split


def split_data(train):
    train_X, test_X, train_y, test_y = train_test_split(
        train.drop(columns=['sales']),
        train['sales'],
        test_size=0.2,
        random_state=48)
    return train_X, test_X, train_y, test_y
