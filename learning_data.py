from sklearn.ensemble import GradientBoostingRegressor


def learn_data(train_X, train_y):
    GBoost = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.4,
                                       max_depth=7, max_features='sqrt',
                                       min_samples_leaf=7, min_samples_split=8,
                                       loss='huber', random_state=2021)
    gb_model = GBoost.fit(train_X, train_y)
    return GBoost, gb_model
