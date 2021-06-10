def sort_features(train_X, GBoost):
    print(sorted(
        zip(train_X.columns, GBoost.feature_importances_),
        key=lambda p: p[1],
        reverse=True
    ))

