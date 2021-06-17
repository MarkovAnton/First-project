from sklearn.preprocessing import LabelEncoder


def code_data(train, test, features):
    for c in features:
        lbl = LabelEncoder()
        train[c] = lbl.fit_transform(train[c])
        test[c] = lbl.fit_transform(test[c])
    print(train[features].head(10))
    print(test[features].head(10))
