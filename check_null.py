def check(train, test):
    print(train.info())
    print(test.info())
    print(train.isna().mean())
    print(test.isna().mean())

