import pandas as pd


def write_to_csv(test, GBoost):
    test = test.reindex(labels=test.columns, axis=1)
    df = pd.DataFrame()
    df['id'] = test.index
    df['sales'] = GBoost.predict(test)
    df.to_csv('D://Проект/Прогноз продаж/submission.csv', index=False)
