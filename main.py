import numpy as np
from read_files import read_train, read_test
from hisplot_for_data import his_plot
from check_null import check
from code_features import code_data
from correlations import corr_data
from splitting_data import split_data
from learning_data import learn_data
from sorted_features import sort_features
from shap_value import shap_val
from write_to_file import write_to_csv
from plot_predict import plot_pred
from distribution_of_residuals import distrib_of_residual


def main():
    train = read_train('D://Проект/Прогноз продаж/train.csv')
    test = read_test('D://Проект/Прогноз продаж/test.csv')
    print(train)
    print(test)
    his_plot(train)
    check(train, test)
    features = ['date', 'store', 'item']
    code_data(train, test, features)
    print(corr_data(train))
    train_X = split_data(train)[0]
    test_X = split_data(train)[1]
    train_y = split_data(train)[2]
    test_y = split_data(train)[3]
    GBoost = learn_data(train_X, train_y)[0]
    gb_model = learn_data(train_X, train_y)[1]
    print(GBoost.score(test_X, test_y))
    final_pred = GBoost.predict(test_X)
    plot_pred(final_pred, test_y)
    residuals = final_pred - test_y
    distrib_of_residual(residuals)
    sort_features(train_X, GBoost)
    train_X = train_X.loc[np.random.choice(train_X.index, 10000, replace=False)]
    print(train_X)
    shap_val(gb_model, train_X)
    write_to_csv(test, GBoost)


if __name__ == '__main__':
    main()

