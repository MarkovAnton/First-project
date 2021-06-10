import shap
from matplotlib import pyplot as plt


def shap_val(gb_model, train_X):
    shap_values = shap.TreeExplainer(gb_model).shap_values(train_X)
    print(shap_values.shape)
    shap.summary_plot(shap_values, train_X)
    plt.show()

