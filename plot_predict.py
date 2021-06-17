import matplotlib.pyplot as plt
import seaborn as sns


def plot_pred(final_pred, test_y):
    plt.rcParams['font.size'] = 24
    plt.style.use('fivethirtyeight')

    sns.kdeplot(final_pred, label='Predictions')
    sns.kdeplot(test_y, label='Values')

    plt.xlabel('Energy Star Score')
    plt.ylabel('Density')
    plt.title('Test Values and Predictions')
    plt.legend()
    plt.show()

