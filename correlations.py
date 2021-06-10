import seaborn as sns
import matplotlib.pyplot as plt


def corr_data(train):
    correlations_data = train.corr()['sales'].sort_values()
    sns.heatmap(train.corr())
    plt.title('Сorrelation for data')
    plt.show()
    return correlations_data

