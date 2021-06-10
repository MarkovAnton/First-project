import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def his_plot(train):
    pl = pd.DataFrame()
    pl['sales'] = train['sales']
    sns.histplot(data=pl, x='sales', bins=40)
    plt.title('Sales bar chart')
    plt.show()

