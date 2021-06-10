import matplotlib.pyplot as plt


def distrib_of_residual(residuals):
    plt.hist(residuals, color='red', bins=20, edgecolor='black')
    plt.xlabel('Error')
    plt.ylabel('Count')
    plt.title('Distribution of Residuals')
    plt.show()

