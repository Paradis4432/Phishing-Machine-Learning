
import re


def make14x14CorrHeatmap(data, title, xlabel, ylabel, savepath, showAnnot):
    """
    Correlation heatmap for 14x14 matrix
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Set up the seaborn plot
    sns.set(style="white", palette="muted", color_codes=True)
    f, ax = plt.subplots(figsize=(14, 14))
    sns.heatmap(data.corr(), annot = showAnnot, ax=ax)

    # Add title and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Save the plot
    f.savefig(savepath)

def make7x7ConsusionMatrix(X, Y, title, xlabel, ylabel, savepath, showAnnot, returnMatrix):
    """
    Correlation heatmap for 7x7 matrix
    """
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    mat = confusion_matrix(Y, X)
    plt.subplots(figsize=(7, 7))
    ax = plt.axes()

    # Set up the seaborn plot
    sns.heatmap(mat.T, square=True, annot = showAnnot, fmt='d', cbar=False,
            xticklabels=['Negativo','Positivo'], yticklabels=['Negativo','Positivo'], ax=ax)

    # Add title and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Save the plot
    plt.savefig(savepath)

    if returnMatrix:
        return mat