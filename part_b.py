# Naddav Gover 308216340
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from part_a import SvmWithSmo


def import_data(path='iris.csv'):
    """Imports the data from a csv file.
    returns data of shape (150, 5) where the first column is id number and the rest 4 columns are features"""
    data = np.genfromtxt(path, delimiter=",")
    data = data[1:, : -1]  # get rid of titles and species name
    return data


def get_label(index):
    """Get the label of a current index.
    scatter_plot helper function"""
    if index == 0:
        return "Sepal Length"
    if index == 1:
        return "Sepal Width"
    if index == 2:
        return "Petal Length"
    if index == 3:
        return "Petal Width"

def scatter_plot(data, show=False):

    data = data[:, 1:]  # Get rid of id
    colors = ["red"] * 50 + ["green"] * 50 + ["blue"] * 50  # the data is ordered so this is ok
    fig, axs = plt.subplots(4, 4)  # we have 4 features, so the scatter matrix is 4x4

    # legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Iris-setosa', markerfacecolor='r', markersize=5),
                       Line2D([0], [0], marker='o', color='w', label='Iris-versicolor', markerfacecolor='g', markersize=5),
                       Line2D([0], [0], marker='o', color='w', label='Iris-virginica', markerfacecolor='b', markersize=5),
                      ]
    fig.legend(handles=legend_elements, loc='upper left')

    # title
    plt.suptitle("Scatter Plot Matrix")

    # filling each cell in the matrix
    for i in range(4):
        for j in range(4):
            if i == j:
                plt.delaxes(axs[i, j])
                continue
            ax = axs[i, j]  # get the current axis
            # hide tick and tick label of the big axes
            ax.tick_params(direction="in", labelcolor='none', top=False, bottom=False, left=False, right=False, pad=0.0)
            x = data[:, i].reshape((150, ))  # plot only the relevant feature
            y = data[:, j].reshape((150, ))  # plot only the relevant feature
            ax.scatter(x, y, s=1, color=colors)
            ax.set_xlabel(get_label(i), labelpad=-8, fontsize= 10)  # xlabel
            ax.set_ylabel(get_label(j), labelpad=-5, fontsize=10)  # ylabel
    if show:
        plt.show()


if __name__ == '__main__':

    # Get the iris data
    iris = import_data()

    # plot the scatter plot for section a in the programming assignment
    scatter_plot(iris)

    # sections c, d, e in the programming assignment
    kernels = ["linear", "rbf"]
    for kernel in kernels:
        print("Evaluating kernel: {}".format(kernel.upper()))
        # instantiate a class of the SVM with SMO algorithm with the iris data set
        # kernel is set to either "rbf" or "linear"
        svm = SvmWithSmo(iris, kernel=kernel)

        # Train the model and get predictions from all classes
        # Also get test results which is true pos, true neg, false pos, false neg
        # in the iris data set we have 3 classes
        # the number of classes is hard coded in this API
        predictions, test_results = svm.train_and_predict_all_classes()

        # Get the confusion matrix
        confusion_matrix = svm.test_all_classes(predictions)

        # plot confusion matrix and confusion table for each class
        svm.plot_confusion_matrix_and_tables(confusion_matrix=confusion_matrix, test_results=test_results, show_plots=False)

        print("Finished evaluating kernel: {}".format(kernel.upper()))

    # show the plots
    plt.show()