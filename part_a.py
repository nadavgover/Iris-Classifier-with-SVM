#Nadav Gover 308216340

import numpy as np
import matplotlib.pyplot as plt
import sys, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class SvmWithSmo(object):

    def __init__(self, inputs, labels=None, c=None, kernel="rbf", gamma=None, tol=1e-3, epsilon=1e-3):
        """Instance initialization.
            Inputs:
                c: float or int, default is None, penalty parameter.
                    if None then 1.0 is used
                kernel: string, default is "rbf", kernel type,
                        select one kernel among ["linear", "rbf"].
                gamma: float, default is None, parameter for RBF
                          If None, 1 / n_features is used
                tol: float, default is 1e-3, the tolerance of inaccuracies
                     around the KKT conditions.
                epslion: float, default is 1e-3, tolerance for updating
                         Lagrange multiplier.
                """


        # Initialize parameters
        self.c = c
        self.kernel = kernel.lower()
        assert self.kernel == "rbf" or self.kernel == "linear"  # Making sure legal user input
        self.gamma = gamma
        self.tol = tol
        self.epsilon = epsilon

        self.inputs = self.prepare_input(inputs[:, 1:])  # shape (number of examples, number of features)
        self.labels = labels
        if self.labels is None:
            self.labels = np.zeros(inputs.shape[0])  # just zeros for now, it will be one vs all eventually
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_validation = None
        self.y_validation = None
        self.id_train = None
        self.id_test = None
        self.id_validation = None
        self.split_data(inputs, self.labels)  # splits the data and initializes the values above
        self.N, self.F = self.x_train.shape  # number of training examples and features
        if self.gamma is None:
            self.gamma = 1.0 / self.F
        if self.c is None:
            self.c = 1.0

        # Initialize all Lagrange multipliers as 0
        self.alphas = np.zeros(self.N)
        # Compute initial errors cache
        self.b = 0.0  # Threshold
        self.E = self.compute_error()


    def split_data(self, x, y, validation_set=True, test_percentage=0.2, validation_percentage=0.2):
        """Splits the data into train, test and validation data sets.
        Puts the values in the corresponding self fields"""
        if validation_set:
            x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=test_percentage + validation_percentage)
            test_percentage /= test_percentage + validation_percentage  # adjusting test percentage
            x_validation, x_test, y_validation, y_test = train_test_split(x_temp, y_temp, test_size=test_percentage)
            self.x_validation = x_validation[:, 1:]  # get rid of the ids
            self.y_validation = y_validation
            self.id_validation = x_validation[:, 0]  # get the ids

        else:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_percentage)

        self.x_train = x_train[:, 1:]  # get rid of the ids
        self.y_train = y_train
        self.x_test = x_test[:, 1:]  # get rid of the ids
        self.y_test = y_test
        self.id_train = x_train[:, 0]  # get the ids
        self.id_test = x_test[:, 0]  # get the ids

    def prepare_input(self, inputs):
        """Normalize inputs by the mean and std values of data set"""
        scaler = StandardScaler()
        scaled_inputs = scaler.fit_transform(inputs)
        # self.x_train = scaler.fit_transform(self.x_train)
        # self.x_test = scaler.fit_transform(self.x_test)
        # self.x_validation = scaler.fit_transform(self.x_validation)

        return scaled_inputs


    def objective_function(self, alphas=None):
        """Input:
                alphas : Lagrange multipliers, shape(n_train_samples, )
            Output:
                the objection value."""


        if alphas is None:
            alphas = self.alphas

        # Objective function as Equation 11 in [1]
        obj = 0.5 * np.sum(alphas ** 2 *
                           self.kernel_function(self.x_train, self.x_train) *
                           self.y_train ** 2) - np.sum(alphas)
        return obj

    def compute_error(self):
        """Compute error of all training samples.
            Output:
                the errors cache, shape (n_train_samples, )"""


        # Compute error according to Equation 10 and
        return self.decision(x=self.x_train) - self.y_train

    def decision(self, x=None):
        """Decision function to make prediction of given data.
            Input:
                x : features array of samples to be predicted, shape (n_samples, n_features)
                  if None, predict training samples.
            Output:
                the prediction of input data, shape (n_samples, )"""


        if x is None:
            x = self.x_train
        # Predict as Equation 10
        prediction = np.dot(self.alphas * self.y_train,
                            self.kernel_function(self.x_train, x)) - self.b

        return prediction

    def kernel_function(self, x1, x2):
        """Generates a kernel function, which is either linear or RBF.
            Inputs:
                x1, x2: features arrays, shape (n_samples, n_features)
                      x1 and x2 may have different number of samples, but the same number of features.
            Output:
                array, shape (n_x1_sample, n_x2_samples)"""

        if self.kernel == "linear":
            return np.dot(x1, x2.T)
        elif self.kernel == "rbf":
            denominator = 2 * (self.gamma ** 2)
            x1_ndim, x2_ndim = np.ndim(x1), np.ndim(x2)
            # Compute RBF kernel function when the input arrays have
            # different number of samples
            if x1_ndim == 1 and x2_ndim == 1:
                return np.exp(-np.linalg.norm(x1 - x2) / denominator)
            elif (x1_ndim > 1 and x2_ndim == 1) or \
                 (x1_ndim == 1 and x2_ndim > 1):
                return np.exp(-np.linalg.norm(x1 - x2, axis=1) / denominator)
            elif x1_ndim > 1 and x2_ndim > 1:
                return np.exp(-np.linalg.norm(
                    x1[:, np.newaxis] - x2[np.newaxis, :], axis=2) / denominator)

    def take_step(self, i1, i2):
        """take_step
            Inputs:
                i1, i2 : two indices of two chosen
                       Lagrange multipliers.
            Output:
                return 0 if failed to update
                return 1 if successfully update
        """

        if i1 == i2:
            return 0

        a1_old = self.alphas[i1]
        a2_old = self.alphas[i2]
        y1, y2 = self.y_train[i1], self.y_train[i2]
        x1, x2 = self.x_train[i1], self.x_train[i2]
        e1, e2 = self.E[i1], self.E[i2]
        s = y1 * y2

        # Compute L and H as Equation 13 and 14
        if y1 == y2:
            l = max(0, a2_old + a1_old - self.c)
            h = min(self.c, a2_old + a1_old)
        else:
            l = max(0, a2_old - a1_old)
            h = min(self.c, self.c + a2_old - a1_old)

        if l == h:
            return 0

        # Compute eta as Equation 15
        k11 = self.kernel_function(x1, x1)
        k12 = self.kernel_function(x1, x2)
        k22 = self.kernel_function(x2, x2)
        eta = k11 + k22 - 2 * k12

        if eta > 0:
            # Compute new alpha2 as Equation 16
            a2_new = a2_old + y2 * (e1 - e2) / eta
            # Clip alpha2 as Equation 17
            if a2_new > h:
                a2_new = h
            elif a2_new < l:
                a2_new = l
        else:
            # Update alpha2 as Equation 19 when
            # the eta is not positive
            alphas_temp = np.copy(self.alphas)
            alphas_temp[i2] = l
            # Objective function at a2=L
            l_obj = self.objective_function(alphas=alphas_temp)
            alphas_temp[i2] = h
            # Objective function at a2=H
            h_obj = self.objective_function(alphas=alphas_temp)

            if l_obj < (h_obj - self.epsilon):
                a2_new = l
            elif l_obj > (h_obj + self.epsilon):
                a2_new = h
            else:
                a2_new = a2_old

        # If the update of alpha 2 is smaller than
        # the tolerance, stop updating other variables
        if np.abs(a2_new - a2_old) < self.epsilon * (a2_new + a2_old + self.epsilon):
            return 0

        # Compute new alpha1 as Equation 18
        a1_new = a1_old + s * (a2_old - a2_new)

        # Update threshold as Equation 20, 21 and
        # the explanation under Equation 21
        b1_new = (self.b + e1 +
                  y1 * k11 * (a1_new - a1_old) +
                  y2 * k12 * (a2_new - a2_old))
        b2_new = (self.b + e2 +
                  y1 * k12 * (a1_new - a1_old) +
                  y2 * k22 * (a2_new - a2_old))

        if 0 < a1_new < self.c:
            self.b = b1_new
        elif 0 < a2_new < self.c:
            self.b = b2_new
        else:
            self.b = (b1_new + b2_new) / 2.0

        # Store new Lagrange multipliers
        self.alphas[i1] = a1_new
        self.alphas[i2] = a2_new

        # Update error cache using new alphas
        self.E = self.compute_error()

        return 1

    def examine_example(self, i2):
        """examine_example
            Input:
                i2 : int, the index of one chosen Lagrange multiplier.
            Output:
                return 0 if failed to update
                return 1 if successfully update
        """

        y2 = self.y_train[i2]
        a2 = self.alphas[i2]
        e2 = self.E[i2]
        r2 = e2 * y2

        # Check if the example satisfy the KKT condition
        if ((r2 < -self.tol and a2 < self.c) or
           (r2 > self.tol and a2 > 0)):
            # If not satisfy KKT condition
            # Indices of Lagrange multiplier which is not 0 and not C
            n0nc_list = np.where((self.alphas != 0) &
                                 (self.alphas != self.c))[0]
            if len(n0nc_list) > 1:
                if self.E[i2] > 0:
                    i1 = np.argmin(self.E)
                else:
                    i1 = np.argmax(self.E)
                if self.take_step(i1, i2):
                    return 1

            # Loop over all non-0 and non-C alpha,
            # starting at a random point
            rnd_n0nc_list = np.random.permutation(n0nc_list)
            for i1 in rnd_n0nc_list:
                if self.take_step(i1, i2):
                    return 1

            # Loop over all possible i1, starting at a random point
            rnd_all_list = np.random.permutation(self.N)
            for i1 in rnd_all_list:
                if self.take_step(i1, i2):
                    return 1
        return 0

    def train(self, class_number=None):
        """train
            Apply Sequential Minimal Optimization to train a SVM classifier.
            trains on the train data set
        """
        if class_number is not None:
            if class_number == 0:
                class_name = "Setosa"
            elif class_number == 1:
                class_name = "Versicolor"
            else:
                class_name = "Virginica"
        if class_number is not None:
            print("Training class {}".format(class_name))
        else:
            print("Training")

        # restoring values
        self.alphas = np.zeros(self.N)
        self.E = self.compute_error()

        # Main routine
        num_changed = 0
        examine_all = 1

        while num_changed > 0 or examine_all:
            num_changed = 0
            if examine_all:
                # Loop i2 over all training examples
                for i2 in range(self.N):
                    num_changed += self.examine_example(i2)
            else:
                # Loop i2 over examples where alpha
                # is not 0 and not C
                i2_list = np.where((self.alphas != 0) &
                                   (self.alphas != self.c))[0]
                for i2 in i2_list:
                    num_changed += self.examine_example(i2)

            if examine_all == 1:
                examine_all = 0
            elif num_changed == 0:
                examine_all = 1

    def predict(self, x_test, sign=True):
        """predict
            Return the prediction of given data.
            Inputs:
                x_test : features array of test set, shape (n_test_sample, n_features)
                sign : boolean, default is True. If True, return the classification result
                       else, return the original prediction.
            Output:
                prediction : original prediction if sign is False
                             else, return the binary classification result.
        """

        # Original prediction
        prediction = self.decision(x=x_test)

        # Classification result
        if sign:
            prediction = np.sign(prediction)
        return prediction

    def find_optimal_c(self, class_number=None):
        """Returns the optimal c to be used. Using a grid search method"""
        temp_x_train = self.x_train  # value to be restored after finding c
        temp_y_train = self.y_train  # value to be restored after finding c
        self.x_train = self.x_validation
        self.y_train = self.y_validation
        self.N, self.F = self.x_train.shape  # number of training examples and features
        self.alphas = np.zeros(self.N)
        self.E = self.compute_error()
        best_acc = 0
        best_c_region = 0

        class_name = None
        if class_number is not None:
            if class_number == 0:
                class_name = "Setosa"
            elif class_number == 1:
                class_name = "Versicolor"
            else:
                class_name = "Virginica"
        if class_number is not None:
            print("Finding optimal c for class {}".format(class_name))
        else:
            print("Finding optimal c")
        block_print()  # blocking print of train function

        # Coarse grid search
        for i in range(-5, 8, 2):
            self.c = 2 ** i
            self.alphas = np.zeros(self.N)  # clean slate
            self.E = self.compute_error()
            self.train()

            prediction = self.predict(x_test=self.x_test)
            accuracy = self.calculate_accuracy(prediction, self.y_test)
            if accuracy > best_acc:
                best_acc = accuracy
                best_c_region = i

        best_acc = 0
        best_c = 0
        # Fine grid search
        for i in np.arange(best_c_region - 1, best_c_region + 1, 0.2):
            self.c = 2 ** i
            self.alphas = np.zeros(self.N)
            self.E = self.compute_error()
            self.train()
            prediction = self.predict(x_test=self.x_test)
            accuracy = self.calculate_accuracy(prediction, self.y_test)
            if accuracy > best_acc:
                best_acc = accuracy
                best_c = self.c

        # Restore the values
        self.x_train = temp_x_train
        self.y_train = temp_y_train
        self.N, self.F = self.x_train.shape  # number of training examples and features
        self.alphas = np.zeros(self.N)
        self.E = self.compute_error()
        enable_print()  # enabling print again
        print("Optimal c is: {:0.3f}".format(best_c))

        return best_c

    def find_optimal_gamma(self, class_number=None):
        """Returns the optimal gamma to be used. Using a grid search method"""

        if self.kernel != "rbf":  # only relevant if this is rbf kernel
            return self.gamma

        temp_x_train = self.x_train  # value to be restored after finding gamma
        temp_y_train = self.y_train  # value to be restored after finding gamma
        self.x_train = self.x_validation
        self.y_train = self.y_validation
        self.N, self.F = self.x_train.shape
        self.alphas = np.zeros(self.N)
        self.E = self.compute_error()
        best_acc = 0
        best_gamma_region = 0

        class_name = None
        if class_number is not None:
            if class_number == 0:
                class_name = "Setosa"
            elif class_number == 1:
                class_name = "Versicolor"
            else:
                class_name = "Virginica"
        if class_number is not None:
            print("Finding optimal gamma for class {}".format(class_name))
        else:
            print("Finding optimal gamma")
        block_print()  # blocking print of train function

        # Coarse grid search
        for i in range(-15, 3, 2):
            self.gamma = 2 ** i
            self.alphas = np.zeros(self.N)  # clean slate
            self.E = self.compute_error()
            self.train()  # train
            pred = self.predict(x_test=self.x_test)
            acc = self.calculate_accuracy(pred, self.y_test)
            if acc > best_acc:
                best_acc = acc
                best_gamma_region = i

        # Fine grid search
        best_acc = 0
        best_gamma = 0

        for i in np.arange(best_gamma_region - 1, best_gamma_region + 1, 0.1):
            self.gamma = 2 ** i
            self.alphas = np.zeros(self.N)
            self.E = self.compute_error()
            self.train()
            pred = self.predict(x_test=self.x_test)
            acc = self.calculate_accuracy(pred, self.y_test)
            if acc > best_acc:
                best_acc = acc
                best_gamma = self.gamma

        # Restore the values
        self.x_train = temp_x_train
        self.y_train = temp_y_train
        self.N, self.F = self.x_train.shape  # number of training examples and features
        self.alphas = np.zeros(self.N)
        self.E = self.compute_error()
        enable_print()  # enabling print again
        print("Optimal gamma is: {:0.3f}".format(best_gamma))

        return best_gamma

    def calculate_accuracy(self, y_pred, y_true):
        """calculate accuracy of prediction.
            Inputs:
                y_pred : Labels predicted by the model, shape (n_samples, )
                y_true : Real labels of data set, shape (n_samples, )
            Output:
                float, the accuracy of prediction.
        """

        return 100 * np.mean((y_pred == y_true) * 1.0)

    def calculate_accuracy_all_classes(self, confusion_matrix):
        """Calculates the accuracy of the classifier"""
        return np.trace(confusion_matrix)/np.sum(confusion_matrix)

    def train_and_predict_all_classes(self):
        """Train all 3 classes of data set
        Returns:
            predictions of all classes shape (3, test set size)
            test results which is true pos, true neg, false pos, false neg, shape (3, 4)"""
        predictions = None
        test_results = None
        for i in range(3):
            # Creating one vs all labels
            self.one_vs_all_labels(i)
            # labels = np.ones(150) * -1
            # labels[i: i+50] = 1
            # self.labels = labels
            # self.split_data(self.inputs, self.labels)
            c = self.find_optimal_c(class_number=i)
            self.c = c
            gamma = self.find_optimal_gamma(class_number=i)
            self.gamma = gamma
            self.train(class_number=i)
            prediction = self.predict(x_test=self.x_test, sign=False)
            # accuracy = self.calculate_accuracy(prediction, svm.y_test)
            test_result = self.test(prediction)
            if i == 0:
                predictions = prediction
                test_results = test_result
            else:
                predictions = np.vstack((predictions, prediction))
                test_results = np.vstack((test_results, test_result))

        return predictions, test_results

    def test(self, predicitions):
        """Tests one class
        Returns true pos, true neg, false pos, false neg"""
        predictions = np.sign(predicitions)  # making the predictions -1, 1

        true_pos = np.sum((predictions == self.y_test)[predictions == 1])
        true_neg = np.sum((predictions == self.y_test)[predictions == -1])
        false_pos = np.sum((predictions != self.y_test)[predictions == 1])
        false_neg = np.sum((predictions != self.y_test)[predictions == -1])

        return np.array([true_pos, true_neg, false_pos, false_neg])

    def one_vs_all_labels(self, class_to_classify):
        """Creates one vs all labels
        class to classify is in [0,1,2]"""
        index = 0
        for id in self.id_train:
            if class_to_classify * 50 < id <= class_to_classify * 50 + 50:
                self.y_train[index] = 1
            else:
                self.y_train[index] = -1
            index += 1

        index = 0
        for id in self.id_test:
            if class_to_classify * 50 < id <= class_to_classify * 50 + 50:
                self.y_test[index] = 1
            else:
                self.y_test[index] = -1
            index += 1

        index = 0
        for id in self.id_validation:
            if class_to_classify * 50 < id <= class_to_classify * 50 + 50:
                self.y_validation[index] = 1
            else:
                self.y_validation[index] = -1
            index += 1

    def test_all_classes(self, predictions):
        """Tests all the classes, returns confusion matrix
        Input:
            predictions: shape (3, 30) each row is the predictions of one class"""

        predictions = np.argmax(predictions, axis=0)  # taking the best prediction of each class

        # initialize the confusion matrix to zeros
        # rows are predicted class and columns are actual class
        confusion_matrix = np.zeros((3, 3), dtype="int32")
        # true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
        for i in range(self.y_test.shape[0]):  # going through all test examples
            current_prediction = predictions[i]
            if self.id_test[i] <= 50:
                current_label = 0
            elif self.id_test[i] <= 100:
                current_label = 1
            else:
                current_label = 2
            confusion_matrix[current_prediction][current_label] += 1

        return confusion_matrix

    def plot_confusion_matrix(self, confusion_matrix, show_plot=False):
        """Plots the confusion table of the multi-class PLA.
        Parameters: confusion_matrix - size 10x10
                    Rows are predicted class and columns are actual class as returned from test_multiclass"""

        accuracy = self.calculate_accuracy_all_classes(confusion_matrix=confusion_matrix)

        axis_names = ["Setosa", "Versicolor", "Virginica"]

        plt.figure()
        ax = plt.gca()
        ax.matshow(confusion_matrix, cmap=plt.get_cmap('Blues'))

        # Putting the value of each cell in its place and surrounding it with a box so you could see the text better
        for (i, j), z in np.ndenumerate(confusion_matrix):
            ax.text(i, j, '{}'.format(z), ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

        # setting x and y axis to be the class numbers
        tick_marks = np.arange(len(axis_names))
        plt.xticks(tick_marks, axis_names)
        plt.yticks(tick_marks, axis_names)

        # setting title and labels for the axis
        plt.title("Confusion Matrix of the Multi-Class SVM with SMO {} Kernel\n".format(self.kernel.upper()))
        plt.ylabel('Predicted Class')
        plt.xlabel('Actual Class\naccuracy={:0.4f} %'.format(accuracy*100))
        if show_plot:
            plt.show()

    def calculate_sensitivity(self, true_pos, false_neg):
        """Calculate the sensitivity of a classifier"""
        return  true_pos/(true_pos+false_neg)

    def plot_confusion_table(self, true_pos, true_neg, false_pos, false_neg, number_to_classify, show_plot=False):
        """Plots the confusion table of a specific class.
        Where the class is the number_to_classify"""

        sensitivity = self.calculate_sensitivity(true_pos, false_neg)
        table = np.array([[true_pos, false_pos], [false_neg, true_neg]])

        if number_to_classify == 0:
            current_class = "Setosa"
        elif number_to_classify == 1:
            current_class = "Versicolor"
        else:
            current_class = "Virginica"
        axis_names = ["{}".format(current_class), "Not {}".format(current_class)]

        # plt.figure(number_to_classify)
        plt.figure()
        ax = plt.gca()
        ax.matshow(table, cmap=plt.get_cmap('Blues'))

        for (i, j), z in np.ndenumerate(table):
            if i == 0:
                if j == 0:
                    string = "True Positive"
                else:
                    string = "False Positive"
            else:
                if j == 0:
                    string = "False Negative"
                else:
                    string = "True Negative"

            ax.text(i, j, string + '\n{}'.format(z), ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

        tick_marks = np.arange(len(axis_names))
        plt.xticks(tick_marks, axis_names)
        plt.yticks(tick_marks, axis_names)
        plt.title("Confusion Table of class {}, {} Kernel\n".format(current_class, self.kernel.upper()))
        plt.ylabel('Actual Class')
        plt.xlabel('Predicted Class\nsensitivity={:0.4f}'.format(sensitivity))
        if show_plot:
            plt.show()

    def plot_confusion_table_of_all_classes(self, test_results, show_table=False):

        for i in range(3):
            true_pos, true_neg, false_pos, false_neg = test_results[i]
            self.plot_confusion_table(true_pos=true_pos, true_neg=true_neg, false_pos=false_pos,
                                      false_neg=false_neg, number_to_classify=i)
        if show_table:
            plt.show()

    def plot_confusion_matrix_and_tables(self, confusion_matrix, test_results, show_plots=False):
        """Plots the confusion matrix and the confusion table of each class
        Parameters: confusion_matrix - size 10x10 as returned from test_multiclass
                    test_results - and array of tp, tn, fp, fn as returned from test_all_classes_part_a"""

        self.plot_confusion_table_of_all_classes(test_results)
        self.plot_confusion_matrix(confusion_matrix)
        if show_plots:
            plt.show()



def block_print():
    """Disable printing"""
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    """Enable printing"""
    sys.stdout = sys.__stdout__


if __name__ == '__main__':
    """Usage Example"""

    from part_b import import_data

    # Get the iris data
    iris = import_data()

    # instantiate a class of the SVM with SMO algorithm with the iris data set
    # kernel is set to "rbf"
    # kernel can be either "rbf" or "linear"
    svm = SvmWithSmo(iris, kernel="rbf")

    # Train the model and get predictions from all classes
    # Also get test results which is true pos, true neg, false pos, false neg
    # in the iris data set we have 3 classes
    # the number of classes is hard coded in this API
    predictions, test_results = svm.train_and_predict_all_classes()

    # Get the confusion matrix
    confusion_matrix = svm.test_all_classes(predictions)

    # plot confusion matrix and confusion table for each class
    svm.plot_confusion_matrix_and_tables(confusion_matrix=confusion_matrix, test_results=test_results, show_plots=True)
