from sklearn import model_selection, ensemble

__author__ = 'isaac waweru'

from sklearn.tree import DecisionTreeClassifier
from util import visualize_tree, prepare_car_data


def run(write_tree_to_file):
    features, x, y = prepare_car_data("car.data.txt")

    # Let's randomly split the data into training and testing
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3, random_state=42)

    dt = DecisionTreeClassifier(min_samples_split=250)

    # We train the decision tree using the training data
    dt.fit(x_train, y_train)

    # We get the accuracy of the decision tree by using the testing data
    print "Single Tree accuracy: {}".format(dt.score(x_test, y_test))

    # We can increase the accuracy of the decision tree by using adaboost (adaptive boosting)
    num_of_trees = 5
    clf = ensemble.AdaBoostClassifier(n_estimators=num_of_trees, base_estimator=dt)
    clf.fit(x_train, y_train)

    if write_tree_to_file:
        visualize_tree(dt, features, "tree_1")

    print "Ada boost with {} trees accuracy: {}".format(num_of_trees, clf.score(x_test, y_test))


if __name__ == "__main__":
    write_tree_to_file = False
    run(write_tree_to_file)
