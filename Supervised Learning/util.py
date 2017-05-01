import subprocess
from sklearn.tree import export_graphviz
import pandas as pd


# buying       v-high, high, med, low
# maint        v-high, high, med, low
# doors        2, 3, 4, 5-more
# persons      2, 4, more
# lug_boot     small, med, big
# safety       low, med, high
def visualize_tree(tree, feature_names, file_name):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    print "Writing tree to: " + file_name + ".png"
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", file_name + ".png"]
    try:
        subprocess.check_call(command)
        print "Tree visualization completed. File:" + file_name + ".png"
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")


def prepare_car_data(car_data_file):
    df = pd.read_csv(car_data_file, header=0)

    df = df.replace('vhigh', 4)
    df = df.replace('high', 3)
    df = df.replace('med', 2)
    df = df.replace('low', 1)
    df = df.replace('big', 3)
    df = df.replace('small', 1)
    df = df.replace('more', 5)
    df = df.replace('5more', 6)

    df = df.replace('unacc', 1)
    df = df.replace('acc', 2)
    df = df.replace('good', 3)
    df = df.replace('vgood', 4)
    features = list(df.columns[:6])

    x = df[features]
    y = df['class']

    return features, x, y
