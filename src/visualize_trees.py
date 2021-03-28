import os
import sys
from sklearn.tree import export_graphviz
import graphviz
import six
import pydot
from sklearn import tree
from joblib import dump, load
import pandas as pd
import argparse
from subprocess import check_call
from itertools import compress

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
sys.path.append('C:/Program Files (x86)/Graphviz2.38/bin/')

#TODO Save path is not yet working
# save_path = '../trees/HEA_alloys/Calphad_Gen1'

model_path = '../model_checkpoints/model_checkpoint_Calphad_gs_SSOL6_model_1_2019-11-08-11-23.joblib'

estimator = load(model_path)

dotfile = six.StringIO()


def visualize(args: argparse.Namespace, i_tree=0) -> None:

    input_ = args.input
    df_orig = pd.read_excel(input_)
    col = list(df_orig.columns)[1:-1]

    ## Initial feature elimination if you have a predetermined mask
    transform_first = True
    if transform_first == True:

        transform_mask_init = pd.read_csv('../transform_mask/RFECV_transform_SSOL6_HEC_Multi-cation2019-11-08-11-15.csv')
        truth_series = pd.Series(transform_mask_init['0'], name='bools')
        df_orig.drop(['Name', 'EFA'], axis=1, inplace=True)
        df_orig = pd.DataFrame(df_orig.iloc[:, truth_series.values])

        col = list(df_orig.columns)[:]


    dotfile = six.StringIO()
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    sys.path.append('C:/Program Files (x86)/Graphviz2.38/bin/')

    for tree_in_forest in estimator:
        export_graphviz(tree_in_forest, out_file='tree.dot',
                        feature_names=col,
                        filled=True,
                        rounded=True)
        (graph,) = pydot.graph_from_dot_file('tree.dot')
        name = 'tree' + str(i_tree)
        print('Now exporting: ' + name)
        check_call(['dot', '-Tpng', 'tree.dot', '-o', name + '.png'])

        i_tree += 1


def parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('input', type=str, help='path/to/data/file.xlsx')

    return parser.parse_args()


def main():
    args = parser()
    visualize(args)


if __name__ == '__main__':
    main()

    # for tree_in_forest in estimator:
    #     dot_data = export_graphviz(tree_in_forest, out_file='tree.dot',
    #                         feature_names=col,
    #                         filled=True,
    #                         rounded=True)
    #     graph = graphviz.Source(dot_data)
    #     graph.render("name")
