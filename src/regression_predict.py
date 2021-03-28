import pandas as pd
import numpy as np
import argparse
import time
import os
from pathlib import Path
import sklearn
from joblib import dump, load

df_transform = True


def model_predict(args: argparse.Namespace) -> None:
    rf = load('../model_checkpoints/model_checkpoint_Calphad_gs_SSOL6_model_1_2019-11-08-11-23.joblib')
    # output_direc = '../HEA_alloys/wCALPHAD/Feature_set_2/'

    input_ = args.input

    transform_df = '../transform_mask/RFECV_transform_SSOL6_HEC_Multi-cation2019-11-08-11-15.csv'


    p = Path(input_)
    p = p.parent
    p = p.parent

    new_predictions = p / 'new_predictions'
    if not new_predictions.exists():
        new_predictions.mkdir()

    df_orig = pd.read_excel(input_)
    # print(df_orig)

    orig = df_orig.as_matrix()[:, 1:]

    alloy_id = pd.DataFrame(df_orig['Name'])
    # print(alloy_id)

    whereNan = np.isnan(list(orig[:, -1]))

    news = orig[whereNan]

    X_test = news[:, :-1]
    if df_transform is True:
        transform_df = pd.read_csv(transform_df)
        X_test = X_test[:,transform_df['0'].as_matrix()]

    predictions = rf.predict(X_test)
    # print(predictions)
    df_pred = pd.DataFrame(predictions)
    # print(df_pred)

    df_join = alloy_id.merge(df_pred, left_index=True, right_index=True)
    # df_join = df_join.rename(index=str, columns={'Name': 'Composition', '0': 'Predicted_EFA'})
    print(df_join)

    # prediction_csv = pd.DataFrame.to_csv(df_join, columns=['predictions']).to_csv('prediction.csv')
    df_join.to_csv(str(new_predictions) + '/HEMC_P67_Fe_model_1_' + str(time.strftime("%Y-%m-%d-%I-%M")) + '.csv')

def parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('input', type=str, help='path/to/data/file.xlsx')

    return parser.parse_args()


def main():
    args = parser()
    model_predict(args)


if __name__ == '__main__':
    main()
