import json
import os

import dill
import pandas as pd

path = os.environ.get('PROJECT_PATH', '..')


def get_model():
    file_name = os.listdir(f'{path}/data/models')
    with open(f'{path}/data/models/{file_name[0]}', 'rb') as file:
        model = dill.load(file)

    return model


def predict():
    model = get_model()
    file_path = f'{path}/data/test'
    pred_df = pd.DataFrame(columns=['id', 'result'])
    for file_json in os.listdir(file_path):
        with open(f'{file_path}/{file_json}', 'r') as file:
            df = pd.DataFrame.from_dict([json.load(file)])
            result = model.predict(df)
        pred_df.loc[len(pred_df), :] = [df['id'].values[0], result[0]]

    pred_df.to_csv(f'{path}/data/predictions/result_pred.csv')


if __name__ == '__main__':
    predict()
