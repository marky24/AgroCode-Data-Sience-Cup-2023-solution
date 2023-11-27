import os
import argparse
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error

def calculate_score(test_dataset_path, submission_path):
    test_dataset = pd.read_csv(test_dataset_path)
    submission = pd.read_csv(submission_path)

    merged_Y = test_dataset.merge(
        submission, left_on=['animal_id', 'lactation'], right_on=['animal_id', 'lactation'], how='outer'
    )
    mean_squared_errors = []

    median_value = np.nanmedian(submission[[f'milk_yield_{i}' for i in range(3, 11)]].values)

    for index, row in tqdm(merged_Y.iterrows()):
        # yapf: disable
        arr_real = (
            row[
                [f'milk_yield_{i}_x' for i in range(3, 11)]
            ].fillna(method='ffill')
             .fillna(method='bfill')
        )
        arr_predict = (
            row[
                [f'milk_yield_{i}_y' for i in range(3, 11)]
            ].fillna(method='ffill')
             .fillna(method='bfill')
             .fillna(value=median_value)
             .fillna(value=0)
        )
        # yapf: enable
        try:
            mean_squared_errors.append(mean_squared_error(arr_real, arr_predict))
        except:
            continue

    rmse_score = np.sqrt(np.mean(mean_squared_errors))

    return rmse_score

def file_path(string):
    if os.path.exists(string):
        return string
    else:
        raise NotAFileError(string)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run solution")
    parser.add_argument("-p", "--predicts", help="Path to predictions file", required=True)
    parser.add_argument("-t", "--target", help="Path to targets file", type=file_path, required=True)
    args = parser.parse_args()
    score = calculate_score(test_dataset_path=args.target, 
                            submission_path=args.predicts)
    print ('Your score is: ', score)