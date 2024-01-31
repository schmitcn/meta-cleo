from os import listdir
from os.path import join

import pandas as pd

RUN_CONFIGS_FOLDER = 'ensemble-learners'


def find_csv_filenames(path_to_dir, suffix=".csv"):
    filenames = listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]


if __name__ == '__main__':
    files = find_csv_filenames(RUN_CONFIGS_FOLDER)
    df = pd.DataFrame()
    for file in files:
        data = pd.read_csv(join(RUN_CONFIGS_FOLDER, file))
        df = pd.concat([df, data], axis=0)
    df.to_csv('merged_run_configs.csv', index=False)
