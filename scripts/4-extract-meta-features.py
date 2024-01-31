# This script:
# 1. Receives a folder containing the AutoML-generated ensembles
# 2. Loads one ensemble at a time
# 3. Generates the meta-features
# 4. Stores the meta-features in an arff file
import re

import numpy
import openml
import pandas as pd

from pymfe.mfe import MFE

if __name__ == '__main__':
    # filename = "ensemble-learners/el_task-236_ens-size-5_budget-m-60_budget-f-60_seed-73.pkl"
    # configs = extract_integers_from(filename)
    # task_id = configs[0]
    # with open(filename, "rb") as file:
    #     ensemble = pickle.load(file)
    # print(ensemble.models)

    # all selected tasks
    tasks_ids = [75115, 75121, 75125, 75126, 189859, 189878, 167204, 75156, 190158, 168791, 146597, 167203, 167085, 190154, 126030, 146594, 189864, 189863, 189858, 75236, 211720, 75108, 146592, 166866, 146576, 75154, 146574, 75180, 166951, 189828, 3049, 75139, 167100, 75146, 126031, 288, 189899, 75133, 75092, 75100, 75239, 146596, 189840, 189779, 75199, 75235, 75233, 75159, 189875, 75187, 167086, 167089, 75089, 166905, 189845, 75219, 126021, 75185, 211721, 75147, 189843, 75142, 75112, 167094, 167101, 75149, 146593, 166950, 167096, 279, 166915, 75176, 75141, 75171, 2121, 75134, 166872, 166932, 167103, 2122, 75223, 3048, 75148, 3053, 126024, 167105, 75131, 166931, 75210, 146586, 166956, 166958, 189902, 189887, 189890, 167099, 189881, 189882, 189894, 189846]
    repository: pd.DataFrame = None
    for task_id in tasks_ids:
        print(f"working on task {task_id}")
        openml_task = openml.tasks.get_task(task_id)
        print(f"downloading data")
        # cannot use x from here since it is already label-encoded and mfe would not identify
        # categorical features
        _, y = openml_task.get_X_and_y()
        dataset = openml.datasets.get_dataset(openml_task.dataset_id)
        X, _, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)
        print(f"downloaded data")
        print(f"calculating meta-features")
        mfe = MFE(groups=["general", "statistical", "info-theory"])
        try:
            mfe.fit(X.values, y, verbose=True, transform_cat="one-hot")
            ft = mfe.extract(verbose=1)
            print(f"calculated meta-features")
            mf_df = pd.DataFrame([ft[1]], columns=ft[0])
            mf_df['task_id'] = pd.Series(task_id).values
            mf_df['successful_mf_extraction'] = pd.Series(True).values
            mf_df.to_csv("metafeatures.csv", mode='a')
        except ValueError:
            mf_df = pd.DataFrame([[numpy.NaN for x in range(0, len(mfe.features))]], columns=list(mfe.features))
            mf_df['task_id'] = pd.Series(task_id).values
            mf_df['successful_mf_extraction'] = pd.Series(False).values
            mf_df.to_csv("metafeatures.csv", mode='a')
    print(repository)
    # repository.to_csv("metafeatures.csv")
