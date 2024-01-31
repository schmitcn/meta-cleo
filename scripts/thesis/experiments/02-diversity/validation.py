from sklearn.neighbors import NearestNeighbors
from scripts import MetaLearner
import numpy as np
import openml
import pandas as pd

task_ids = [
     # 2122,
     # 3048, 3049, 3053, 75089, 75108,
     # 279,
     # 288,
     # 75112,
     # 75115, 75121,
    # 2121, # index out of bounds
     # 75125, 75126, 75131,
     # 75134,
     # 75139, 75141, 75142,
     # 75146,
     # 75147, 75148, 75149,
     # 75154,
     # 75156, 75171, 75176, 75180, 75185,
     # 75187, 75199, 75210, 75219,
    #  75223,
    #  75233, 75235,
    #  75236,
    # 75239,
     # 126021,
     # 126024, 126030, 126031,
     # 146574, 146576, 146586,
     # 146592,
     # 146593, 146594, 146596, 146597, 166866, 166872, 166905, 166915,
     # 166931, 166932, 166950, 166951, 166956, 166958, 167085, 167086, 167089,
     # 189859,
     # 189863,
     # 189864,
     # 189875,
     # 189878,
    189881, 189882, 189887, 189890, 189894, 189899, 189902,
    # 190158,
    211720, 211721
]


def get_mf_df(mf_file: str) -> pd.DataFrame:
    # Load meta-features
    mf = pd.read_csv(mf_file, float_precision='round_trip')
    # Remove those with unsuccessful extraction
    mf = mf[mf.successful_mf_extraction]
    mf = mf.drop('successful_mf_extraction', axis=1)
    # Drop all columns containing at least one nan value
    mf = mf.dropna(axis=1)
    # Convert inf to nan and drop nan again
    # Doing dropna again here as in the future we might avoid the most general dropna from above
    mf = mf.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    return mf


if __name__ == '__main__':
    # Load Meta-features
    mf_df = get_mf_df('../../../metafeatures_cleansed.csv')
    # Load run configs
    run_configs = pd.read_csv('../../../merged_run_configs.csv')
    # Create Meta-learner
    knn = NearestNeighbors(n_neighbors=2)  # first is the dataset itself and second is the most similar dataset
    mtlearner = MetaLearner(metafeatures_df=mf_df,
                            run_configs_df=run_configs,
                            knn=knn)

    for task_id in task_ids:
        # task_id = 75112
        print(f"Fetching data for task {task_id}")
        openml_task = openml.tasks.get_task(task_id)
        train_i, test_i = openml_task.get_train_test_split_indices()
        X_ndarray, y_ndarray = openml_task.get_X_and_y()
        dataset = openml.datasets.get_dataset(openml_task.dataset_id)
        X_non_encoded_df, _, _, _ = dataset.get_data(dataset_format="dataframe",
                                                     target=dataset.default_target_attribute)

        print(f"Deciding selected ensemble learner for task {task_id}")
        # returns list of (ensemble_learner, metadata)
        ensembles_info = mtlearner.fit(X_non_encoded_df.values,
                                       y_ndarray,
                                       X_ndarray[train_i],
                                       y_ndarray[train_i],
                                       X_ndarray[test_i],
                                       y_ndarray[test_i],
                                       n_classes=len(np.unique(y_ndarray)),
                                       task_id=task_id,
                                       timeout_seconds=3600)

        print(f"Storing evaluation results to file for task {task_id}")
        for ensemble_info in ensembles_info:
            ensemble_info[1]['task_id'] = task_id
            # making the multivalued evaluations as strings to avoid flattening
            ensemble_info[1]['precision'] = str(ensemble_info[1]['precision'])
            ensemble_info[1]['recall'] = str(ensemble_info[1]['recall'])

        pd.DataFrame.from_records([x[1] for x in ensembles_info], index='task_id'). \
            to_csv('../03-ranking/perf_results_nearest_neighbor-retrain_all_ensembles_from_nn.csv',
                   mode='a',
                   header=True,
                   index=True)
        # break
