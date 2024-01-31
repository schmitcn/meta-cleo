import numpy as np
import openml
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import automlopen.constants as const


from scripts.meta_learner import MetaLearner

if __name__ == '__main__':
    # Load meta-features
    mf_df = pd.read_csv('metafeatures_cleansed.csv')
    # Remove those with unsuccessful extraction
    mf_df = mf_df[mf_df.successful_mf_extraction]
    mf_df = mf_df.drop('successful_mf_extraction', axis=1)
    # Drop all columns containing at least one nan value
    mf_df = mf_df.dropna(axis=1)
    # Convert inf to nan and drop nan again
    # Doing dropna again here as in the future we might avoid the most general dropna from above
    mf_df = mf_df.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

    # Load run configs
    run_configs = pd.read_csv('merged_run_configs.csv')

    # Create Meta-learner
    knn = NearestNeighbors(n_neighbors=3)
    mtlearner = MetaLearner(metafeatures_df=mf_df,
                            run_configs_df=run_configs,
                            knn=knn)

    # Load data
    task_id = 146576
    openml_task = openml.tasks.get_task(task_id)
    train_i, test_i = openml_task.get_train_test_split_indices()
    X_ndarray, y_ndarray = openml_task.get_X_and_y()
    dataset = openml.datasets.get_dataset(openml_task.dataset_id)
    X_non_encoded_df, _, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)

    # Generate model using meta-learner
    ensemble_learner = mtlearner.fit(X_non_encoded_df.values,
                                     y_ndarray,
                                     X_ndarray[train_i],
                                     y_ndarray[train_i],
                                     X_ndarray[test_i],
                                     y_ndarray[test_i])

    predictions = ensemble_learner.predict(X=X_ndarray[test_i], n_processes=1)
    perf_metrics = [const.ACCURACY, const.BALANCED_ACCURACY, const.PRECISION, const.PRECISION_MICRO,
                    const.PRECISION_MACRO, const.RECALL, const.RECALL_MICRO, const.RECALL_MACRO, const.F1_MICRO,
                    const.F1_MACRO, const.F1_SAMPLES, const.JACCARD_MICRO, const.JACCARD_MACRO]
    evaluation = ensemble_learner.evaluate(y_test=y_ndarray[test_i], y_pred=predictions, metric=perf_metrics)
    print(evaluation)
