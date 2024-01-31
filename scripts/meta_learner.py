import pickle

import pandas as pd
import sklearn as skl
from numpy import ndarray
from pymfe.mfe import MFE
from sklearn.preprocessing import MinMaxScaler
import automlopen.constants as const
from func_timeout import func_timeout, FunctionTimedOut

import automlopen as ao

ANY_ID = 1234


class MetaLearner:

    def __init__(self, metafeatures_df: pd.DataFrame = None,
                 run_configs_df: pd.DataFrame = None,
                 knn: skl.neighbors.NearestNeighbors = None):
        self.metafeatures_df = metafeatures_df.set_index('task_id')
        self.run_configs_df = run_configs_df
        self.mfe = MFE(groups=["general", "statistical", "info-theory"])
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.metafeatures_df)
        self.metafeatures_df[self.metafeatures_df.columns] = self.scaler.transform(
            self.metafeatures_df[self.metafeatures_df.columns])
        self.knn = knn.fit(self.metafeatures_df)

    def fit(self, X_non_encoded: ndarray, y: ndarray,
            X_train: ndarray = None, y_train: ndarray = None,
            X_test: ndarray = None, y_test: ndarray = None,
            n_classes: int = None, task_id: int = None,
            timeout_seconds: int = 60, top_k: int = -1) -> (ao.EnsembleLearner, dict):
        # calculate mf and create a dataframe of it
        # new_mf = self.mf_from_dataset(X_non_encoded, y) # calculates for new dataset
        new_mf = self.precalculated_mf_for(task_id) # fetches from mf repo - used for framework evaluations
        # get nearest neighbor and its task id(s)
        nearest_neighbors_distances, nearest_neighbors_indices = self.knn.kneighbors(new_mf, return_distance=True)
        nearest_neighbor_task_ids = self.metafeatures_df.index[nearest_neighbors_indices]
        # assign the first nearest neighbor as the selected one
        nearest_neighbor_task_id = nearest_neighbor_task_ids[0][0]
        nearest_neighbor_distance = nearest_neighbors_distances[0][0]
        # get the second-nearest neighbor when the first is an identical dataset
        if nearest_neighbor_task_id == task_id:  # distance is zero between the meta-features, means identical datasets
            nearest_neighbor_task_id = nearest_neighbor_task_ids[0][1]
            nearest_neighbor_distance = nearest_neighbors_distances[0][1]
        # get run config for the nearest neighbor
        run_configs_from_nn = self.run_configs_df.loc[self.run_configs_df['openml_task_id'] == nearest_neighbor_task_id]
        # rank run configs and select top k
        run_configs_from_nn = run_configs_from_nn.loc[
            run_configs_from_nn.accuracy
            .sort_values(ascending=False)
            .index]

        print(f"Closest task is {nearest_neighbor_task_id}")
        print(f"Ensembles from closest task and ordered for evaluation are "
              f"{[x['ens_pkl_file'] for _, x in run_configs_from_nn.iterrows()]}")
        print(f"Meta-learner will run until {top_k} ensemble is(are) re-trained successfully")
        print(f"Timeout is set for {timeout_seconds} seconds")

        evaluated_ensembles = []
        rank = -1
        for i, run_config in run_configs_from_nn.iterrows():
            rank = rank + 1

            try:
                ensemble: ao.EnsembleLearner

                with open(f"../../../{run_config['ens_pkl_file']}", "rb") as file:
                    ensemble = pickle.load(file)

                # try re-train the ensemble in a given timeframe
                func_timeout(timeout_seconds, ensemble.retrain_ensemble, args=(X_train, y_train, n_classes))

                predictions = ensemble.predict(X=X_test, n_processes=1, n_classes=n_classes)
                perf_metrics = [const.ACCURACY, const.BALANCED_ACCURACY, const.PRECISION, const.PRECISION_MICRO,
                                const.PRECISION_MACRO, const.RECALL, const.RECALL_MICRO, const.RECALL_MACRO,
                                const.F1_MICRO,
                                const.F1_MACRO, const.JACCARD_MICRO, const.JACCARD_MACRO]
                evaluation = ensemble.evaluate(y_test=y_test,
                                               y_pred=predictions,
                                               metric=perf_metrics,
                                               n_classes=n_classes)
                metadata = {
                    "nearest_neighbor_task_id": nearest_neighbor_task_id,
                    "nearest_neighbor_distance": nearest_neighbor_distance,
                    'nearest_neighbor_used_run_config': run_config['ens_pkl_file'],
                    'accuracy_rank_in_original_task': rank
                }
                # move evaluation info to metadata
                for key in evaluation.keys():
                    metadata[key] = evaluation[key]
                # add original task info to metadata
                for key in run_config.keys():
                    metadata['original_task_' + key] = run_config[key]

                evaluated_ensembles.append((ensemble, metadata))
            except FunctionTimedOut:
                print(f"Ensemble {run_config['ens_pkl_file']} re-training for task {task_id} timed out.")
                continue
            except Exception as e:
                print(f"Exception raised during re-training ensemble {run_config['ens_pkl_file']}: {e}")
                continue

            # evaluated successfully k ensembles
            # required since with top_k=5 we could have e.g. 6 ensembles tried out, but only 5 were successful
            if len(evaluated_ensembles) == top_k:
                break

        return evaluated_ensembles

    def mf_from_dataset(self, X_non_encoded, y):
        self.mfe.fit(X_non_encoded, y, transform_cat="one-hot")
        ft = self.mfe.extract(verbose=1)
        new_mf = pd.DataFrame([ft[1]], columns=ft[0])
        # any id as it is ignored when calculating the KNN
        new_mf['task_id'] = pd.Series(ANY_ID).values
        new_mf = new_mf.set_index('task_id')
        # keeping only the mf used in the repo
        new_mf = new_mf[self.metafeatures_df.columns]
        # scaling
        new_mf[new_mf.columns] = self.scaler.transform(new_mf[new_mf.columns])
        return new_mf

    def precalculated_mf_for(self, task_id: int) -> pd.Series:
        return self.metafeatures_df.loc[[task_id]]

