import openml
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

task_ids = [
    2122,
    3048, 3049, 3053, 75089, 75108,
    279,
    288,
    75112,
    75115, 75121,
    2121,
    75125, 75126, 75131,
    75134,
    75139, 75141, 75142,
    75146,
    75147, 75148, 75149,
    75154,
    75156, 75171, 75176, 75180, 75185,
    75187, 75199, 75210, 75219,
    75223,
    75233, 75235,
    75236,
    75239,
    126021,
    126024, 126030, 126031,
    146574, 146576, 146586,
    146592,
    146593, 146594, 146596, 146597, 166866, 166872, 166905, 166915,
    166931, 166932, 166950, 166951, 166956, 166958, 167085, 167086, 167089,
    189859,
    189863,
    189864,
    189875,
    189878,
    189881, 189882, 189887, 189890, 189894, 189899, 189902,
    190158,
    211720, 211721
]

if __name__ == '__main__':

    for task_id in task_ids:
        # task_id = 75112
        print(f"Fetching data for task {task_id}")
        openml_task = openml.tasks.get_task(task_id)
        train_i, test_i = openml_task.get_train_test_split_indices()
        X_ndarray, y_ndarray = openml_task.get_X_and_y()
        dataset = openml.datasets.get_dataset(openml_task.dataset_id)
        X_non_encoded_df, _, _, _ = dataset.get_data(dataset_format="dataframe",
                                                     target=dataset.default_target_attribute)

        print(f"Running adaboost for task {task_id}")
        rf = AdaBoostClassifier(random_state=42)
        rf.fit(X_ndarray[train_i], y_ndarray[train_i])

        y_pred = rf.predict(X_ndarray[test_i])
        accuracy = accuracy_score(y_ndarray[test_i], y_pred)
        print("Accuracy:", accuracy)

        print(f"Storing evaluation results to file for task {task_id}")

        pd.DataFrame.from_records([{"task_id": task_id, "accuracy": accuracy}], index='task_id'). \
            to_csv('perf_results_adaboost.csv',
                   mode='a',
                   header=True,
                   index=True)
        # break
