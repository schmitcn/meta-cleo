import argparse
import logging
import sys
import traceback

import pandas as pd
import automlopen as ao
import openml
import csv

from automlopen.optimization.bo import BayesianOptimization

openml.config.apikey = ''

FILENAME_TEMPLATE = "{}/el_task-{}_ens-size-{}_budget-m-{}_budget-f-{}_seed-{}"


def set_log_level():
    logging.basicConfig(stream=sys.stdout)
    logger = logging.getLogger('automlopen')
    logger.setLevel(logging.DEBUG)


if __name__ == '__main__':
    set_log_level()
    parser = argparse.ArgumentParser(description="Run Ensemble Learner and store result",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-t", "--task-id", type=int, help="OpenML Task Id")
    parser.add_argument("-e", "--ensemble-size", type=int, help="Target ensemble size")
    parser.add_argument("-m", "--budget-m", type=int, help="Budget in trials for model selection")
    parser.add_argument("-f", "--budget-f", type=int, help="Budget in trial for fusion selection")
    parser.add_argument("-s", "--seed", type=int, help="Seed value")
    parser.add_argument("-d", "--result-directory", type=str, help="Path where to store the ensemble learner pickle "
                                                                   "file")
    args = parser.parse_args()
    config = vars(args)
    print(f"Received parameters {config}")

    task_id = config['task_id']
    print(f"working on {task_id}")

    # Download and split data
    openml_task = openml.tasks.get_task(task_id)
    train_i, test_i = openml_task.get_train_test_split_indices()
    openml_dataset = openml.datasets.get_dataset(openml_task.dataset_id)
    X_ndarray, y_nd_array = openml_task.get_X_and_y()
    X_train = pd.DataFrame.from_records(X_ndarray[train_i])
    y_train = pd.Series(y_nd_array[train_i])
    X_test = pd.DataFrame.from_records(X_ndarray[test_i])
    y_test = pd.Series(y_nd_array[test_i])

    result_generic_filename = FILENAME_TEMPLATE.format(config['result_directory'],
                                                       task_id,
                                                       config['ensemble_size'],
                                                       config['budget_m'],
                                                       config['budget_f'],
                                                       config['seed'])
    pkl_filename = result_generic_filename + ".pkl"
    try:
        # Create ensemble using automl
        el = ao.EnsembleLearner(ens_size=config['ensemble_size'],
                                budget_m=config['budget_m'],
                                budget_f=config['budget_f'],
                                seed=config['seed'],
                                solver=BayesianOptimization)
        ensemble_metrics, div_metrics, models_metrics = el.fit_evaluate(X_train=X_train,
                                                                        y_train=y_train,
                                                                        X_test=X_test,
                                                                        y_test=y_test)
        # Store ensemble and metrics
        el.save_ensemble(pkl_filename)
        run_config_to_store = {"openml_task_id": task_id,
                               "openml_dataset_id": openml_dataset.dataset_id,
                               "ens_pkl_file": pkl_filename,
                               "budget_m": config['budget_m'],
                               "budget_f": config['budget_f'],
                               "seed": config['seed'],
                               "ensemble_size": len(el.models),
                               "base_classifiers_metrics": models_metrics}
        run_config_to_store.update(ensemble_metrics)
        run_config_to_store.update(div_metrics)
        with open(result_generic_filename + ".csv", 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=run_config_to_store.keys())
            writer.writeheader()
            writer.writerow(run_config_to_store)
    except Exception as e:
        # Store txt file to indicate failure of a config run
        with open(result_generic_filename + ".txt", "w") as f:
            format_exc = traceback.format_exc()
            print(format_exc)
            f.write(format_exc)
