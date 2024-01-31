import openml
from sklearn import preprocessing


def label_encode(_y):
    le = preprocessing.LabelEncoder()
    return le.fit_transform(_y)


def get_X_and_y_from_openml_task(_task_id):
    openml_task = openml.tasks.get_task(_task_id)
    print(f"downloading data")
    _X, _y = openml_task.get_X_and_y()
    print(f"downloaded data")
    print(f"encoding y")
    _y = label_encode(_y)
    print(f"encoded y")
    return _X, _y
