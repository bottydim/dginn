import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import inspect

current_path = os.path.realpath(inspect.getfile(inspect.currentframe()))
print(current_path)
parent_dir = os.path.dirname(current_path)
dataset_path = os.path.join(os.path.dirname(parent_dir), "data")
#### ADD Dataset TODO
# 1.
# get_fx
# 2.
# sensitive_features_dict
# 3.
#

german_sensitive_features_dict = {"gender": 8, "age": 12}


# german_sensitive_features_dict = {"gender": 8, "age": 12, "foreign_work": -1}


def get_german(sensitive_feature_name, filepath=None, remove_z=False, **kwargs):
    if filepath is None:
        filepath = os.path.join(dataset_path, "german.data-numeric")
    df = pd.read_csv(filepath, header=None, delim_whitespace=True)

    # change label to 0/1
    cols = list(df)
    label_idx = len(cols) - 1
    df[label_idx] = df[label_idx].map({2: 0, 1: 1})

    M = df.values
    X = M[:, :-1]
    y = M[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    z_idx = get_z_idx(sensitive_feature_name, german_sensitive_features_dict)
    Xtr, Xts, Ztr, Zts, ytr, yts = extract_z(X_test, X_train, y_test, y_train, z_idx, remove_z=remove_z)
    return Xtr, Xts, ytr, yts, Ztr, Zts


def get_z_idx(sensitive_feature_name, sensitive_features_dict):
    z_idx = sensitive_features_dict.get(sensitive_feature_name, None)
    if z_idx is None:
        print("Feature {} not recognized".format(sensitive_feature_name))
        z_idx = 0
    return z_idx


adult_sensitive_features_dict = {"gender": 9, "age": 0, "race": 8}
adult_column_names = ['age', 'workclass', 'fnlwgt', 'education',
                      'education-num', 'marital-status', 'occupation', 'relationship',
                      'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                      'native-country', 'income-per-year']


def get_adult(sensitive_feature_name, scale=True, remove_z=False, verbose=0, file_path=dataset_path,
              **kwargs):
    if scale:
        file_name = "adult.npz"
        arr_holder = np.load(os.path.join(file_path, file_name))
        fit_scale = arr_holder[arr_holder.files[0]]

        M = fit_scale
    else:
        file_name = "adult.data"
        df = pd.read_csv(os.path.join(file_path, file_name), sep=",", header=None)
        categorical_columns = []
        for c in df.columns:
            if df[c].dtype is np.dtype('O'):
                categorical_columns.append(c)
        from collections import defaultdict
        d_label = defaultdict(LabelEncoder)
        fit = df.apply(lambda x: d_label[x.name].fit_transform(x) if x.dtype == np.dtype('O') else x)
        M = fit.values

        # TODO persistent & dynamic reading
    X = M[:, :-1]
    y = M[:, -1]
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if verbose:
        # print shapes
        for x in [X_train, X_test, y_train, y_test]:
            print(x.shape)
    z_idx = get_z_idx(sensitive_feature_name, adult_sensitive_features_dict)
    Xtr, Xts, Ztr, Zts, ytr, yts = extract_z(X_test, X_train, y_test, y_train, z_idx, remove_z)
    return Xtr, Xts, ytr, yts, Ztr, Zts


bank_sensitive_features_dict = {"marital": 2, "age": 0}


def get_bank(sensitive_feature_name, remove_z=False, file_path=dataset_path, **kwargs):
    # assume
    # 0 age
    # 2 marital

    z_idx = get_z_idx(sensitive_feature_name, bank_sensitive_features_dict)

    file_name = "bank.npz"
    file_add = os.path.join(file_path, file_name)
    if os.path.exists(file_add):
        arr_holder = np.load(os.path.join(file_path, file_name))
        fit_scale = arr_holder[arr_holder.files[0]]
    else:
        print("LOADING BANK WARNING! SHOULD ONLY BE USED IN SPECIAL CASES")
        filepath = "/home/btd26/datasets/bank/bank-additional/bank-additional-full.csv"
        filepath = os.path.join(dataset_path, "bank-additional-full.csv")
        df = pd.read_csv(filepath, sep=";")
        categorical_columns = []
        for c in df.columns:
            if df[c].dtype is np.dtype('O'):
                categorical_columns.append(c)
        from collections import defaultdict
        d_label = defaultdict(LabelEncoder)
        fit = df.apply(lambda x: d_label[x.name].fit_transform(x) if x.dtype == np.dtype('O') else x)
        # scale
        d_scale = defaultdict(StandardScaler)
        scaler = StandardScaler()
        feature_columns = fit.columns[:-1]
        fit_scale = scaler.fit_transform(fit[feature_columns])
        fit_scale = np.concatenate([fit_scale, fit.iloc[:, -1].values.reshape(-1, 1)], axis=1)

    M = fit_scale
    # M = fit.values
    X = M[:, :-1]
    y = M[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    Xtr, Xts, Ztr, Zts, ytr, yts = extract_z(X_test, X_train, y_test, y_train, z_idx, remove_z=remove_z)

    return Xtr, Xts, ytr, yts, Ztr, Zts


compas_sensitive_features_dict = {"sex": 0, "race": 2, "age": 1}


def get_compass(sensitive_feature_name, remove_z=False, file_path=dataset_path,
                file_name="compas.npy", **kwargs):
    z_idx = get_z_idx(sensitive_feature_name, compas_sensitive_features_dict)

    # load file
    file_add = os.path.join(file_path, file_name)
    if os.path.exists(file_add):
        M = np.load(file_add)
    else:
        from aif360.datasets import CompasDataset
        compas = CompasDataset()
        M = np.concatenate([compas.features, compas.labels], axis=1)
        np.save(file_add, M)
    X = M[:, :-1]
    y = M[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    Xtr, Xts, Ztr, Zts, ytr, yts = extract_z(X_test, X_train, y_test, y_train, z_idx, remove_z=remove_z)

    return Xtr, Xts, ytr, yts, Ztr, Zts


dataset_fs = [get_german, get_adult, get_bank, get_compass]
dataset_names = ["german", "adult", "bank", "compas"]
n_features_list = [24, 14, 20, 401]
dataset_feature_dict = {"german": german_sensitive_features_dict, "bank": bank_sensitive_features_dict,
                        "adult": adult_sensitive_features_dict, "compas": compas_sensitive_features_dict}

feature_name_dict = {}
for ds in dataset_names:
    keys = dataset_feature_dict[ds]
    feature_name_dict[ds] = {v: k for k, v in dataset_feature_dict[ds].items() for ds in dataset_names}

f_sensitive_list = [sorted(feature_name_dict[ds].keys()) for ds in dataset_names]


def get_n_features_list():
    n_features_list = []
    for i, f in enumerate(dataset_fs):
        Xtr, Xts, ytr, yts, Ztr, Zts = f(0, remove_z=False)
        X_test, X_train, Y_test, Y_train = prep_data(Xtr, Xts, ytr, yts, verbose=0)
        n_features = X_train.shape[-1]
        n_features_list.append(n_features)
    return n_features_list


def get_data_list():
    data_list = []
    for i, f in enumerate(dataset_fs):
        Xtr, Xts, ytr, yts, Ztr, Zts = f(0, remove_z=False)
        data_list.append(prep_data(Xtr, Xts, ytr, yts, verbose=0))
    return data_list


def generate_x_labels(dataset_names, f_sensitive_list):
    x_labels = []
    for i, f in enumerate(dataset_names):
        #     print("models #{}".format(len((models_list_p[i]))))
        for j, m in enumerate(f_sensitive_list[i]):
            data_set_name = dataset_names[i]
            feature_name = feature_name_dict[data_set_name][f_sensitive_list[i][j]]
            x_labels.append(data_set_name + "-" + feature_name)
    return x_labels


x_labels = generate_x_labels(dataset_names, f_sensitive_list)


# f_sensitive_list = [[8, 12], [9, 0, 8], [0, 2], [0, 2]]


def prep_data(Xtr, Xts, ytr, yts, verbose=1):
    from tensorflow.keras.utils import to_categorical
    X_train = np.hstack([Xtr])
    Y_train = to_categorical(ytr)
    X_test = np.hstack([Xts])
    Y_test = to_categorical(yts)

    for x in [X_train, X_test, Y_train, Y_test]:
        if verbose > 1:
            print(x.shape)

    return X_test, X_train, Y_test, Y_train


def extract_z(X_test, X_train, y_test, y_train, z_idx, remove_z):
    if remove_z:
        ix = np.delete(np.arange(X_train.shape[1]), z_idx)
    else:
        ix = np.arange(X_train.shape[1])
    Xtr = X_train[:, ix]
    Ztr = X_train[:, z_idx].reshape(-1, 1)
    Xts = X_test[:, ix]
    Zts = X_test[:, z_idx].reshape(-1, 1)
    ytr = y_train
    yts = y_test
    return Xtr, Xts, Ztr, Zts, ytr, yts


def nulify_feature(X_train, Y_train, i):
    x = np.copy(X_train[:, :])
    x[:, i] = 0
    return x, Y_train


binarise_dict = {"german": {
    "gender": {"targets": [[2], [1, 3, 4]], "assign_vals": [0, 1]},
    "age": {"targets": [[1], [2]], "assign_vals": [0, 1]},
},
    "adult":
        {"gender": {"targets": None, "assign_vals": [0, 1]},
         "age": {"targets": lambda x: x > -0.99570562, "assign_vals": [0, 1]},
         "race": {"targets": lambda x: x > 0.39, "assign_vals": [0, 1]}, }
    ,
    "bank": {"age": {"targets": lambda x: x > -1.44169297e+00, "assign_vals": [0, 1]},
             "marital": {"targets": lambda x: x == np.unique(x)[2], "assign_vals": [0, 1]}, },
    "compas": {"age": {"targets": lambda x: x > 25, "assign_vals": [0, 1]},
               "sex": {"targets": [[0], [1]], "assign_vals": [1, 0]},
               "race": {"targets": [[0], [1]], "assign_vals": [0, 1]},
               }
}


def sample_data(data, sample, use_train):
    X_test, X_train, Y_test, Y_train = data
    if use_train:
        X = X_train
        Y = Y_train

    else:
        X = X_test
        Y = Y_test
    # SAMPLE
    if sample is not None:
        random.seed(30)
        n_samples = int(X.shape[0])
        ix_sample = random.sample(range(n_samples), min(sample, n_samples))
        X = X[ix_sample, :]
        Y = Y[ix_sample, :]
    return X, Y
