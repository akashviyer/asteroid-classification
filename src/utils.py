import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle

from src.config import *

from src.exception import CustomException

from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler, PolynomialFeatures, LabelEncoder

import matplotlib.pyplot as plt

from optuna import Trial

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def specific_nn_generator(input_dim, num_outputs):
    def create_nn():
        model = Sequential()
        model.add(Dense(64, input_dim=input_dim, activation='relu'))
        model.add()
        model.add(Dense(64, activation='relu'))
        model.add(Dense(num_outputs, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    return create_nn

def get_train_df():
    return pd.read_csv(TRAIN_PATH)

def get_test_df():
    return pd.read_csv(TEST_PATH)

def get_data_df():
    return pd.read_csv(DATA_PATH)

def clean_column_names(X):
    df = X.copy()
    df.columns = [col.split('_', 1)[-1].strip('_') for col in df.columns]
    return df

def add_noise_feature(X):
    df = X.copy()
    random_noise = np.random.normal(loc=0, scale=1, size=X.shape[0])
    df['Noise'] = random_noise
    return df

def get_model_scores_and_params_string(config, model_names, model_scores):
    result_string = "------------------\n"
    result_string += "Model Cross Validated Scores:\n"
    result_string += "------------------\n"

    for name, score in zip(model_names, model_scores):
        result_string += f"{name}: {score}\n"

        if name == 'XGBoost':
            model_params = config.models[name].get_xgb_params()
        else:
            model_params = config.models[name].get_params()

        # Format model_params into a string and add it to the result_string
        params_string = ', '.join(f"{param}={value}" for param, value in model_params.items())
        result_string += f"Model Params for {name}: {params_string}\n\n"
    result_string += '------------------'    
    return result_string

def get_X_and_y(df, target_name=TARGET):
    return df.drop(target_name, axis=1), df[target_name]

def feature_adder(X):
    df = X.copy()
    numerical_df = pd.DataFrame()
    numerical_df['alb_per_diam'] = df['albedo'] / df['diameter']
    numerical_df['orbital_period^2'] = df['osc_semimaj_ax']  ** 3
    numerical_df['semimaj_times_inclin'] = df['osc_semimaj_ax'] * df['osc_inclin']

    #numerical_df['radius^2'] = (df['diameter'] / 2) ** 2
    #numerical_df['semimaj_times_inclin^2'] = numerical_df['semimaj_times_inclin'] ** 2
    # numerical_df['diam_times_orbital_period^2'] = df['diameter'] * numerical_df['orbital_period^2']
    # numerical_df['semimaj_times_orbital_period^2'] = df['osc_semimaj_ax'] * numerical_df['orbital_period^2']
    # numerical_df['albedo_times_orbital_period^2'] = df['albedo'] * numerical_df['orbital_period^2']

    standard_scaler = StandardScaler().set_output(transform='pandas')
    numerical_df_standard = standard_scaler.fit_transform(numerical_df)

    df = pd.concat([df, numerical_df_standard], axis=1)

    return df

def polynomial_feature_adder(X):
    df = X.copy()
    poly = PolynomialFeatures(2, include_bias=False).set_output(transform='pandas')
    df_poly = poly.fit_transform(df)
    return df_poly

def feature_excluder(excluded_list):
    def exclude_features(X):
        df = X.copy()
        df = df.drop(excluded_list, axis = 1)

def label_encode(s):
    le = LabelEncoder()
    return le.fit_transform(s)

def get_sample_ratio(cat, ratio, df='train'):
    if df == 'train':
        df = get_train_df()
        X, y = get_X_and_y(df=df, target_name=TARGET)
        
        return int(np.round((y == cat).sum() * ratio))

def mean_arrs(preds_arr):
    return np.mean(preds_arr, axis=0)

def weight_mean_arrays(trial:Trial, arrays, model_names):
    weights = []
    leftover_weight = 0.99
    for i in range(len(model_names) - 1):
        weight = trial.suggest_float(f'model{model_names[i]}', 0.01, leftover_weight)
        weights.append(weight)
        leftover_weight -= weight

    weights.append(1-sum(weights))
    weighted_arrays = [arr * weight for arr, weight in zip(arrays, weights)]
    weighted_mean = np.sum(weighted_arrays, axis=0)
    weighted_mean /= weighted_mean.sum(axis=1, keepdims=True)
    
    return weighted_mean

def agg_weighted_arrs(arrays):
    sum_so_far = None

    for arr in arrays:
        if sum_so_far is None:
            sum_so_far = arr
        else:
            sum_so_far += arr
        
    return sum_so_far / len(arrays)

def display_cm(model_name, confusion_matrices):
    sum_confusion_matrix = np.sum(confusion_matrices, axis=0)

    row_sums = sum_confusion_matrix.sum(axis=1, keepdims=True)
        
    average_confusion_matrix = (sum_confusion_matrix / row_sums)

    cm = plt.imshow(average_confusion_matrix, cmap=plt.cm.Blues, interpolation='nearest')
    plt.colorbar()

    for i in range(average_confusion_matrix.shape[0]):
        for j in range(average_confusion_matrix.shape[1]):
            plt.text(j, i, f"{average_confusion_matrix[i, j]:.2f}",
                ha="center", va="center", color="black", fontsize=10)
            
    ax = plt.gca()
    ax.set_xticks(np.arange(len(CLASS_LABELS)))
    ax.set_yticks(np.arange(len(CLASS_LABELS)))
    ax.set_xticklabels(CLASS_LABELS)
    ax.set_yticklabels(CLASS_LABELS)

    plt.title(f'Averaged Confusion Matrix ({model_name})')
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.show()

def get_str_label(num):
    return LABEL_ENCODE_MAP[num[0]]