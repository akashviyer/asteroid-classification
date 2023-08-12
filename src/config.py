RANDOM_SEED = 42
DATA_PATH = 'project/artifacts/data.csv'
TRAIN_PATH = 'project/artifacts/train.csv'
TEST_PATH = 'project/artifacts/test.csv'

TARGET = 'class1'
CLASS_LABELS = ['A', 'C', 'D', 'L', 'Q', 'S', 'V', 'X']

LABEL_ENCODE_MAP = {
    0:'A',
    1:'C',
    2:'D',
    3:'L',
    4:'Q',
    5:'S',
    6:'V',
    7:'X'
}

ALL_MODELS = []

IMPLEMENTED_MODELS = ['XGBoost', 'CatBoost', 'RandomForest']

BEST_XGB_PARAMS = {
        'objective': 'multi:softprob',
        'random_state': RANDOM_SEED,
        'learning_rate': 0.099, 
        'max_depth': 5,
        'min_child_weight': 0.2784016726023859,
        'subsample': 0.4, 
        'gamma': 1.8685907147813174
    }

BEST_CATBOOST_PARAMS = {'learning_rate': 0.01, 
                        'n_estimators': 512,
                        'max_depth': 8,
                        'l2_leaf_reg': 0.5433106820990699,
                        'rsm': 0.6973423627892332
                        }

BEST_RANDOMFOREST_PARAMS = {'n_estimators': 462,
                            'max_depth': 10, 
                            'min_samples_leaf': 6,
                            'min_samples_split': 15
                            }

XGBOOST_SMOTE_STRAT = {'sample_counts':'auto', 'k': 5}
RANDOMFOREST_SMOTE_STRAT = {'sample_counts':{ # weights adjusted from cv weights by a factor of 1.2194
        0: 3159,
        1: 10179,
        2: 4130,
        3: 6055,
        4: 2179,
        5: 11147,
        6: 2104,
        7: 3984
        },
        'k':24
    }

CATBOOST_SMOTE_STRAT = {'sample_counts':'auto', 'k': 5}


BEST_MODEL_CONFIGS = {
    'XGBoost': {'class_weight': None,
                'params': BEST_XGB_PARAMS,
                'smote': False,
                'smote_sampling_strategy': XGBOOST_SMOTE_STRAT,
                'under_sample': False},

    'CatBoost': {'class_weight': None,
                'params': BEST_CATBOOST_PARAMS,
                'smote': False,
                'smote_sampling_strategy': CATBOOST_SMOTE_STRAT,
                'under_sample': False},

    'RandomForest': {'class_weight': None,
                'params': BEST_RANDOMFOREST_PARAMS,
                'smote': False,
                'smote_sampling_strategy': RANDOMFOREST_SMOTE_STRAT,
                'under_sample': False},
    'model_weights': {'XGBoost': 0.24666802312783118, 'CatBoost': 0.46249480706693025, 'RandomForest':0.29083716980523855}

}