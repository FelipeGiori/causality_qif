import fbleau
import pandas as pd
import numpy as np

from .utils import fbleau_train_test_split


def _check_class_frequency_nn_estimate(df, x, y):
    """
        For nn to run on categorical or binary data, we need to have at least
        one sample of each class both in the train and test data. The nn estimate
        model cannot handle unseen labels.
    """
    
    for column in [x, y]:
        var_count = df[column].value_counts()
        classes_to_remove = var_count[var_count < 6].index.values
        df = df[~df[column].isin(classes_to_remove)]
    
    return df


def _check_variable_dtype(df, x, y):
    """
        Checks if the categorical variables have a numeric enconding.
    """
    for column in [x, y]:
        if df[column].dtype.name == 'category':
            df[column] = df[column].cat.codes
    
    return df


def _check_train_test_same_classes(y_train, y_test):
    y_train_set = set(y_train)
    y_test_set = set(y_test)
    return y_train_set == y_test_set



class FBLEAU:
    """
        Computes the risk of a given dataset.
    """

    def compute_risk(self, df, x, y, estimate='nn', knn_strategy='ln', distance='euclidean'):
        
        if estimate is 'nn':
            df = _check_class_frequency_nn_estimate(df, x, y)

        df = _check_variable_dtype(df, x, y)
    
        X_train, X_test, y_train, y_test = fbleau_train_test_split(df, x, y)

        bayes_risk = {}

        if _check_train_test_same_classes(y_train, y_test):
            try:
                bayes_risk_direct = fbleau.run_fbleau(
                    X_train, y_train, X_test, y_test, estimate=estimate, knn_strategy=knn_strategy,
                    distance=distance, log_errors=False, log_individual_errors=False,delta=None,
                    qstop=None, absolute=False, scale=False)

                bayes_risk['direct'] = bayes_risk_direct

                risk_reverse = fbleau.run_fbleau(
                    X_test, y_test, X_train, y_train, estimate=estimate, knn_strategy=knn_strategy,
                    distance=distance, log_errors=False, log_individual_errors=False, delta=None,
                    qstop=None, absolute=False, scale=False)

                bayes_risk['reverse'] = risk_reverse
            except Exception as e:
                print(e)
                return {}
        
        else:
            print("Test data contains labels unseen in training data.")

        return bayes_risk


class QIF:
    """
        Computes the additive and multiplicative g-leakages using the frequentist
        approach (for now).
    """

    def __init__(self):
        pass


    def compute_leakage(self, df, x, y):
        return df, x, y