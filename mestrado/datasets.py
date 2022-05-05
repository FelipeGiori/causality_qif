import os
import pickle
import numpy as np
import pandas as pd

from glob import glob
from pathlib import Path

DATA_PATH = "/home/felipegiori/Mestrado/research/src/mestrado/data/"


databases_ce_pairs = {
    'breast_tumor':[('tumor_size', 'inv_nodes'), ('tumor_size', 'deg_malig')],
    'cholesterol':[('chol', 'trestbps'), ('chol', 'fbs')],
    'pbc':[('stage', 'albumin'), ('stage', 'protime'), ('stage', 'bili')],
    'pollution':[('mort', 'ovr65')],
    'autompg':[('horsepower', 'mpg'), ('weight', 'mpg')],
    'cpu':[('myct', 'erp'), ('mmax', 'erp'), ('cach', 'erp')],
    'breastw':[('target', 'Clump_Thickness'), ('target', 'Cell_Shape_Uniformity'), ('target', 'Cell_Size_Uniformity')],
    'balance_scale':[('left_weight', 'target'), ('right_weight', 'target'), ('left_distance', 'target'), ('right_distance', 'target')],
    'servo':[('pgain', 'class'), ('vgain', 'class')],
    'sensory':[('trellis', 'score')],
    'pwlinear':[(f'a{n}', 'binaryClass') for n in range(1, 11)],
    'wisconsin':[('diagnosis', 'perimeter_mean'), ('diagnosis', 'smoothness_mean'), ('diagnosis', 'concavity_mean')]
}

class CEPairs:
    """
        Class to enclapsulate each dataset information. The fields are:
        - name: name of the data file
        - a_type: type of the A variable. Possible types are: Numerical, Categorical or Binary
        - b_type: type of the B variable. Possible types are: Numerical, Categorical or Binary
        - target: Direction of the cause effect pairs.
            - 1: A -> B
            - -1: B -> A
            - 0: A|B (independent) or A-B (correlated due to confounding)
        - details:1 for A->B; 2 for A<-B; 3 for A-B; 4 for A|B
    """
    def __init__(self, name, a_type, b_type, target, details, data):
        self.name = name
        self.a_type = a_type
        self.b_type = b_type
        self.target = target
        self.details = details
        self.data = data


def load_adult():
    adult_columns = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
                 "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week",
                 "native_country", "target"]
    df = pd.read_csv(DATA_PATH + "adult/adult.data", names=adult_columns)
    df2 = pd.read_csv(DATA_PATH + "adult/adult.test", names=adult_columns)
    df = pd.concat([df, df2])
    df.replace(to_replace=" ?", value=np.nan, inplace=True)
    df = df.dropna()
    df['target'] = df['target'].apply(_remove_dot)
    df['race_norm'] = df['race'].apply(_process_race_adult)
    return df


def load_german():
    cols = ['status_existing_checking_account', 'duration', 'credit_history', 'purpose', 'credit_amount', 'savings',
        'present_employment', 'installment_rate', 'sex', 'debtors', 'present_residance_since', 'property', 'age', 
        'installment_plans', 'housing', 'number_existing_credits', 'job', 'number_of_guarantors', 'phone',
        'foreign_worker', 'target']
    german = pd.read_csv(DATA_PATH + "german/german.data", sep=" ", names=cols)
    german['sex_norm'] = german['sex'].apply(_process_german_sex)
    return german


def load_compas():
    compas = pd.read_csv(DATA_PATH + "compas/compas_data.csv")
    compas['race_norm'] = compas['race'].apply(_process_compas_race)
    return compas


def load_breastw():
    df = pd.read_csv(DATA_PATH + "breast_w/breast-w_csv.csv")
    df.rename(columns={'Class':'target'}, inplace=True)

    dtypes = {
        'category':['target', 'Clump_Thickness', 'Cell_Shape_Uniformity', 'Cell_Size_Uniformity'],
        'int64':[],
        'float64':[]
    }

    df = _assign_dtypes_each_column(df, dtypes)
    return df


def load_hepatitis():
    cols = ['target', 'age', 'sex', 'steroid', 'antivirals', 'fatigue', 'malaise', 'anorexia', 'liver_big',
        'liver_firm', 'spleen_palpable', 'spiders', 'ascites', 'varices', 'bilirubin', 'alk_phostate',
        'sgot', 'albumin', 'protime', 'histology']
    df = pd.read_csv(DATA_PATH + "hepatitis/hepatitis.data", names=cols)
    return df


def load_breast_tumor():
    cols = ['target', 'age', 'menopause', 'tumor_size', 'inv_nodes', 'node_caps', 'deg_malig', 'breast',
        'breast_quad', 'irradiat']
    df = pd.read_csv(DATA_PATH + "breast_tumor/breast-cancer.data", names=cols)

    dtypes = {
        'category':['target', 'age', 'menopause', 'tumor_size', 'inv_nodes', 'node_caps',
                    'deg_malig', 'breast', 'breast_quad', 'irradiat'],
        'int64':[],
        'float64':[]
    }

    df = _assign_dtypes_each_column(df, dtypes)

    return df


def load_cholesterol():
    cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
        'ca', 'thal', 'target']
    df = pd.read_csv(DATA_PATH + "cholesterol/processed.cleveland.data", names=cols)
    
    dtypes = {
        'category':['sex', 'cp', 'fbs', 'restecg'],
        'int64':['age'],
        'float64':['trestbps', 'chol']
    }

    df = _assign_dtypes_each_column(df, dtypes)
    return df


def load_balance_scale():
    cols = ['target', 'left_weight', 'left_distance', 'right_weight', 'right_distance']
    df = pd.read_csv(DATA_PATH + "balance_scale/balance-scale.data", names=cols)

    dtypes = {
        'category':['target', 'left_weight', 'right_weight', 'left_distance', 'right_distance'],
        'int64':[],
        'float64':[]
    }

    df = _assign_dtypes_each_column(df, dtypes)

    return df


def load_heart_disease():
    cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak',
        'slope', 'ca', 'thal', 'target']
    df = pd.read_csv(DATA_PATH + "heart_disease/processed.cleveland.data", names=cols)
    return df


def load_sensory():
    df = pd.read_csv(DATA_PATH + "sensory/sensory.csv")
    
    dtypes = {
        'category':['trellis', 'score'],
        'int64':[],
        'float64':[]
    }

    df = _assign_dtypes_each_column(df, dtypes)
    return df


def load_pbc():
    df = pd.read_csv(DATA_PATH + "pbc/pbc.csv")

    dtypes = {
        'category':['stage'],
        'int64':[],
        'float64':['albumin', 'protime', 'bili']
    }

    df = _assign_dtypes_each_column(df, dtypes)
    df.dropna(subset=['protime'], inplace=True)

    return df


def load_pollution():
    df = pd.read_csv(DATA_PATH + "pollution/pollution.csv")

    dtypes = {
        'category':[],
        'int64':[],
        'float64':['ovr65', 'mort']
    }

    df = _assign_dtypes_each_column(df, dtypes)

    return df


def load_autompg():
    df = pd.read_csv(DATA_PATH + "autompg/autompg.csv")
    df = df[df['horsepower'] != '?']

    dtypes = {
        'category':[],
        'int64':['horsepower'],
        'float64':['weight', 'mpg']
    }

    df = _assign_dtypes_each_column(df, dtypes)

    return df


def load_cpu():
    cols = ['vendor_name', 'model', 'myct', 'mmin', 'mmax', 'cach', 'chmin', 'chmax', 'prp', 'erp']
    df = pd.read_csv(DATA_PATH + "cpu/machine.data", names=cols)

    dtypes = {
        'category':[],
        'int64':['myct', 'mmax', 'erp'],
        'float64':[]
    }

    df = _assign_dtypes_each_column(df, dtypes)

    return df


def load_servo():
    cols = ['motor', 'screw', 'pgain', 'vgain', 'class']
    df = pd.read_csv(DATA_PATH + "servo/servo.data", names=cols)

    dtypes = {
        'category':['motor', 'screw', 'pgain', 'vgain'],
        'int64':[],
        'float64':[]
    }

    df = _assign_dtypes_each_column(df, dtypes)

    return df


def load_pwlinear():
    df = pd.read_csv(DATA_PATH + "pwlinear/pwLinear.csv")

    for column in df.columns:
        df[column] = df[column].astype('category')

    return df


def load_wisconsin():
    df = pd.read_csv(DATA_PATH + "wisconsin/wisconsin.csv")

    dtypes = {
        'category':['diagnosis'],
        'int64':[],
        'float64':['perimeter_mean', 'smoothness_mean', 'concavity_mean']
    }

    df = _assign_dtypes_each_column(df, dtypes)

    return df


def load_ce_pairs_info(name, target=True):
    publicinfo_path = DATA_PATH + f"ce_pairs/{name}/{name}_publicinfo.csv"
    df_info = pd.read_csv(publicinfo_path, usecols=['SampleID', 'A type', 'B type'])
    df_info.columns = ['sample_id', 'a_type', 'b_type']
    
    if target:
        target_path = DATA_PATH + f"ce_pairs/{name}/{name}_target.csv"
        if os.path.exists(target_path):
            df_target = pd.read_csv(target_path)
            df_target.columns = ['sample_id', 'target', 'details']
            df_info = df_info.merge(df_target, on='sample_id')
        else:
            raise FileNotFoundError("Target csv file does not exist!")
    
    df_info['sample_id'] = name + "/" + df_info['sample_id']
    return df_info


def load_ce_pair_csv(name, df_info, target=True):
    file_path = DATA_PATH + f"ce_pairs/{name}/{name}_pairs.csv"
    
    df = pd.read_csv(file_path)
    df.rename(columns={'SampleID':'sample_id', 'A':'a', 'B':'b'}, inplace=True)
    df['sample_id'] = name + "/" + df['sample_id']
    df = df.merge(df_info, on='sample_id')
    
    ce_pairs_list = []

    for _, row in df.iterrows():
        df_sample = _format_ce_pair_from_csv(row)
        ce_pair = CEPairs(
            row['sample_id'],
            row['a_type'],
            row['b_type'],
            row['target'],
            row['details'],
            df_sample
        )
        
        ce_pairs_list.append(ce_pair)
        
    return ce_pairs_list


def load_ce_pair_split(df_info, target=True):
    ce_pairs_list = []
    for _, row in df_info.iterrows():
        file_path = DATA_PATH + f"ce_pairs/{row['sample_id']}.txt"
        
        df_sample = pd.read_csv(file_path, sep="\t", header=None)
        df_sample.columns = ['a', 'b']
        
        ce_pair = CEPairs(
            row['sample_id'],
            row['a_type'],
            row['b_type'],
            row['target'],
            row['details'],
            df_sample
        )
        
        ce_pairs_list.append(ce_pair)
        
    return ce_pairs_list


def load_ce_pair(name, a_type=None, b_type=None):
    df_info = load_ce_pairs_info(name)
    
    if type(a_type) is list:
        a_type = [type_name.title() for type_name in a_type]
        df_info = df_info[df_info['a_type'].isin(a_type)]
    
    if type(b_type) is list:
        b_type = [type_name.title() for type_name in b_type]
        df_info = df_info[df_info['b_type'].isin(b_type)]
        
    if _get_data_format(name) == 'split':
        ce_pair = load_ce_pair_split(df_info)
    elif _get_data_format(name) == 'csv':
        ce_pair = load_ce_pair_csv(name, df_info)
        
    return ce_pair


def load_ce_pairs(databases=None, a_type=None, b_type=None, use_cache=True):
    if databases is None:
        databases = ['CEfinal_train', 'SUP1data', 'SUP2data', 'SUP3data', 'CEnew_valid']
    elif type(databases) is not list:
        raise TypeError("databases argument must be a list")
    
    if use_cache:
        cache_path = DATA_PATH + "/ce_pairs/cache/ce_pairs.pkl"
        if os.path.isfile(cache_path):
            with open(cache_path, "rb") as f:
                ce_pairs_list = pickle.load(f)
            
            # TODO: filter ce_pairs_list by a_type and b_type
            return ce_pairs_list
        else:
            print("Could not find cache file. Loading data from raw...")
        
    ce_pairs_list = []
    
    for database in databases:
        ce_pairs_database = load_ce_pair(database, a_type, b_type)
        ce_pairs_list.extend(ce_pairs_database)
        
    return ce_pairs_list


def _get_data_format(name):
    folder_path = DATA_PATH + "ce_pairs/" + f"{name}/"
    files = glob(folder_path + "*")
    if len([filename for filename in files if ".txt" in filename]) > 1:
        return 'split'
    else:
        return 'csv'


def _format_ce_pair_from_csv(row):
    a = list(map(eval, row['a'].strip().split(" ")))
    b = list(map(eval, row['b'].strip().split(" ")))
    data = pd.DataFrame([a, b])
    data = data.T
    data.columns = ['a', 'b']
    return data


def _remove_dot(s):
    return s.strip(".")


def _process_race_adult(s):
    if(s == " White"):
        return " White"
    else:
        return " Other"
    

def _process_german_sex(s):
    if(s == 'A91'):
        return 'male'
    elif(s == 'A92'):
        return 'female'
    elif(s == 'A93'):
        return 'male'
    elif(s == 'A94'):
        return 'male'
    elif(s == 'A95'):
        return 'female'
    

def _process_compas_race(s):
    if(s == 'Caucasian'):
        return "Protected"
    else:
        return "Favored"


def _assign_dtypes_each_column(df, dtypes):
    """
        Assign the dtypes to each column according to the dtypes dict.

        Parameters
        ----------
        df: pandas.DataFrame
        
        dtypes: dict
            Each key is a dtype and each value is a list with the columns with that
            dtype.

        Returns
        -------
        pandas.DataFrame with the correct dtypes.
    """

    for dtype, columns in dtypes.items():
        for column in columns:
            df[column] = df[column].astype(dtype)

    return df
