import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH = "/home/felipegiori/Mestrado/research/src/mestrado/data/"

class CEPairs:
    """
        Class to enclapsulate each dataset information. The field are:
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


def load_ce_pairs(a_type=None, b_type=None, sup=True):
    """
        Load cause effect pairs from the causality benchmark data repository.
        
        Parameters
        ----------
        a_type: list, default=None
            List containing the types that you want to load. Possible types are: 'Numerical',
            'Categorical' and 'Binary'. If None, load all.
            
        b_type: list, default=None
            List containing the types that you want to load. Possible types are: 'Numerical',
            'Categorical' and 'Binary'. If None, load all.
            
        sup: bool, default=True
            If True loads the support dataset along with the main one. There is no apparent
            major difference in structure between the datasets.
        
        Returns
        -------
        dfs: list of CEPairs objects.
            See the doc string for the CEPairs class for further information.
    """
    
    df_info = load_ce_pairs_info(sup)
    
    if type(a_type) is list:
        a_type = [type_name.title() for type_name in a_type]
        df_info = df_info[df_info['a_type'].isin(a_type)]
    
    if type(b_type) is list:
        b_type = [type_name.title() for type_name in b_type]
        df_info = df_info[df_info['b_type'].isin(b_type)]
    
    ce_pairs_list = []
    for _, row in df_info.iterrows():
        file_path = DATA_PATH + row['sample_id'] + ".txt"
        
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


def load_ce_pairs_info(sup=True):
    ce_paths = ['ce_pairs_train/']
    if sup == True:
        ce_paths.append("ce_pairs_sup/")

    df_info_list = []
    for ce_path in ce_paths:
        df_types = pd.read_csv(DATA_PATH + ce_path + "CEdata_train_publicinfo.csv")
        df_types.columns = ['sample_id', 'a_type', 'b_type']

        df_target = pd.read_csv(DATA_PATH + ce_path + "CEdata_train_target.csv")
        df_target.columns = ['sample_id', 'target', 'details']

        df_info = df_types.merge(df_target)
        df_info_list.append(df_info)
        df_info['sample_id'] = ce_path + df_info['sample_id']

    df_info = pd.concat(df_info_list)
    return df_info



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
