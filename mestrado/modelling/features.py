from joblib.parallel import delayed
import qif
import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.preprocessing import KBinsDiscretizer

from mestrado import datasets

def joint_distribution(df, x, y):
    """
    Computes the joint distribution of x and y

    Parameters
    ----------
    df : pandas.DataFrame  
        Dataframe with x and y variables. We assume x and y are discrete.
    x : str
        Name of the x column
    y : [type]
        Name of the y column

    Returns
    -------
    joint_dist : pandas.DataFrame
        Dataframe with the joint distribution of x and y.
    """
      
    joint_dist = df.groupby([x, y]).size().unstack()
    joint_dist.columns = joint_dist.columns.get_level_values(0)
    joint_dist = joint_dist.div(joint_dist.sum().sum())
    joint_dist.fillna(0, inplace=True)
    return joint_dist


def compute_leakages(df, x, y):
    """
    Computes leakage and vulnerability measures between variables x and y.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with x and y variables. We assume x and y are discrete.
    x : str
        Name of the x column
    y : str
        Name of the y column

    Returns
    -------
    pandas.DataFrame
        Dataframe with the leakage and vulnerability measures.
    """
    
    J = joint_distribution(df, x, y)
    
    leakages = {}
    leakages.update(_compute_bayes_vulnerability(J))
    leakages.update(_compute_shannon_leakage(J))
    leakages.update(_compute_bayes_risk(J))
    
    corr, p_value = spearmanr(df[x], df[y])
    leakages['spearman'] = corr
    leakages['corr_p_value'] = p_value
    return leakages


def _extract_priors_and_channels(J):
    """
    Extracts the prior distribution and conditional distribution of x given y.

    Parameters
    ----------
    J : pandas.DataFrame
        Joint distribution of x and y

    Returns
    -------
    pandas.Series
        Prior distribution of x
    pandas.DataFrame
        Conditional distribution of x given y
    """
    
    p_x = J.sum(axis=1)
    C_xy = J.divide(p_x, axis=0)
    p_y = J.sum(axis=0)
    C_yx = J.T.divide(p_y, axis=0)
    return p_x, C_xy, p_y, C_yx


def _compute_bayes_vulnerability(J):
    """
    Computes the Bayes vulnerability measure.

    Parameters
    ----------
    p_x : pandas.Series
        Prior distribution of x
    p_y : pandas.Series
        Prior distribution of y
    C_xy : pandas.DataFrame
        Conditional distribution of x given y
    C_yx : pandas.DataFrame
        Conditional distribution of y given x

    Returns
    -------
    dict
        Bayes vulnerability measures
    """
    
    p_x, C_xy, p_y, C_yx = _extract_priors_and_channels(J)
    
    leakages = {}
    leakages['direct_bayes_vuln_mult_leakage'] = qif.measure.bayes_vuln.mult_leakage(p_x, C_xy)
    leakages['reverse_bayes_vuln_mult_leakage'] = qif.measure.bayes_vuln.mult_leakage(p_y, C_yx)
    leakages['direct_bayes_vuln_add_leakage'] = qif.measure.bayes_vuln.add_leakage(p_x, C_xy)
    leakages['reverse_bayes_vuln_add_leakage'] = qif.measure.bayes_vuln.add_leakage(p_y, C_yx)
    leakages['direct_bayes_vuln_min_entropy_leakage'] = qif.measure.bayes_vuln.min_entropy_leakage(p_x, C_xy)
    leakages['reverse_bayes_vuln_min_entropy_leakage'] = qif.measure.bayes_vuln.min_entropy_leakage(p_y, C_yx)
    leakages['direct_bayes_vuln_mult_capacity'] = qif.measure.bayes_vuln.mult_capacity(C_xy)
    leakages['reverse_bayes_vuln_mult_capacity'] = qif.measure.bayes_vuln.mult_capacity(C_yx)
    leakages['direct_bayes_vuln_posterior'] = qif.measure.bayes_vuln.posterior(p_x, C_xy)
    leakages['reverse_bayes_vuln_posterior'] = qif.measure.bayes_vuln.posterior(p_y, C_yx)
    leakages['direct_bayes_vuln_prior'] = qif.measure.bayes_vuln.prior(p_x)
    leakages['reverse_bayes_vuln_prior'] = qif.measure.bayes_vuln.prior(p_y)
    return leakages


def _compute_shannon_leakage(J):
    """
    Computes the Shannon leakage measure.

    Parameters
    ----------
    p_x : pandas.Series
        Prior distribution of x
    p_y : pandas.Series
        Prior distribution of y
    C_xy : pandas.DataFrame
        Conditional distribution of x given y
    C_yx : pandas.DataFrame
        Conditional distribution of y given x

    Returns
    -------
    dict
        Shannon leakage measure
    """
    
    p_x, C_xy, p_y, C_yx = _extract_priors_and_channels(J)
    
    leakages = {}
    leakages['direct_shannon_mult_leakage'] = qif.measure.shannon.mult_leakage(p_x, C_xy)
    leakages['reverse_shannon_mult_leakage'] = qif.measure.shannon.mult_leakage(p_y, C_yx)
    leakages['direct_shannon_posterior'] = qif.measure.shannon.posterior(p_x, C_xy)
    leakages['reverse_shannon_posterior'] = qif.measure.shannon.posterior(p_y, C_yx)
    return leakages


def _compute_bayes_risk(J):
    """
    Computes the bayes risk measure.

    Parameters
    ----------
    p_x : pandas.Series
        Prior distribution of x
    p_y : pandas.Series
        Prior distribution of y
    C_xy : pandas.DataFrame
        Conditional distribution of x given y
    C_yx : pandas.DataFrame
        Conditional distribution of y given x

    Returns
    -------
    dict
        Bayes risk measure
    """
    
    p_x, C_xy, p_y, C_yx = _extract_priors_and_channels(J)
    
    risks = {}
    risks['direct_bayes_risk_mult_risk'] = qif.measure.bayes_risk.mult_leakage(p_x, C_xy)
    risks['reverse_bayes_risk_mult_risk'] = qif.measure.bayes_risk.mult_leakage(p_y, C_yx)
    risks['direct_bayes_risk_add_risk'] = qif.measure.bayes_risk.add_leakage(p_x, C_xy)
    risks['reverse_bayes_risk_add_risk'] = qif.measure.bayes_risk.add_leakage(p_y, C_yx)
    risks['direct_bayes_risk_posterior'] = qif.measure.bayes_risk.posterior(p_x, C_xy)
    risks['reverse_bayes_risk_posterior'] = qif.measure.bayes_risk.posterior(p_y, C_yx)
    risks['direct_bayes_risk_prior'] = qif.measure.bayes_risk.prior(p_x)
    risks['reverse_bayes_risk_prior'] = qif.measure.bayes_risk.prior(p_y)
    qif.measure.bayes_risk
    return risks


def generate_ce_pairs_features(bins_min=4, bins_max=10, bins_step=1):
    """
    Computes the CE pairs features.

    Parameters
    ----------
    bins_min : int, optional
        Minimum number of bins, by default 4
    bins_max : int, optional
        Maximum number of bins, by default 10
    bins_step : int, optional
        Step size for bins gereration, by default 1

    Returns
    -------
    df_leakages : pandas.DataFrame
        Dtaframe with the CE pairs features for each dataset
    """    
    
    ce_pairs_list = datasets.load_ce_pairs()
    
    results = Parallel(n_jobs=-1)(delayed(_compute_ce_pair_leakages)(ce_pairs, n_bins) for ce_pairs in ce_pairs_list for n_bins in range(bins_min, bins_max, bins_step)) 
            
    df_leakages = pd.DataFrame(results)
    df_leakages = df_leakages.replace([np.inf, -np.inf], np.nan)
    
    return df_leakages
    
    
def _compute_ce_pair_leakages(ce_pair, n_bins):
    """
    Computes the CE pairs features.

    Parameters
    ----------
    ce_pair : qif.datasets.CEPair
        CE pair
    kbins : KBinsDiscretizer
        KBinsDiscretizer object
    n_bins : int
        Number of bins
    causal : bool
        True if causal, False if anticausal

    Returns
    -------
    dict
        CE pairs features
    """
    
    leakage = {}
    kbins = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    
    try:
        cause = 'a'
        effect = 'b'

        if ce_pair.a_type == "Numerical":
            ce_pair.data['a_cat'] = kbins.fit_transform(ce_pair.data['a'].values.reshape(-1, 1))
            cause = 'a_cat'

        if ce_pair.b_type == "Numerical":
            ce_pair.data['b_cat'] = kbins.fit_transform(ce_pair.data['b'].values.reshape(-1, 1))
            effect = 'b_cat'

        leakage = compute_leakages(ce_pair.data, cause, effect)
        leakage['name'] = ce_pair.name
        leakage['target'] = ce_pair.target
        leakage['details'] = ce_pair.details
        leakage['a_type'] = ce_pair.a_type
        leakage['b_type'] = ce_pair.b_type
        leakage['kbins'] = n_bins
    except Exception as e:
        print(e)

    return leakage


def compute_hand_crafted_features(df_leakages):
    """
    Adds hand engineered features to the DataFrame

    Parameters
    ----------
    df_leakages : pandas.DataFrame
        DataFrame with the computed leakages

    Returns
    -------
    df_leakages : pandas.DataFrame
        DataFrame with the new features
    """
    
    # Feature engineering
    df_leakages['direct_bayes_vuln_mult_leakage_normalized'] = \
        (2 * df_leakages['direct_bayes_vuln_posterior'])/(df_leakages['direct_bayes_vuln_prior'] + df_leakages['reverse_bayes_vuln_prior'])
    df_leakages['reverse_bayes_vuln_mult_leakage_normalized'] = \
        (2 * df_leakages['reverse_bayes_vuln_posterior'])/(df_leakages['direct_bayes_vuln_prior'] + df_leakages['reverse_bayes_vuln_prior'])
    df_leakages['direct_bayes_vuln_mult_leakage_perf'] = \
        df_leakages['direct_bayes_vuln_posterior']/df_leakages['direct_bayes_vuln_mult_capacity']
    df_leakages['reverse_bayes_vuln_mult_leakage_perf'] = \
        df_leakages['reverse_bayes_vuln_posterior']/df_leakages['reverse_bayes_vuln_mult_capacity']
    df_leakages['direct_bayes_vuln_mult_leakage_normalized_perf'] = \
        df_leakages['direct_bayes_vuln_mult_leakage_normalized']/df_leakages['direct_bayes_vuln_mult_leakage_perf']
    df_leakages['reverse_bayes_vuln_mult_leakage_normalized_perf'] = \
        df_leakages['reverse_bayes_vuln_mult_leakage_normalized']/df_leakages['reverse_bayes_vuln_mult_leakage_perf']
    df_leakages['direct_bayes_vuln_mult_leakage_ratio'] = \
        df_leakages['direct_bayes_vuln_mult_leakage']/df_leakages['reverse_bayes_vuln_mult_leakage']
    df_leakages['reverse_bayes_vuln_mult_leakage_ratio'] = \
        df_leakages['reverse_bayes_vuln_mult_leakage']/df_leakages['direct_bayes_vuln_mult_leakage']
    
    #df_leakages = pd.get_dummies(df_leakages, columns=['a_type', 'b_type'])
    return df_leakages