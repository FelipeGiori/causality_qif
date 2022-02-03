import numpy as np

from sklearn.model_selection import train_test_split


def fbleau_train_test_split(df, x, y):
    """
        Performs a train test split and formats the data to run the fbleau package.

        Parameters
        ----------
        df: pandas.DataFrame
            Dataframe object with your data

        x: str
            Name of the column representing your X variable (or secret)

        y: str
            Name of the column representing your Y variable (or channel output)

        It is important to note that, for some reason, the fbleau package can only run if
        the train and test sets have the same size and have specific dtypes.
    """

    Y = df[x]
    X = df[y]
        
    X_train, X_test, y_train, y_test = train_test_split(X.values, Y.values, test_size=0.5, stratify=Y.values)
    X_train = X_train.reshape(-1, 1).astype(np.float64)
    X_test = X_test.reshape(-1, 1).astype(np.float64)
    y_train = y_train.astype(np.uint64)
    y_test = y_test.astype(np.uint64)
    
    return X_train, X_test, y_train, y_test