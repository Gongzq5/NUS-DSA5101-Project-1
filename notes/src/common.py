import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

########################## PUBLIC ##########################

def loadDatasetFromFilepath(filepath: str) -> (np.array, np.array, np.array, np.array):
    """
    Overall, read and process and split to train and test
    @param filepath: the path of the data file
    @return (X_train, X_test, y_train, y_test)
    """
    data = readData(filepath)
    data = processData(data)
    return splitDataset(data)


########################## PRIVATE ##########################

def readData(filepath: str) -> pd.DataFrame:
    """
    Read data from file
    @param filepath: the path of the data file;
    @return: the data in DataFrame object;
    """
    df = pd.read_csv(filepath, ';')
    return df


def processData(origin_data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Process the original data
    @param origin_data: original data in DataFrame object;
    @return: the data like [(datas, labels)] after processing;
    """
    month_cat_dtype = pd.CategoricalDtype(['jan', 'feb', 'mar','apr', 'may', 'jun', 'jul', 'aug', 'sep',  'oct', 'nov', 'dec'], ordered=True)
    origin_data["month"] = origin_data["month"].astype(month_cat_dtype).cat.codes
    
    dummies_df = pd.get_dummies(origin_data, columns=["job", "marital", "education", "default", "housing", "loan", "contact", "poutcome", "y"])
    
    return dummies_df


def splitDataset(data: pd.DataFrame) -> (np.array, np.array, np.array, np.array):
    """
    Split the dataset to train/test and data/label
    @param data: all data after processing, the last col should be the target;
    @return: (X_train, X_test, y_train, y_test)
    """
    X = data[data.columns[0:-1]].values
    y = data[data.columns[-1]].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.33)
    return X_train, X_test, y_train, y_test