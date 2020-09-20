import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE

########################## PUBLIC ##########################

def loadDatasetFromFilepath(filepath: str) -> (np.array, np.array, np.array, np.array):
    """
    Overall, read and process and split to train and test
    @param filepath: the path of the data file
    @return (X_train, X_test, y_train, y_test)
    """
    data = readData(filepath)
    data = processData(data)#data = processData_2(data)
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

def processData_2(bank_data: pd.DataFrame) -> (pd.DataFrame):
    """
    Process the original data
    @param origin_data: original data in DataFrame object;
    @return: the data like [(datas, labels)] after processing;
    """
    numeric_lst = ['age', 'balance', 'pdays', 'duration', 'campaign', 'previous']
    # intersection variables
    bank_data['house_balance'] = ((bank_data['housing'] == 'yes') * 0.5) * bank_data['balance'] + ((bank_data['housing'] == 'no') * 1) * bank_data['balance']
    bank_data['loan_balance'] = ((bank_data['loan'] == 'yes') * 0.5) * bank_data['balance'] + ((bank_data['loan'] == 'no') * 1) * bank_data['balance']
    # Polynomial
    bank_data['balance_poly'] = bank_data['balance'].apply(lambda x: x ** 2)
    bank_data['balance_sqrt'] = np.abs(bank_data['balance'].apply(lambda x: x ** 0.5))
    bank_data['contact_poly'] = bank_data['previous'].apply(lambda x: x ** 2)
    # discreteize age feature
    bank_data['is_young'] = (bank_data['age'] < 35) * 1
    bank_data['is_middle'] = ((bank_data['age'] >= 35) & (bank_data['age'] < 55)) * 1
    bank_data['is_old'] = (bank_data['age'] >= 55) * 1
    # Part III Encoding the categorical variables
    bank_data['is_previous_contacted'] = (bank_data['pdays'] == -1) * 1
    bank_data['is_debt'] = ((bank_data['loan'] == 'yes') * 1) + ((bank_data['housing'] == 'yes') * 1)
    bank_data['is_pdays'] = (bank_data['pdays'] == -1) * 1
    # Part IV standardize the variable to eliminate the skewness for linear model
    new_numeric_lst = ['balance_poly', 'balance_sqrt', 'contact_poly', 'house_balance', 'loan_balance']
    full_numeric_lst = new_numeric_lst + numeric_lst
    min_max_scaler = preprocessing.MinMaxScaler()
    bank_data_scaled = bank_data.copy()
    bank_data_scaled[full_numeric_lst] = min_max_scaler.fit_transform(bank_data_scaled[full_numeric_lst])
    bank_data_scaled.to_csv("bank_data_features_added_numeric_scaled.csv", index=False)
    # Handling category variables
    bank_data_scaled_onehot = processData_inner(bank_data_scaled)
    bank_data_scaled_onehot['y'] = (bank_data_scaled_onehot['y'] == 'yes') * 1
    # Resampling
    result_df = SMOTE_resampling(bank_data_scaled_onehot, 0.3)
    
    return result_df

def processData_inner(origin_data: pd.DataFrame, cat_lst: list = ["job", "marital", "education", "default", "housing", "loan", "contact", "poutcome"]) -> (pd.DataFrame, pd.DataFrame):
    """
    Process the original data
    @param origin_data: original data in DataFrame object;
    @return: the data like (datas, labels) after processing;
    """
    origin_data["month"] = origin_data["month"].astype(pd.CategoricalDtype(['jan', 'feb', 'mar','apr', 'may', 'jun', 'jul', 'aug', 'sep',  'oct', 'nov', 'dec'], ordered=True)).cat.codes
    dummies_df = pd.get_dummies(origin_data, columns=cat_lst)
    return dummies_df

def SMOTE_resampling(origin_data: pd.DataFrame, target_percent: float, target_variable: str='y', random_state: int=0):
    """
    Process the original data
    Note: for real usage, we need to apply it only on the test set. Otherwise it may lead to potential data leak.
    
    @param origin_data: original data in DataFrame object;
    @param target_percent: target percent we want to achieve
    @param random_state: random state for reproductivity. default is 0.
    @return: the data like after Smote resampling;
    """
    col_name_lst = origin_data.drop(columns=[target_variable]).columns
    X = origin_data.drop(columns=[target_variable]).values
    y = origin_data[target_variable].values.reshape(-1,1)
    print(X.shape)
    print(y.shape)
    
    sampler = SMOTE(sampling_strategy=target_percent, random_state=random_state)
    X_res, y_res = sampler.fit_resample(X, y)
    
    result_df = pd.DataFrame(X_res, columns=col_name_lst)
    result_df = pd.concat([result_df, pd.DataFrame(y_res, columns=[target_variable])], axis=1)
    return result_df


def splitDataset(data: pd.DataFrame) -> (np.array, np.array, np.array, np.array):
    """
    Split the dataset to train/test and data/label
    @param data: all data after processing, the last col should be the target;
    @return: (X_train, X_test, y_train, y_test)
    """
    X = data[data.columns[0:-2]].values
    y = data[data.columns[-1]].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.33)
    return X_train, X_test, y_train, y_test