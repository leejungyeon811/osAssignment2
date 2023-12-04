import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

def sort_dataset(dataset_df):
    data = dataset_df.sort_values(by='year')
    return data

def split_dataset(dataset_df):
    dataset_df['salary'] *= 0.001

    X = dataset_df
    X_train = X.iloc[:1718]
    X_test = X.iloc[1718:]

    y = dataset_df['salary']
    y_train = y.iloc[:1718]
    y_test = y.iloc[1718:]

    return X_train, X_test, y_train, y_test

def extract_numerical_cols(dataset_df):
    data = pd.DataFrame(dataset_df, columns = ['age', 'G', 'PA', 'AB', 'R', 'H', '2B',
                                            '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP',
                                            'SO', 'GDP', 'fly', 'war'])
    return data

def train_predict_decision_tree(X_train, Y_train, X_test):
    dt_model = DecisionTreeRegressor()
    dt_model.fit(X_train, Y_train)
    dt_predictions = dt_model.predict(X_test)
    return dt_predictions


def train_predict_random_forest(X_train, Y_train, X_test):
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, Y_train)
    rf_predictions = rf_model.predict(X_test)
    return rf_predictions


def train_predict_svm(X_train, Y_train, X_test):
    pipe=make_pipeline(
        StandardScaler(),
        SVR()
    )
    pipe.fit(X_train,Y_train)
    svm_predictions = pipe.predict(X_test)
    return svm_predictions

def calculate_RMSE(labels, predictions):
    return np.sqrt(np.mean((predictions-labels)**2))


if __name__ == '__main__':
    # DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
    data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

    sorted_df = sort_dataset(data_df)
    X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)

    X_train = extract_numerical_cols(X_train)
    X_test = extract_numerical_cols(X_test)

    dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
    rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
    svm_predictions = train_predict_svm(X_train, Y_train, X_test)

    print("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))
    print("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))
    print("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))