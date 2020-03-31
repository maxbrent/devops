import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class ETLDataPipeline:
    def __init__(self, train, test):
        self.train = train
        self.test = test

    def read_data(self):
        self.train = pd.read_csv(self.train)
        self.test = pd.read_csv(self.test)
        return self.train, self.test

    def drop_cols(self, col_list):
        self.train.drop(col_list, axis=1, inplace=True)
        self.train.dropna(axis=0, inplace=True)
        return self.train

    def convert_dtypes(self, col_list):
        for i in col_list:
            self.train[i] = self.train[i].astype('category')
        return self.train

    def encoder(self, col_list):
        encoding = LabelEncoder()

        for i in col_list:
            self.train[i] = encoding.fit_transform(self.train[i])
        return self.train

    def get_target(self, target_col_name):
        self.target = self.train.pop(target_col_name)
        return self.target

    def get_train_test(self, x, y):
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.33, random_state=42)
        return x_train, x_valid, y_train, y_valid
