from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

def group_to_cat(s: str):
    group_mapping = { 'кон' : 0, 'алк' : 1 }
    return group_mapping[s] if s in group_mapping else None


class LogisticRegressionWrapper:
    
    def __init__(self):
        self.model = self.get_model()


    def get_model(self):
        return LogisticRegression(max_iter=200, solver='liblinear', multi_class='ovr')


    def train(self, train_dataset):
        train_dataset_file, train_dataset_folder = train_dataset

        df_train = pd.read_csv(train_dataset_file)
        train_files = df_train['File'].tolist()

        X_train = np.nan_to_num(np.array(list(map(lambda f: pd.read_csv(f'{train_dataset_folder}/{f}').to_numpy().reshape((-1)), train_files))))
        y_train = np.array([group_to_cat(g) for g in df_train['Group']])

        self.model.fit(X_train, y_train)


    def test(self, test_dataset):
        test_dataset_file, test_dataset_folder = test_dataset
        
        dataset_df = pd.read_csv(test_dataset_file)
        files = dataset_df['File'].tolist()

        X = np.nan_to_num(np.array(list(map(lambda f:  pd.read_csv(f'{test_dataset_folder}/{f}').to_numpy().reshape((-1)), files))))
        y = np.array([group_to_cat(g) for g in dataset_df['Group']])

        y_pred = self.model.predict(X)

        return accuracy_score(y, y_pred)
    

    @staticmethod
    def cycle(train_dataset, val_dataset, test_dataset):
        bar = tqdm(total=100)

        bar.set_description('Creating LogisticRegression model')
        model_wrapper = LogisticRegressionWrapper()
        bar.update(10)

        bar.set_description('Training LogisticRegression model')
        model_wrapper.train(train_dataset)
        bar.update(45)

        bar.set_description('Testing LogisticRegression model')
        acc = model_wrapper.test(test_dataset)
        bar.update(45)

        return acc


class DecisionTreeClassifierWrapper:
    
    def __init__(self):
        self.model = self.get_model()


    def get_model(self):
        return DecisionTreeClassifier()


    def train(self, train_dataset):
        train_dataset_file, train_dataset_folder = train_dataset

        df_train = pd.read_csv(train_dataset_file)
        train_files = df_train['File'].tolist()

        X_train = np.nan_to_num(np.array(list(map(lambda f: pd.read_csv(f'{train_dataset_folder}/{f}').to_numpy().reshape((-1)), train_files))))
        y_train = np.array([group_to_cat(g) for g in df_train['Group']])

        self.model.fit(X_train, y_train)


    def test(self, test_dataset):
        test_dataset_file, test_dataset_folder = test_dataset
        
        dataset_df = pd.read_csv(test_dataset_file)
        files = dataset_df['File'].tolist()

        X = np.nan_to_num(np.array(list(map(lambda f:  pd.read_csv(f'{test_dataset_folder}/{f}').to_numpy().reshape((-1)), files))))
        y = np.array([group_to_cat(g) for g in dataset_df['Group']])

        y_pred = self.model.predict(X)

        return accuracy_score(y, y_pred)
    

    @staticmethod
    def cycle(train_dataset, val_dataset, test_dataset):
        bar = tqdm(total=100)

        bar.set_description('Creating DecisionTreeClassifier model')
        model_wrapper = DecisionTreeClassifierWrapper()
        bar.update(10)

        bar.set_description('Training DecisionTreeClassifier model')
        model_wrapper.train(train_dataset)
        bar.update(45)

        bar.set_description('Testing DecisionTreeClassifier model')
        acc = model_wrapper.test(test_dataset)
        bar.update(45)

        return acc


class KNeighborsClassifierWrapper:
    
    def __init__(self):
        self.model = self.get_model()


    def get_model(self):
        return KNeighborsClassifier()


    def train(self, train_dataset):
        train_dataset_file, train_dataset_folder = train_dataset

        df_train = pd.read_csv(train_dataset_file)
        train_files = df_train['File'].tolist()

        X_train = np.nan_to_num(np.array(list(map(lambda f: pd.read_csv(f'{train_dataset_folder}/{f}').to_numpy().reshape((-1)), train_files))))
        y_train = np.array([group_to_cat(g) for g in df_train['Group']])

        self.model.fit(X_train, y_train)


    def test(self, test_dataset):
        test_dataset_file, test_dataset_folder = test_dataset
        
        dataset_df = pd.read_csv(test_dataset_file)
        files = dataset_df['File'].tolist()

        X = np.nan_to_num(np.array(list(map(lambda f:  pd.read_csv(f'{test_dataset_folder}/{f}').to_numpy().reshape((-1)), files))))
        y = np.array([group_to_cat(g) for g in dataset_df['Group']])

        y_pred = self.model.predict(X)

        return accuracy_score(y, y_pred)
    

    @staticmethod
    def cycle(train_dataset, val_dataset, test_dataset):
        bar = tqdm(total=100)

        bar.set_description('Creating KNeighborsClassifier model')
        model_wrapper = KNeighborsClassifierWrapper()
        bar.update(10)

        bar.set_description('Training KNeighborsClassifier model')
        model_wrapper.train(train_dataset)
        bar.update(45)

        bar.set_description('Testing KNeighborsClassifier model')
        acc = model_wrapper.test(test_dataset)
        bar.update(45)

        return acc


class LinearDiscriminantAnalysisWrapper:
    
    def __init__(self):
        self.model = self.get_model()


    def get_model(self):
        return LinearDiscriminantAnalysis()


    def train(self, train_dataset):
        train_dataset_file, train_dataset_folder = train_dataset

        df_train = pd.read_csv(train_dataset_file)
        train_files = df_train['File'].tolist()

        X_train = np.nan_to_num(np.array(list(map(lambda f: pd.read_csv(f'{train_dataset_folder}/{f}').to_numpy().reshape((-1)), train_files))))
        y_train = np.array([group_to_cat(g) for g in df_train['Group']])

        self.model.fit(X_train, y_train)


    def test(self, test_dataset):
        test_dataset_file, test_dataset_folder = test_dataset
        
        dataset_df = pd.read_csv(test_dataset_file)
        files = dataset_df['File'].tolist()

        X = np.nan_to_num(np.array(list(map(lambda f:  pd.read_csv(f'{test_dataset_folder}/{f}').to_numpy().reshape((-1)), files))))
        y = np.array([group_to_cat(g) for g in dataset_df['Group']])

        y_pred = self.model.predict(X)

        return accuracy_score(y, y_pred)
    

    @staticmethod
    def cycle(train_dataset, val_dataset, test_dataset):
        bar = tqdm(total=100)

        bar.set_description('Creating LinearDiscriminantAnalysis model')
        model_wrapper = LinearDiscriminantAnalysisWrapper()
        bar.update(10)

        bar.set_description('Training LinearDiscriminantAnalysis model')
        model_wrapper.train(train_dataset)
        bar.update(45)

        bar.set_description('Testing LinearDiscriminantAnalysis model')
        acc = model_wrapper.test(test_dataset)
        bar.update(45)

        return acc


class GaussianNBWrapper:
    
    def __init__(self):
        self.model = self.get_model()


    def get_model(self):
        return GaussianNB()


    def train(self, train_dataset):
        train_dataset_file, train_dataset_folder = train_dataset

        df_train = pd.read_csv(train_dataset_file)
        train_files = df_train['File'].tolist()

        X_train = np.nan_to_num(np.array(list(map(lambda f: pd.read_csv(f'{train_dataset_folder}/{f}').to_numpy().reshape((-1)), train_files))))
        y_train = np.array([group_to_cat(g) for g in df_train['Group']])

        self.model.fit(X_train, y_train)


    def test(self, test_dataset):
        test_dataset_file, test_dataset_folder = test_dataset
        
        dataset_df = pd.read_csv(test_dataset_file)
        files = dataset_df['File'].tolist()

        X = np.nan_to_num(np.array(list(map(lambda f:  pd.read_csv(f'{test_dataset_folder}/{f}').to_numpy().reshape((-1)), files))))
        y = np.array([group_to_cat(g) for g in dataset_df['Group']])

        y_pred = self.model.predict(X)

        return accuracy_score(y, y_pred)
    

    @staticmethod
    def cycle(train_dataset, val_dataset, test_dataset):
        bar = tqdm(total=100)

        bar.set_description('Creating GaussianNB model')
        model_wrapper = GaussianNBWrapper()
        bar.update(10)

        bar.set_description('Training GaussianNB model')
        model_wrapper.train(train_dataset)
        bar.update(45)

        bar.set_description('Testing GaussianNB model')
        acc = model_wrapper.test(test_dataset)
        bar.update(45)

        return acc


class SVCWrapper:
    
    def __init__(self):
        self.model = self.get_model()


    def get_model(self):
        return SVC()


    def train(self, train_dataset):
        train_dataset_file, train_dataset_folder = train_dataset

        df_train = pd.read_csv(train_dataset_file)
        train_files = df_train['File'].tolist()

        X_train = np.nan_to_num(np.array(list(map(lambda f: pd.read_csv(f'{train_dataset_folder}/{f}').to_numpy().reshape((-1)), train_files))))
        y_train = np.array([group_to_cat(g) for g in df_train['Group']])

        self.model.fit(X_train, y_train)


    def test(self, test_dataset):
        test_dataset_file, test_dataset_folder = test_dataset
        
        dataset_df = pd.read_csv(test_dataset_file)
        files = dataset_df['File'].tolist()

        X = np.nan_to_num(np.array(list(map(lambda f:  pd.read_csv(f'{test_dataset_folder}/{f}').to_numpy().reshape((-1)), files))))
        y = np.array([group_to_cat(g) for g in dataset_df['Group']])

        y_pred = self.model.predict(X)

        return accuracy_score(y, y_pred)
    

    @staticmethod
    def cycle(train_dataset, val_dataset, test_dataset):
        bar = tqdm(total=100)

        bar.set_description('Creating SVC model')
        model_wrapper = SVCWrapper()
        bar.update(10)

        bar.set_description('Training SVC model')
        model_wrapper.train(train_dataset)
        bar.update(45)

        bar.set_description('Testing SVC model')
        acc = model_wrapper.test(test_dataset)
        bar.update(45)

        return acc