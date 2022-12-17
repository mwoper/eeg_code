import os
import shutil as sh
import git
from datamanager import DataManager
from sklearn_models import LogisticRegressionWrapper, DecisionTreeClassifierWrapper, KNeighborsClassifierWrapper, LinearDiscriminantAnalysisWrapper, GaussianNBWrapper, SVCWrapper

class Main:
    repo_name              = 'eeg_dataset'
    unpacked_files_folder  = 'unpacked'
    samples_folder         = 'samples'
    converted_files_folder = 'converted'
    database_file          = 'database.csv'
    train_dataset_file     = 'dataset.train.csv'
    test_dataset_file      = 'dataset.test.csv'
    val_dataset_file       = 'dataset.val.csv'
    data_folder            = 'data'

    @staticmethod
    def download():
        git.Git().clone('https://github.com/mwoper/eeg_dataset.git')


    @staticmethod
    def unpack():
        if not os.path.exists(Main.unpacked_files_folder):
            os.mkdir(Main.unpacked_files_folder)

        if not os.path.exists(f'{Main.unpacked_files_folder}/{Main.data_folder}'):
            os.mkdir(f'{Main.unpacked_files_folder}/{Main.data_folder}')

        for i in range(0, 4):
            sh.unpack_archive(f'{Main.repo_name}/data.{i}.zip', f'{Main.unpacked_files_folder}/{Main.data_folder}', 'zip')
            sh.copyfile(f'{Main.repo_name}/{Main.database_file}', f'{Main.unpacked_files_folder}/{Main.database_file}')


    @staticmethod
    def main():
        # file:///C:/Users/tzorake/Downloads/Telegram%20Desktop/l02_03%20(1).pdf
        if not os.path.exists(Main.repo_name):
            Main.download()

        if not os.path.exists(Main.unpacked_files_folder):
            Main.unpack()

        if not os.path.exists(Main.samples_folder):
            DataManager.split()
        
        if not os.path.exists(Main.converted_files_folder):
            DataManager.convert(Main.samples_folder, Main.train_dataset_file, Main.converted_files_folder)
            DataManager.convert(Main.samples_folder, Main.test_dataset_file,  Main.converted_files_folder)
            DataManager.convert(Main.samples_folder, Main.val_dataset_file,  Main.converted_files_folder)

        models = [LogisticRegressionWrapper, DecisionTreeClassifierWrapper, KNeighborsClassifierWrapper, LinearDiscriminantAnalysisWrapper, GaussianNBWrapper, SVCWrapper]

        for model in models:
            acc = model.cycle(
                train_dataset = (Main.train_dataset_file, Main.converted_files_folder), 
                val_dataset   = (Main.val_dataset_file,   Main.converted_files_folder),
                test_dataset  = (Main.test_dataset_file,  Main.converted_files_folder)
            )

            print('Accuracy:', acc)

if __name__ == '__main__':
    Main.main()