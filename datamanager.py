import os
import shutil as sh
import random as rnd
import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import skew
from scipy.stats import kurtosis


class DataManager:
  @staticmethod
  def split():
    folder = 'unpacked'
    database_file = 'database.csv'
    database_file_filename, database_file_ext = database_file.split('.')

    train_perc = 0.65
    val_perc = 0.15

    df = pd.read_csv(f'{folder}/{database_file}', encoding='utf')

    indices = list(range(len(df)))
    rnd.shuffle(indices)

    train_part = int(train_perc*len(df))
    val_part = int(val_perc*len(df))

    train_indices = indices[:train_part]
    val_indices = indices[train_part:train_part + val_part]
    test_indices  = indices[train_part + val_part:]

    train_database = df.iloc[train_indices]
    val_database = df.iloc[val_indices]
    test_database  = df.iloc[test_indices]

    train_database.to_csv(f'{database_file_filename}.train.{database_file_ext}', encoding='utf', index=False)
    val_database.to_csv(f'{database_file_filename}.val.{database_file_ext}', encoding='utf', index=False)
    test_database.to_csv( f'{database_file_filename}.test.{database_file_ext}',  encoding='utf', index=False)

    DataManager.create_datasets()
  
  @staticmethod
  def convert(dataset_folder, dataset_file, save_folder):
    # https://www.kaggle.com/code/oybekeraliev/time-domain-feature-extraction-methods
    def __mean__(signal):
      return np.sum(signal)/len(signal)

    def __std__(signal):
        return np.sqrt((np.sum((signal - (np.sum(signal)/len(signal)))**2))/(len(signal)-1))

    def __variance__(signal):
        return (np.sum(np.sqrt(abs(signal)))/len(signal))**2

    def __rms__(signal):
        return np.sqrt(np.sum(signal**2)/len(signal))

    def __min__(signal):
        return min(signal)

    def __absmax__(signal):
        return max(abs(signal))

    def __skewness__(signal):
        return skew(signal)

    def __kurtosis__(signal):
        return kurtosis(signal)

    dataset_df = pd.read_csv(dataset_file)
    rows = ['mean', 'std', 'variance', 'rms', 'absmax', 'skewness', 'kurtosis']
    funcs = [__mean__, __std__, __variance__, __rms__, __absmax__, __skewness__, __kurtosis__]

    '''

         |  ch0  |  ch1  |  ...  |  chK  |
         |-------|-------|-------|-------|
     st0 |       |       |  ...  |       |
     st1 |       |       |  ...  |       |
     ... |  ...  |  ...  |  ...  |  ...  |
     stN |       |       |  ...  |       |

         | ch0_st0 | ch0_st1 | ... | chK_stN |
         |---------|---------|-----|---------|
         |         |         | ... |         |

    '''
    files = dataset_df['File'].tolist()

    for file in files:
      df = pd.read_csv(f'{dataset_folder}/{file}')
      columns = df.columns
      cols = []
      stats = []
      for i, func in enumerate(funcs):
        cols.extend(list(map(lambda x: x + '_'+ rows[i], columns)))
        stats.extend(list(map(func, np.nan_to_num(df.to_numpy().T))))

      if not os.path.exists(save_folder):
        os.mkdir(save_folder)

      df_stats = pd.DataFrame(data=[stats], columns=cols)
      df_stats.to_csv(f'{save_folder}/{file}', index=False)


  def create_dataset(database_file, dataset_file):
    df = pd.read_csv(database_file, encoding='utf')
    names_list = df.iloc[:,0].tolist()
    groups_list = df.iloc[:,1].tolist()

    names = []
    groups = []

    data_folder = 'unpacked/data'

    for idx, folder in enumerate(names_list):
      files = os.listdir(f'{data_folder}/{folder}')
      names.extend(files)
      groups.extend([groups_list[idx]]*len(files))

      for file in files:
        sh.move(f'{data_folder}/{folder}/{file}', f'{data_folder}/temp/{file}')
      
      sh.rmtree(f'{data_folder}/{folder}')

    rus = RandomUnderSampler(random_state=42)
    X, y = rus.fit_resample(np.array(names).reshape([-1, 1]), groups)

    data = {'File' : X.reshape(-1).tolist(), 'Group' : y }

    df_out = pd.DataFrame(data)
    df_out.to_csv(f'{dataset_file}', encoding='utf', index=False)

  def create_datasets():
    database_file = 'database.csv'
    database_file_filename, database_file_ext = database_file.split('.')
    dataset_file_filename = 'dataset'

    train_database_file = database_file_filename + '.train.' + database_file_ext
    val_database_file = database_file_filename + '.val.' + database_file_ext
    test_database_file  = database_file_filename + '.test.'  + database_file_ext

    train_dataset_file = dataset_file_filename   + '.train.' + database_file_ext
    val_dataset_file = dataset_file_filename   + '.val.' + database_file_ext
    test_dataset_file  = dataset_file_filename   + '.test.'  + database_file_ext

    unzip_folder  =  'unpacked'
    data_folder   = f'{unzip_folder}/data'
    export_folder =  'samples'

    os.mkdir(f'{data_folder}/temp')

    DataManager.create_dataset(train_database_file, train_dataset_file)
    DataManager.create_dataset(val_database_file, val_dataset_file)
    DataManager.create_dataset(test_database_file,  test_dataset_file)

    files = os.listdir(f'{data_folder}/temp')

    if os.path.exists(f'{export_folder}'):
      sh.rmtree(f'{export_folder}')
    
    os.mkdir(f'{export_folder}')

    for file in files:
      sh.move(f'{data_folder}/temp/{file}', f'{export_folder}/{file}')

    sh.rmtree(unzip_folder)
    os.remove(train_database_file)
    os.remove(val_database_file)
    os.remove(test_database_file)