import os
import pickle
import xgboost
import catboost
import numpy as np
import pandas as pd
import lightgbm as lgbm



from typing import Any
from tqdm.auto import tqdm
from copy import deepcopy as dp
from collections import Counter
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler

feats_name = ['cow_age', 
              'num_of_moms_children', 
              'num_of_dads_children', 
              'calving_year',
              'calving_month',
              'week_day', 
              'lactation',
              'farm', 
              'farmgroup',
              # 'total_milk',
              # 'total_milk_y',
              'total_milk_prev',
              'std_milk_prev',
              'mean_milk_prev',
              'calving_date_unix',
              'milk_yield_1_prev',
              'milk_yield_2_prev',
              # 'milk_yield_10_prev',
              # 'mother_id',
              # 'father_id',
              'num_of_children',
              'calving_date_diff',
              'calving_date_prev',
              'current_diff_2_1',
              # 'mother_id',
              # 'father_id'
              # 'animal_id'
              # 'calving_date_unix',
              # 'moms_same_lact', #TODO
              # 'moms_mean_same_lact',#TODO
              # 'moms_mean',#TODO
              # 'sisters_mean_same_lact', #TODO
              # 'sisters_mean',#TODO
              # 'farm_same_date_same_cow_mean', #TODO
              # 'farm_grp_same_date_same_cow_mean', #TODO
             ] + ['diff_yield_1', 'diff_yield_2']# + ['farm_mean_cow_age', 'farm_mean_cow_total_milk', 'farm_num_of_cows'] 
cat_feats = ['calving_month', 'week_day', 'farm', 'farmgroup']

# train_path      = os.path.join('data', 'train_local.csv')
# submission_path = os.path.join('data', 'submission_lr_local.csv')
# x_test_path     = os.path.join('data', 'test_local.csv')

train_path      = 'train.csv'
submission_path = os.path.join('data', 'submission.csv')
x_test_path     = os.path.join('data', 'X_test_public.csv')

class lin_reg_simple(BaseEstimator):
    def __init__(self):
        self.lr   = LinearRegression()
        self.lg   = lgbm.LGBMRegressor(n_estimators = 100)
        self.cat  = catboost.CatBoostRegressor(boosting_type = 'Ordered')
        self.xg   = xgboost.XGBRegressor(booster = 'gbtree', 
                                         enable_categorical=True, 
                                         verbosity = 2, 
                                         eta = 0.1)
        self.additional = 0
        
    def fit(self, X_train, y_train, addit):
        self.lg.fit  (X_train, y_train - addit,  categorical_feature = cat_feats)
        self.cat.fit (X_train, y_train - addit,  cat_features = cat_feats)
        self.xg.fit (X_train, y_train - addit)
        
    def predict(self, X_test, addit):
        # X_test = X_test.loc[:,total_feats]
        # X_test = self.scaler.transform(X_test)
        # return self.xg.predict(X_test) + addit
        return ((1/3) * (self.cat.predict(X_test) + addit) \
                    + (1/3) * (self.lg.predict(X_test) + addit) \
                        + (1/3) * (self.xg.predict(X_test) + addit))
    
class lin_reg_cascade(BaseEstimator):
    def __init__(self):
        self.lrs_lr  = {i: lin_reg_simple() for i in range(3, 11)}
        self.lrs_dif = {i: lin_reg_simple() for i in range(3, 11)}
        self.lrs     = {i: LinearRegression() for i in range(3, 11)}
    
    def fit(self, X_train, y_train):
        yield_feats = ['milk_yield_1', 'milk_yield_2']
        print (X_train.shape)
        for i in range(3, 11):
            y_train = X_train.loc[:,f'milk_yield_{i}']
            self.lrs[i].fit(pd.DataFrame(X_train['calving_date_unix']), 
                                         y_train)
            X_train['mean_yield_ts'] = self.lrs[i].predict(pd.DataFrame(X_train['calving_date_unix']))
            self.lrs_lr[i].fit(X_train.loc[:,feats_name + yield_feats + [f'milk_yield_{i}_prev']], \
                               y_train,
                               addit = X_train['mean_yield_ts'])
            
            self.lrs_dif[i].fit(X_train.loc[:,feats_name + yield_feats + [f'milk_yield_{i}_prev']], \
                                y_train, 
                                addit = X_train['milk_yield_2'])
    
    def predict(self, X_test):
        yield_feats = ['milk_yield_1', 'milk_yield_2']
        for i in range(3, 11):
            X_test['mean_yield_ts'] = self.lrs[i].predict(pd.DataFrame(X_test['calving_date_unix']))
            preds_lr   = self.lrs_lr[i].predict(X_test.loc[:,feats_name + yield_feats + [f'milk_yield_{i}_prev']], 
                                                addit = X_test['mean_yield_ts'])
            preds_diff = self.lrs_dif[i].predict(X_test.loc[:,feats_name + yield_feats + [f'milk_yield_{i}_prev']], 
                                                addit = X_test['milk_yield_2'])
            preds = 0.5 * preds_lr + 0.5 * preds_diff
            X_test[f'milk_yield_{i}'] = preds
        return X_test
    
def fill_na_df(df):
    yield_mask = ['milk_yield' in col for col in df.columns]
    new_df = pd.concat([(df.loc[:,yield_mask].fillna(method='ffill', axis = 1).iloc[:,:-1] \
            + df.loc[:,yield_mask].fillna(method='bfill', axis = 1).iloc[:,:-1]) / 2, df.loc[:,yield_mask].fillna(method='ffill', axis = 1).iloc[:,-1]], axis = 1)
    assert new_df.shape == new_df.dropna().shape
    df.loc[:,yield_mask] = new_df.values
    return df

ohe = None
def preprocess_df(df):
    df['birth_date']   = pd.to_datetime(df['birth_date'])
    df['calving_date'] = pd.to_datetime(df['calving_date'])
    df['animal_id_ind'] = df['animal_id'].str.slice(3).apply(int)
    df = df.set_index('animal_id_ind')
    # new_df = fill_na_df(dp(df))
    new_df = dp(df)
    yield_mask = ['milk_yield' in col for col in df.columns]
    new_df.loc[:,yield_mask] = new_df.loc[:,yield_mask].fillna(method = 'ffill', axis = 1)
    assert df.shape == new_df.shape
    assert df.shape == new_df.dropna().shape
    new_df = new_df.sort_values(by = 'calving_date')
    # new_df = new_df[new_df['calving_date'] >= '2012-06']
    return new_df

def preprocess_pedigree(pedigree):
    pedigree['mother_id'] = pedigree['mother_id'].fillna('hui-1')
    pedigree['father_id'] = pedigree['father_id'].fillna('hui-1')
    pedigree['animal_id'] = pedigree['animal_id'].str.slice(3).apply(int)
    pedigree['mother_id'] = pedigree['mother_id'].str.slice(3).apply(int)
    pedigree['father_id'] = pedigree['father_id'].str.slice(3).apply(int)
    pedigree = pedigree.set_index('animal_id')
    return pedigree

def merge_prev_lact_total_milk(df, train_df):
    dfs_by_lactation = [df[df['lactation'] == 1]]
    train_df = preprocess_df(train_df)
    
    yield_mask = ['milk_yield' in col for col in train_df.columns]
    train_df['total_milk'] = train_df.loc[:,yield_mask].sum(axis = 1)
    train_df = train_df.loc[:,['lactation', 'total_milk']]
    
    for i in range(2, 10):
        dfs_by_lactation.append(pd.merge(df[df['lactation'] == i], 
                                         train_df[train_df['lactation'] == i - 1], 
                                         # on = 'animal_id',
                                         left_index=True,
                                         right_index=True,
                                         how = 'left', 
                                         suffixes = ['', '_y']))
    return pd.concat(dfs_by_lactation, axis = 0)

def get_prev_by_col(df, col, new_col_name, IsTest, train_df = None):
    if not IsTest:
        df = df.sort_values(['animal_id', 'calving_date'])
        prev_total_milk = df.groupby('animal_id')[col]\
                            .apply(list)\
                                .apply(lambda x : ([-1] + x)[:-1])\
                                    .apply(pd.Series).stack().reset_index(drop=True)
        df[new_col_name] = prev_total_milk.values
    else:
        total_milk_train_map  = train_df[~train_df.index.duplicated(keep='last')].set_index('animal_id')[col].to_dict()
        df[new_col_name] = df['animal_id'].apply(lambda x : total_milk_train_map.get(x, -1))
        
    return df

def generate_complex_feats(df):
    yield_cols = [f'milk_yield_{i}' for i in range(1,11)]
    df['total_milk'] = df.loc[:,yield_cols].sum(axis = 1)
    df['std_milk'] = df.loc[:,yield_cols].std(axis = 1)
    df['mean_milk'] = df.loc[:,yield_cols].mean(axis = 1)
    df['cow_age'] = (df['calving_date'] - df['birth_date']).dt.days
    
    return df, ['total_milk', 'std_milk', 'mean_milk']

def generate_features(df, pedigree, IsTest):    
    df['cow_age'] = (df['calving_date'] - df['birth_date']).dt.days
    df['cow_age_rounded'] = (round(df['cow_age'] / 100) * 100).astype(int)
    # Не фича
    father_num_of_child_map = pedigree.groupby('father_id').count().sort_values(by = 'mother_id')['mother_id']
    mother_num_of_child_map = pedigree.groupby('mother_id').count().sort_values(by = 'father_id')['father_id']
    # print ('Before merge', df.shape)
    df = pd.merge(df, pedigree, 
                  left_index = True, 
                  right_index = True, 
                  how = 'left', suffixes = ['', '_y'])
    # print ('After merge', df.shape)
    df['num_of_moms_children'] = df['mother_id'].apply(lambda x : mother_num_of_child_map.get(x, 0))
    df['num_of_dads_children'] = df['father_id'].apply(lambda x : father_num_of_child_map.get(x, 0))
    df['num_of_children']      = df['animal_id'].apply(lambda x : father_num_of_child_map.get(x, 0))
    
    df['calving_year']  = df['calving_date'].dt.year
    df['calving_month'] = df['calving_date'].dt.month
    df['week_day']      = df['calving_date'].dt.isocalendar()['day'].astype(int)
    df['calving_date_unix'] = (df['calving_date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    
    #Сложные фичи на предыдущие лактации
    #Генерируем доп сложные фичи
    if not IsTest:
        df, complex_feats_name = generate_complex_feats(df)
        
        farm_mean_cow_age_map        = df.groupby(['farm'])['cow_age'].mean()
        farm_mean_cow_total_milk_map = df.groupby(['farm'])['total_milk'].mean()
        farm_num_of_cows_map             = df.groupby(['farm'])['animal_id'].nunique()
    else:
        train_df = pd.read_csv(train_path)
        train_df = preprocess_df(train_df)
        train_df, complex_feats_name = generate_complex_feats(train_df)
        
        farm_mean_cow_age_map        = train_df.groupby(['farm'])['cow_age'].mean()
        farm_mean_cow_total_milk_map = train_df.groupby(['farm'])['total_milk'].mean()
        farm_num_of_cows_map             = train_df.groupby(['farm'])['animal_id'].nunique()
    
    #немного фермофичей
    df['farm_mean_cow_age']        = df['farm'].apply(lambda x : farm_mean_cow_age_map.get(x, -1))
    df['farm_mean_cow_total_milk'] = df['farm'].apply(lambda x : farm_mean_cow_total_milk_map.get(x, -1))
    df['farm_num_of_cows']         = df['farm'].apply(lambda x : farm_num_of_cows_map.get(x, -1))
    
    
    #Берем сложные фичи с предыдущей лактации
    for col in tqdm(complex_feats_name + ['lactation', 'calving_date'] + [f'milk_yield_{i}' for i in range(1,11)]):
        if not IsTest:
            df = get_prev_by_col(df, col, col + '_prev', IsTest)
        else:
            df = get_prev_by_col(df, col, col + '_prev', IsTest, train_df)
    df['diff_yield_1'] =  df['milk_yield_1'] - df['milk_yield_1_prev']
    df['diff_yield_2'] =  df['milk_yield_2'] - df['milk_yield_2_prev']
    df.loc[df['milk_yield_1_prev'] == -1, 'diff_yield_1'] = -1
    df.loc[df['milk_yield_2_prev'] == -1, 'diff_yield_2'] = -1
    df = df.sort_values('calving_date')
    df['animal_id'] = df['animal_id'].astype(str).astype('category')
    df['mother_id'] = df['mother_id'].astype(str).astype('category')
    df['father_id'] = df['father_id'].astype(str).astype('category')

    df.loc[df['calving_date_prev'] == -1, 'calving_date_prev'] = pd.Timestamp("1970-01-01")
    df['calving_date_prev'] = pd.to_datetime(df['calving_date_prev'])
    df['calving_date_diff'] = None
    df['calving_date_diff'] = (df['calving_date'] - df['calving_date_prev']).dt.days
    df['calving_date_prev'] = (df['calving_date_prev'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    df['current_diff_2_1'] = df['milk_yield_2'] - df['milk_yield_1']
    df['calving_month'] = df['calving_month'].astype(int)
    df['farm']          = df['farm'].astype(int)
    df['farmgroup']     = df['farmgroup'].astype(int)
    df = df.sort_values('calving_date')
    return df

def fit():
    if os.path.exists('lr_cascade'):
        lr_cascade = pickle.load(open('lr_cascade', 'rb'))
    else:
        pedigree = pd.read_csv('pedigree.csv')
        df = pd.read_csv(train_path)
        df = preprocess_df(df)
        pedigree = preprocess_pedigree(pedigree)
        df = generate_features(df, pedigree, IsTest = False)
        lr_cascade = lin_reg_cascade()
        lr_cascade.fit(df, None)
        pickle.dump(lr_cascade, open('lr_cascade', 'wb'))
    return lr_cascade

def predict(model: Any, test_dataset_path: str) -> pd.DataFrame:
    # pedigree = pd.read_csv(os.path.join('data', 'pedigree.csv'))
    pedigree = pd.read_csv('pedigree.csv')
    df = pd.read_csv(test_dataset_path)
    df = preprocess_df(df)
    pedigree = preprocess_pedigree(pedigree)
    df = generate_features(df, pedigree, IsTest = True)
    # print (df.columns)
    x_test = model.predict(df)
    # TODO если меньше 0.5 то сделаем 0.5
    cols_to_ret = ['animal_id', 'lactation', 'milk_yield_3', 'milk_yield_4',
       'milk_yield_5', 'milk_yield_6', 'milk_yield_7', 'milk_yield_8',
       'milk_yield_9', 'milk_yield_10']
    return x_test.loc[:,cols_to_ret]

if __name__ == '__main__':
    lr_cascade = fit()
    submission = predict(lr_cascade, x_test_path)
    submission.to_csv(submission_path, sep=',', index=False)
