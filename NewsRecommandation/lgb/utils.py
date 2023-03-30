import random

import numpy as np
import pandas as pd
import pickle

from sklearn.utils import shuffle

import const
from datetime import datetime
from tqdm import tqdm
import gc, os
import logging
import time
import lightgbm as lgb
from lgb_const import DATA_PATH, SAVE_PATH
from sklearn.preprocessing import MinMaxScaler

from baseline import process_data

#减少内存
def reduce_mem(df):
    starttime = time.time()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.isnull(c_min) or pd.isnull(c_max):
                continue
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction),time spend:{:2.2f} min'.format(end_mem,
                                                                                                    100 * (start_mem - end_mem) / start_mem,
                                                                                                    (time.time() - starttime) / 60))
    return df

# all_click_df指的是训练集
# sample_user_nums 采样作为验证集的用户数量
def trn_val_split(all_click_df, sample_user_nums):
    all_click = all_click_df
    all_user_ids = all_click.user_id.unique()

    # replace=True表示可以重复抽样，反之不可以
    sample_user_ids = np.random.choice(all_user_ids, size=sample_user_nums, replace=False)

    #验证集中的用户不在在训练集中
    click_val = all_click[all_click['user_id'].isin(sample_user_ids)]
    click_trn = all_click[~all_click['user_id'].isin(sample_user_ids)]

    # 将验证集中的最后一次点击给抽取出来作为答案
    click_val = click_val.sort_values(['user_id', 'click_timestamp'])
    val_ans = click_val.groupby('user_id').tail(1)

    click_val = click_val.groupby('user_id').apply(lambda x: x[:-1]).reset_index(drop=True)

    # 去除val_ans中某些用户只有一个点击数据的情况，如果该用户只有一个点击数据，又被分到ans中，
    # 那么训练集中就没有这个用户的点击数据，出现用户冷启动问题，给自己模型验证带来麻烦
    val_ans = val_ans[val_ans.user_id.isin(click_val.user_id.unique())]  # 保证答案中出现的用户再验证集中还有
    click_val = click_val[click_val.user_id.isin(val_ans.user_id.unique())]

    return click_trn, click_val, val_ans

# 获取当前数据的历史点击和最后一次点击
def get_hist_and_last_click(all_click):
    all_click = all_click.sort_values(by=['user_id', 'click_timestamp'])
    click_last_df = all_click.groupby('user_id').tail(1)

    # 如果用户只有一个点击，hist为空了，会导致训练的时候这个用户不可见，此时默认泄露一下
    def hist_func(user_df):
        if len(user_df) == 1:
            return user_df
        else:
            return user_df[:-1]

    click_hist_df = all_click.groupby('user_id').apply(hist_func).reset_index(drop=True)

    return click_hist_df, click_last_df


def get_trn_val_tst_data(trn_path, tst_path):

    click_trn_data = pd.read_csv(trn_path)
    click_trn_data = reduce_mem(click_trn_data)

    click_trn, click_val, val_ans = trn_val_split(click_trn_data)

    click_tst = pd.read_csv(tst_path)

    return click_trn, click_val, click_tst, val_ans

def divide_trn_val_tst_data(all_click, rate=0.1):
    all_click = reduce_mem(all_click)
    all_users = all_click['user_id'].unique()
    all_users = shuffle(all_users)
    print("所有用户数量：" + str(len(all_users)))
    len_users = len(all_users)
    len_tst = int(len_users * rate)
    print("测试集和验证集的用户数量：" + str(len_tst))
    len_trn = len_users - len_tst
    print("训练集的用户数量：" + str(len_trn))
    trn_users = np.random.choice(all_users, size=len_trn, replace=False)
    click_tst = all_click[~all_click['user_id'].isin(trn_users)]
    click_trn_val = all_click[all_click['user_id'].isin(trn_users)]
    click_trn, click_val, val_ans = trn_val_split(click_trn_val, len_tst)

    return click_trn, click_val, val_ans, click_tst

# 返回多路召回列表或者单路召回
def get_recall_list(save_path, single_recall_model=None, multi_recall=False):
    if multi_recall:
        return pickle.load(open(save_path + 'final_recall_items_dict_40.pkl', 'rb'))

    if single_recall_model == 'i2i_itemcf':
        return pickle.load(open(save_path + 'itemcf_lgb_.pkl', 'rb'))

#文章信息
def get_article_info_df():
    article_info_df = pd.read_csv(DATA_PATH + 'articles.csv')
    article_info_df = reduce_mem(article_info_df)

    return article_info_df

def get_article_cate_time_dict(article_info_df):
    article_cate_time_dict = {}
    for _, row in article_info_df.iterrows():
        art = row['article_id']
        cate = row['category_id']
        create_time = row['created_at_ts']
        article_cate_time_dict[art] = (cate, create_time)

    return article_cate_time_dict

def save_recall_dict(recall_dict, model):
    path = SAVE_PATH + model + '.pkl'
    #pickle保存大文件不行，在py3.4后引入了pickle4，可以保存大文件
    #pickle3.7之前protoc默认是3
    pickle.dump(recall_dict, open(path, 'wb'), protocol=4)


def normalize(cur_val, max_val, min_val):
    return (cur_val - min_val)/(max_val - min_val)

if __name__ == '__main__':
    all_click = pd.read_csv(DATA_PATH + 'all_data.csv')
    all_click = reduce_mem(all_click)

    click_trn, click_val, val_ans, click_tst = divide_trn_val_tst_data(all_click)

    click_trn.to_csv(DATA_PATH + 'click_trn_.csv', index=False, header=True)
    click_tst.to_csv(DATA_PATH + 'click_tst_.csv', index=False, header=True)
    click_val.append(val_ans).to_csv(DATA_PATH + 'click_val_.csv', index=False, header=True)

    click_trn_hist_, click_trn_last_ = get_hist_and_last_click(click_trn)
    click_tst_hist_, click_tst_last_ = get_hist_and_last_click(click_tst)

    trn_users = click_trn['user_id'].unique()
    tst_users = click_tst['user_id'].unique()
    val_users = click_val['user_id'].unique()

    print("len_trn:" + str(len(trn_users)))
    print("len_tst:" + str(len(tst_users)))
    print("len_val:" + str(len(val_users)))

    list_same1 = [data for data in trn_users if data in tst_users]
    list_same2 = [data for data in trn_users if data in val_users]
    list_same3 = [data for data in val_users if data in tst_users]
    print(list_same1)
    print(list_same2)
    print(list_same3)

    click_trn_hist_.to_csv(DATA_PATH + 'click_trn_hist_.csv', index=False, header=True)
    click_trn_last_.to_csv(DATA_PATH + 'click_trn_last_.csv', index=False, header=True)

    click_tst_hist_.to_csv(DATA_PATH + 'click_tst_hist_.csv', index=False, header=True)
    click_tst_last_.to_csv(DATA_PATH + 'click_tst_last_.csv', index=False, header=True)

    click_val.to_csv(DATA_PATH + 'click_val_hist_.csv', index=False, header=True)
    val_ans.to_csv(DATA_PATH + 'click_val_last_.csv', index=False, header=True)

    print("保存成功")