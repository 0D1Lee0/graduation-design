import numpy as np
import pandas as pd
import pickle
import const
from datetime import datetime
from tqdm import tqdm
import gc, os
import logging
import time
import lightgbm as lgb
from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler
from baseline.itemcf import reduce_mem
from baseline import process_data
data_path = const.RAW_DATA_FOLDER
save_path = const.OUTPUT_FOLDER

# all_click_df指的是训练集
# sample_user_nums 采样作为验证集的用户数量
def trn_val_split(all_click_df, sample_user_nums=20):
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


def get_trn_val_tst_data():

    click_trn_data = pd.read_csv(save_path + 'train_04-05.csv')
    click_trn_data = reduce_mem(click_trn_data)
    click_trn, click_val, val_ans = trn_val_split(click_trn_data)

    click_tst = pd.read_csv(save_path+'test_04-05.csv')

    return click_trn, click_val, click_tst, val_ans


# 返回多路召回列表或者单路召回
def get_recall_list(save_path, single_recall_model=None, multi_recall=False):
    if multi_recall:
        return pickle.load(open(save_path + 'final_recall_items_dict.pkl', 'rb'))

    if single_recall_model == 'i2i_itemcf':
        return pickle.load(open(save_path + 'itemcf_recall_dict.pkl', 'rb'))
    elif single_recall_model == 'i2i_emb_itemcf':
        return pickle.load(open(save_path + 'itemcf_emb_dict.pkl', 'rb'))
    elif single_recall_model == 'user_cf':
        return pickle.load(open(save_path + 'youtubednn_usercf_dict.pkl', 'rb'))
    elif single_recall_model == 'youtubednn':
        return pickle.load(open(save_path + 'youtube_u2i_dict.pkl', 'rb'))


if __name__ == '__main__':
    print("开始划分数据集......")
    click_trn, click_val, val_ans, test_data = process_data.trn_val_tst_split(data_path)
    print("开始保存训练集......")
    click_trn.to_csv(save_path + 'train_' + datetime.today().strftime('%m-%d') + '.csv', index=False, header=True)
    print("开始保存验证集......")
    click_val.to_csv(save_path + 'val_' + datetime.today().strftime('%m-%d') + '.csv', index=False, header=True)
    print("开始保存验证集答案......")
    val_ans.to_csv(save_path + 'ans_' + datetime.today().strftime('%m-%d') + '.csv', index=False, header=True)
    print("开始保存测试集......")
    test_data.to_csv(save_path + 'test_' + datetime.today().strftime('%m-%d') + '.csv', index=False, header=True)
