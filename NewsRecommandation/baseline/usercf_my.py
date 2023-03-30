# import packages
import time, math, os
from tqdm import tqdm
import gc
import pickle
import random
from datetime import datetime
from operator import itemgetter
import numpy as np
import pandas as pd
import const
import warnings
from collections import defaultdict
from utils import reduce_mem
warnings.filterwarnings('ignore')

data_path = const.RAW_DATA_FOLDER
save_path = const.OUTPUT_FOLDER

# 根据点击时间获取文章的被点击序列   {item1: [(user1, time1), (user2, time2)..]...}
def get_item_user_time(click_df):
    item_user_time_dict = {}

    click_df = click_df.sort_values('click_timestamp')

    for index, row in click_df.iterrows():
        user_id = row['user_id']
        article_id = row['click_article_id']
        timestamp = row['click_timestamp']
        item_user_time_dict.setdefault(article_id, [])
        item_user_time_dict[article_id].append((user_id, timestamp))

    return item_user_time_dict

def usercf_sim(df):
    """
    计算用户与用户之间的相似度
    :param df: 数据集
    :return: 用户与用户的相似矩阵
    """
    item_user_time_dict = get_item_user_time(df)

    u2u_sim = {}

    user_cnt = defaultdict(int)
    for item, user_time_list in item_user_time_dict.items():
        for i, i_click_time in user_time_list:
            user_cnt[i] += 1
            u2u_sim.setdefault(i, {})
            for j, j_click_time in user_time_list:
                if i == j:
                    continue
                u2u_sim[i].setdefault(j, 0)

                u2u_sim[i][j] += 1 / math.log(len(user_time_list) + 1)

    u2u_sim_ = u2u_sim.copy()

    for i, related_user in u2u_sim.items():
        for j, wij in related_user.items():
            u2u_sim_[i][j] = wij / math.sqrt(user_cnt[i] * user_cnt[j])

    sim_path = save_path + 'usercf_u2u_sim' + '_' + datetime.today().strftime('%m-%d') + '.pkl'
    # 将得到的相似性矩阵保存到本地
    pickle.dump(u2u_sim_, open(sim_path, 'wb'))

    return u2u_sim_, sim_path

def get_user_item_list(df):
    user_item_lsit = {}
    for index, row in df.iterrows():
        user_id = row['user_id']
        article_id = row['click_article_id']
        timestamp = row['click_timestamp']
        user_item_lsit.setdefault(user_id, [])
        user_item_lsit[user_id].append((article_id, timestamp))

    return user_item_lsit


def user_based_recommend(user_id, item_user_time_dict, u2u_sim, sim_item_topk, recall_item_num, user_item_list):
    """
    基于用户的协同过滤
    :param user_id:
    :param item_user_time_dict:
    :param u2u_sim:
    :param sim_item_topk:
    :param recall_item_num:
    :param user_item_list:
    :return:
    """
    user_his_items = [id for id, _ in user_item_list[user_id]]

    user_rank = sorted(u2u_sim[user_id], reverse=True)
    item_rank = []

    for user in user_rank:
        history_items = sorted(user_item_list[user], key=lambda x:x[1], reverse=True)
        for item in history_items:
            if item in user_his_items:
                continue
            item_rank.append(item)
            if len(item_rank) >= recall_item_num:
                break
        if len(item_rank) >= recall_item_num:
            break

    return item_rank



