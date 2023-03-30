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

max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
"""
基于物品的协同过滤
计算文章相似度：
"""
def reduce_mem(df):
    starttime = time.time()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
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
    end_mem = df.memory_usage().sum() / 1024**2
    print('-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction),time spend:{:2.2f} min'.format(end_mem,
                                                                                                           100*(start_mem-end_mem)/start_mem,
                                                                                                           (time.time()-starttime)/60))
    return df

warnings.filterwarnings('ignore')
data_path = const.RAW_DATA_FOLDER
save_path = const.OUTPUT_FOLDER

# 根据点击时间获取用户的点击文章序列   {user1: [(item1, time1), (item2, time2)..]...}
def get_user_item_time(click_df):
    click_df = click_df.sort_values('click_timestamp')

    def make_item_time_pair(df):
        return list(zip(df['click_article_id'], df['click_timestamp']))

    user_item_time_df = click_df.groupby('user_id')['click_article_id', 'click_timestamp'].apply(
        lambda x: make_item_time_pair(x)) \
        .reset_index().rename(columns={0: 'item_time_list'})
    user_item_time_dict = dict(zip(user_item_time_df['user_id'], user_item_time_df['item_time_list']))

    return user_item_time_dict

# 获取近期点击最多的文章
def get_item_topk_click(click_df, k):
    topk_click = click_df['click_article_id'].value_counts().index[:k]
    return topk_click

def normalize(i2i_sim):
    nor_sim = {}
    for i, j_w in i2i_sim.items():
        nor_sim[i] = {}
        max_wij = max(j_w.values())
        for j in j_w:
            nor_sim[i][j] = i2i_sim[i][j]/max_wij

    return i2i_sim

def itemcf_sim(df):
    """
        文章与文章之间的相似性矩阵计算
        :param df: 数据表
        :item_created_time_dict:  文章创建时间的字典
        return : 文章与文章的相似性矩阵
        思路: 基于物品的协同过滤， 在多路召回部分会加上关联规则的召回策略
    """

    user_item_time_dict = get_user_item_time(df)

    # 计算物品相似度
    i2i_sim = {}
    item_cnt = defaultdict(int) #当key不存在时，item_cnt[key]=0
    for user, item_time_list in tqdm(user_item_time_dict.items()):
        # 在基于商品的协同过滤优化的时候可以考虑时间因素
        for i, i_click_time in item_time_list:
            item_cnt[i] += 1
            i2i_sim.setdefault(i, {})
            for j, j_click_time in item_time_list:
                if i == j:
                    continue
                i2i_sim[i].setdefault(j, 0)

                i2i_sim[i][j] += 1 / math.log(len(item_time_list) + 1)

    i2i_sim_ = i2i_sim.copy()
    for i, related_items in i2i_sim.items():
        for j, wij in related_items.items():
            i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])
    #i2i_sim_ = normalize(i2i_sim_)
    sim_path = save_path + 'itemcf_i2i_sim' +'_'+ datetime.today().strftime('%m-%d')+'.pkl'
    # 将得到的相似性矩阵保存到本地
    pickle.dump(i2i_sim_, open(sim_path, 'wb'))

    return i2i_sim_, sim_path

def get_item_created_time_dict(df):

    item_created_time_dict = {}
    for index, row in df.iterrows():
        id = row['article_id']
        #cate_id = row['category_id']
        time = row['created_at_ts']
        item_created_time_dict[id] = time

    return item_created_time_dict

def itemcf_related_rule_sim(df, item_created_time_dict):
    """
    文章与文章之间的相似性矩阵计算
    :param df:  数据集
    :param item_created_time_dict:  文章创建时间的字典
    :return:
    """

    user_item_time = get_user_item_time(df)

    # 计算物品相似度
    i2i_sim = {}
    item_cnt = defaultdict(int)
    print("开始读取user_item_time_dict......")
    for user, item_time_list in tqdm(user_item_time.items()):
        # 在基于商品的协同过滤优化的时候可以考虑时间因素
        for loc1, (i, i_click_time) in enumerate(item_time_list):
            item_cnt[i] += 1
            i2i_sim.setdefault(i, {})
            for loc2, (j, j_click_time) in enumerate(item_time_list):
                if i == j:
                    continue

                # 考虑文章的正向顺序点击和反向顺序点击
                if loc2 > loc1:
                    loc_alpha = 1.0
                else:
                    loc_alpha = 0.75
                # 位置信息权重，其中的参数可以调节
                """
                    
                """
                loc_weight = loc_alpha * (0.9 ** (np.abs(loc2 - loc1) - 1))
                # 点击时间权重，其中的参数可以调节
                click_time_weight = np.exp(0.8 ** np.abs(i_click_time - j_click_time))
                # 两篇文章创建时间的权重，其中的参数可以调节
                created_time_weight = np.exp(0.9 ** np.abs(item_created_time_dict[i] - item_created_time_dict[j]))
                i2i_sim[i].setdefault(j, 0)
                # 考虑多种因素的权重计算最终的文章之间的相似度
                sim = round((loc_weight * click_time_weight * created_time_weight) / math.log(len(item_time_list) + 1), 5)
                i2i_sim[i][j] += sim

    i2i_sim_ = i2i_sim.copy()
    for i, related_items in i2i_sim.items():
        for j, wij in related_items.items():
            i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])
    #i2i_sim_ = normalize(i2i_sim_)
    # 将得到的相似性矩阵保存到本地
    path = save_path + 'itemcf_related_rule_i2i_sim.pkl'
    pickle.dump(i2i_sim_, open(path, 'wb'))

    return i2i_sim_, path





# 基于商品的召回i2i
def item_based_recommend(user_id, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num, item_topk_click):
    """
        基于文章协同过滤的召回
        :param user_id: 用户id
        :param user_item_time_dict: 字典, 根据点击时间获取用户的点击文章序列   {user1: [(item1, time1), (item2, time2)..]...}
        :param i2i_sim: 字典，文章相似性矩阵
        :param sim_item_topk: 整数， 选择与当前文章最相似的前k篇文章
        :param recall_item_num: 整数， 最后的召回文章数量
        :param item_topk_click: 列表，点击次数最多的文章列表，用户召回补全
        return: 召回的文章列表 {item1:score1, item2: score2...}
        总结: 基于物品的协同过滤， 在多路召回部分会加上关联规则的召回策略
    """

    # 获取用户历史交互的文章
    #user_hist_items = user_item_time_dict[user_id]
    user_hist_items = user_item_time_dict.get(user_id, {})
    user_hist_items_ = [user_id for user_id, _ in user_hist_items]

    item_rank = {}
    for loc, (i, click_time) in enumerate(user_hist_items):
        for j, wij in sorted(i2i_sim[i].items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]:
            if j in user_hist_items_:
                continue

            item_rank.setdefault(j, 0)
            item_rank[j] += wij

    # 不足10个，用热门商品补全
    if len(item_rank) < sim_item_topk:
        for i, item in enumerate(item_topk_click):
            if item in item_rank.items():  # 填充的item应该不在原来的列表中
                continue
            if item in user_hist_items_:
                continue
            item_rank[item] = - i - 100  # 随便给个负数就行
            if len(item_rank) == sim_item_topk:
                break
    item_sim_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)
    #item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]

    return item_sim_rank#, item_rank

# 生成提交文件
def submit(recall_df, topk=5, model_name=None):
    recall_df = recall_df.sort_values(by=['user_id', 'pred_score'])
    recall_df['rank'] = recall_df.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')

    # 判断是不是每个用户都有5篇文章及以上
    tmp = recall_df.groupby('user_id').apply(lambda x: x['rank'].max())
    assert tmp.min() >= topk

    del recall_df['pred_score']
    submit = recall_df[recall_df['rank'] <= topk].set_index(['user_id', 'rank']).unstack(-1).reset_index()

    submit.columns = [int(col) if isinstance(col, int) else col for col in submit.columns.droplevel(0)]
    # 按照提交格式定义列名
    submit = submit.rename(columns={'': 'user_id', 1: 'article_1', 2: 'article_2',
                                    3: 'article_3', 4: 'article_4', 5: 'article_5'})

    save_name = save_path + model_name + '_' + datetime.today().strftime('%m-%d') + '.csv'
    submit.to_csv(save_name, index=False, header=True)
    return save_name



def save_recall_dict(recall_dict, model):

    path = 'E:/recommend_data/aliyun/result_data/' + model + '_' + datetime.today().strftime('%m-%d') +'.pkl'
    pickle.dump(recall_dict, open(path, 'wb'))