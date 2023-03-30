import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

import measurement
from lgb_const import DATA_PATH, SAVE_PATH
from utils import reduce_mem, get_article_cate_time_dict
from topic_recall import get_user_topic


def get_hot_items(user_click_items):
    hot_items = {}
    for user, series in user_click_items.items():
        for item, time in series:
            hot_items.setdefault(item, 0)
            hot_items[item] += 1

    hot_items_ = []

    for item, count in hot_items.items():
        if count < 10:
            continue
        hot_items_.append((item, count))

    hot_items_rank = sorted(hot_items_, key=lambda x: x[1], reverse=True)

    max_val = hot_items_rank[0][1]
    min_val = hot_items_rank[-1][1]

    hot_items_rank_ = []

    for item, val in hot_items_rank:
        nor_val = (val - min_val) / (max_val - min_val)
        hot_items_rank_.append((item, nor_val))

    return hot_items_rank_

def get_user_click_items(all_click):
    i=0
    user_click_items = {}
    for _, row in all_click.iterrows():
        user = int(row['user_id'])
        item = int(row['click_article_id'])
        time = int(row['click_timestamp'])
        user_click_items.setdefault(user, [])
        user_click_items[user].append((item, time))

    user_click_items_ = {}
    for user, series in user_click_items.items():
        rank = sorted(series, key=lambda x:x[1])
        user_click_items_[user] = rank

    return user_click_items_

def get_article_info(article_df):
    article_info = {}

    for _, info in article_df.iterrows():
        id = int(info['article_id'])
        category = int(info['category_id'])
        time = int(info['created_at_ts'])
        words = int(info['words_count'])
        article_info[id] = (category, time, words)

    return article_info


def is_recall_target(last_click_time, click_items, article_info, item, low_hours=27, high_hours=3):
    """
    判断热门商品是否符合条件
    :param last_click_time: 用户最后一次的点击时间
    :param click_items: 用户的历史点击文章
    :param article_info: 文章的信息
    :param item: 当前文章
    :param low_hours: 最后一次点击前low_hour小时
    :param high_hours: 最后一次点击后high_hour小时
    :return:
    """
    if item in click_items:
        return False

    create_time = article_info[item][1]
    low = low_hours * 60 * 60 * 1000
    high = high_hours * 60 * 60 * 1000
    if create_time < (last_click_time - low):
        return False
    if create_time > (last_click_time + high):
        return False
    return True

def cold_start(user, user_click_items, article_info, user_topics, hot_items, recall_num):
    """

    :param user:
    :param user_click_items:
    :param article_info:
    :param user_topics:
    :param hot_items:
    :param recall_num:
    :return:
    """
    click_hist_items = [item for item, time in user_click_items[user]]
    topics = user_topics[user]
    topic_item = [t for t, _ in topics.items()]
    last_click_time = user_click_items[user][-1][1]
    target_items = []
    for item, level in hot_items:
        if not is_recall_target(last_click_time, click_hist_items, article_info, item):
            continue
        cate = article_info[item][0]
        topic_factor = 0.8
        hot_factor = 0.8
        interest = topics.get(cate, -0.2)
        val = np.exp(hot_factor ** level) * np.exp(topic_factor ** interest)
        target_items.append((item, val))


    recall_items = sorted(target_items, key=lambda x:x[1], reverse=True)[:recall_num]
    if len(recall_items) == 0:
        for item, level in hot_items:
            if item in click_hist_items:
                continue
            recall_items.append((item, level))
            if len(recall_items) == recall_num:
                break

    max_val = recall_items[0][1]
    min_val = recall_items[-1][1]

    if max_val == min_val:
        return recall_items

    recall_items_ = []
    for item, val in recall_items:
        nor_val = (val - min_val) / (max_val - min_val)
        recall_items_.append((item, nor_val))
    return recall_items_

def save_user_target_items(user_target_items):
    path = SAVE_PATH + 'user_target_items' + '.pkl'
    pickle.dump(user_target_items, open(path, 'wb'), protocol=4)

def save_recall_dict(recall_item, recall_num):
    path = SAVE_PATH + 'cold_start_' + str(recall_num) + '.pkl'
    pickle.dump(recall_item, open(path, 'wb'), protocol=4)

if __name__ == '__main__':
    print("开始读取数据......")
    click_trn_hist = pd.read_csv(DATA_PATH + 'click_trn_hist_.csv')
    click_val_hist = pd.read_csv(DATA_PATH + 'click_val_hist_.csv')
    click_tst_hist = pd.read_csv(DATA_PATH + 'click_tst_hist_.csv')

    articles_df = pd.read_csv(DATA_PATH + 'articles.csv')
    articles_df = reduce_mem(articles_df)
    all_click = click_trn_hist.append(click_val_hist)
    all_click = all_click.append(click_tst_hist)



    articles_info = get_article_info(articles_df)

    users_topics = get_user_topic(all_click, articles_df)

    user_click_items = get_user_click_items(all_click)
    all_click = reduce_mem(all_click)
    print("统计热门文章......")
    hot_items = get_hot_items(user_click_items)
    users = all_click['user_id'].unique()


    recall_num = 20
    recall_items = {}
    print("\n开始召回......")
    for user in tqdm(users):
        recall_items[user] = cold_start(user, user_click_items, articles_info, users_topics, hot_items, recall_num)

    print("\n保存召回文章......")
    save_recall_dict(recall_items, recall_num)