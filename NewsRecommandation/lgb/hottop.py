import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

import measurement
from lgb_const import DATA_PATH, SAVE_PATH
from utils import reduce_mem, get_article_cate_time_dict
from topic_recall import get_user_topic
from itemcf import save_recall_dict
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


def get_last_clicked(user_item_time_dict):
    last_clicked = {}

    for user, item_time_list in user_item_time_dict.items():
        last_time = item_time_list[-1][1]
        last_clicked[user] = last_time

    return last_clicked

def get_hot_items(user_item_time_dict):
    hot_items = {}
    for _, items_times in user_item_time_dict.items():
        for id_time in items_times:
            article_id = id_time[0]
            hot_items.setdefault(article_id, 0)
            hot_items[article_id] += 1

    hot_items_rank = sorted(hot_items.items(), key=lambda x : x[1], reverse=True)
    max_val = hot_items_rank[0][1]
    min_val = hot_items_rank[-1][1]
    if max_val==min_val:
        return hot_items_rank
    hot_items_rank_ = []
    for item, val in hot_items_rank:
        nor_val = (val - min_val) / (max_val - min_val)
        hot_items_rank_.append((item, nor_val))

    return hot_items_rank_

def get_article_time(article_info):
    article_time = {}
    for _, row in article_info.iterrows():
        article_id = row['article_id']
        created_at_ts = row['created_at_ts']
        article_time[article_id] = created_at_ts

    return article_time

def is_recall_target(last_clicked_timestamp, art_id, articles_dict, lag_hour_pre=24, lag_hour_next=24):
    # 热度文章在用户最后一次点击时刻起，前lag_hour_pre小时~lag_hour_next小时内的文章
    lag_pre = lag_hour_pre * 60 * 60 * 1000
    lag_next = lag_hour_next * 60 * 60 * 1000


    if articles_dict[art_id][1] < (last_clicked_timestamp - lag_pre):
        return  False
    if articles_dict[art_id][1] > (last_clicked_timestamp + lag_next):
        return False

    return True

def recall_hot_items(user, user_item_time_dict, articles_dict, hot_items):
    recall = []
    lag_hour_pre = 24
    lag_hour_next = 24

    hist_items = []
    hist_items_list = user_item_time_dict.get(user, [])
    last_click_time = hist_items_list[-1][1]
    for item, _ in hist_items_list:
        hist_items.append(item)

    for item, count in hot_items:
        if item in hist_items:
            continue

        if not is_recall_target(last_click_time, item, articles_dict, lag_hour_pre, lag_hour_next):
            continue
        recall.append((item, count))

    recall_rank = sorted(recall, key=lambda x: x[1], reverse=True)

    max_rank = recall_rank[0][1]
    min_rank = recall_rank[-1][1]
    if max_rank==min_rank:
        return recall_rank

    for i in range(0, len(recall_rank)):
        item = recall_rank[i][0]
        score = (recall_rank[i][1] - min_rank) / (max_rank - min_rank)
        recall_rank[i] = (item, score)


    return recall_rank

def timestamp_hot_items(user, user_item_time_dict, lag_hour_max, lag_hour_min):
    """
    获取用户最后一次点击时间前 lag_hour_min - lag_hour_max小时内的热门文章
    :param user: 当前用户
    :param user_item_time_dict: 所有用户点击过的文章和时间序列
    :param lag_hour_max:    最大时间间隔
    :param lag_hour_min:    最小时间间隔
    :return:
    """

    # 最后一次的点击时间
    last_click_time = user_item_time_dict[user][-1][1]


def get_top_item(all_click):
    items = {}
    for _, row in all_click.iterrows():
        article = row['click_article_id']
        items.setdefault(article, 0)
        items[article] += 1

    items_rank = sorted(items.items(), key=lambda x:x[1], reverse=True)
    return items_rank


def get_clicked_items(items_times):
    clicked_items = []


    for id_time in items_times:
        clicked_items.append(id_time[0])

    return clicked_items

def cold_start_items(user, user_item_time_dict, user_hist_type_dict, article_info, hot_items_rank,recall_item_num):
    """
    冷启动策略召回文章-热门文章推荐
    :param user:  当前用户
    :param user_item_time_dict:   用户历史点击文章的记录
    :param user_hist_type_dict:   用户历史兴趣
    :param article_info:   文章信息，包括类别，创建时间
    :param hot_items_rank: 热门文章
    :param recall_item_num:     召回的文章的数量
    :return: 冷启动策略召回的前recall_item_num个的文章的列表
    """
    cold_start_user_items = []

    # 进行文章创建时间筛选
    recall_hot_item_list = recall_hot_items(user, user_item_time_dict, article_info, hot_items_rank)

    user_hist_type = user_hist_type_dict[user]
    # item: 文章， level：被点击的次数
    for item, level in recall_hot_item_list:
        cur_art_type = article_info[item][0]
        interest = 0.0
        for type, count in user_hist_type:
            if cur_art_type == type:
                interest = count
                continue

        if interest == 0:
            continue
        val = level * interest
        cold_start_user_items.append((item, val))

    # 需要控制一下冷启动召回的数量
    cold_start_items_rank = sorted(cold_start_user_items, key=lambda x: x[1], reverse=True)[:recall_item_num]

    pickle.dump(cold_start_items_rank, open(SAVE_PATH + 'cold_start_user_items_dict.pkl', 'wb'))

    return cold_start_items_rank

def get_articles_info(articles_df):
    articles_info = {} #{art_1:(cate_id, create_time, words)}
    for _, row in articles_df.iterrows():
        art_id = row['article_id']
        cate_id = row['category_id']
        create_time = row['created_at_ts']
        words = row['words_count']
        articles_info[art_id] = (cate_id, create_time, words)

    return articles_info

def save_user_hot_items(user_hot_items):
    path = DATA_PATH + 'user_hot_items.pkl'
    pickle.dump(user_hot_items, open(path, 'wb'), protocol=4)

if __name__ == '__main__':
    click_trn_hist = pd.read_csv(DATA_PATH + 'click_trn_hist_.csv')
    click_val_hist = pd.read_csv(DATA_PATH + 'click_val_hist_.csv')
    click_tst_hist = pd.read_csv(DATA_PATH + 'click_tst_hist_.csv')

    articles_info = pd.read_csv(DATA_PATH +'articles.csv')

    all_click = click_trn_hist.append(click_val_hist)
    all_click = all_click.append(click_tst_hist)



    user_item_time = get_user_item_time(all_click)

    print("开始统计热门文章......")
    hot_items_rank = get_hot_items(user_item_time)
    #recall_items = recall_hot_items(user_item_time, article_create_time, 10)

    hot_recall_items = {}
    items_rank = get_top_item(all_click)
    trn_users = all_click['user_id'].unique()

    article_cate_time_dict = get_article_cate_time_dict(articles_info)
    user_hot_items = {}

    print("开始筛选......")
    for user, click_record in user_item_time.items():
        click_last_time = click_record[-1][1]
        click_hist_items = [item for item, time in click_record]
        recall = []
        for item, count in hot_items_rank:
            if item in click_hist_items:
                continue

            if not is_recall_target(click_last_time, item, article_cate_time_dict):
                continue
            recall.append((item, count))
        user_hot_items[user] = recall

    print("开始保存......")
    save_user_hot_items(user_hot_items)
    """recall_num = 10

    article_cate_time_dict = get_article_cate_time_dict(articles_info)
    user_hist_type_dict = get_user_topic(all_click, articles_info)
    print("开始召回......")
    for user in tqdm(trn_users):
        hot_recall_items[user] = cold_start_items(user, user_item_time, user_hist_type_dict, article_cate_time_dict,
                                                  hot_items_rank, recall_num)

    save_recall_dict(hot_recall_items, 'cold_start_rule')
    click_tst_last = pd.read_csv(DATA_PATH + 'click_tst_last_.csv')

    click_ans = {}
    for _, row in click_tst_last.iterrows():
        user_id = row['user_id']
        click_item = row['click_article_id']
        click_ans[user_id] = click_item

    print("开始分析......")
    hr = 0.0
    mrr = 0.0
    sum = 0.0
    hit = 0
    for user, real_item in click_ans.items():
        recall_items = hot_recall_items[user]
        p = 0.0

        for i in range(0, len(recall_items)):
            if real_item == recall_items[i][0]:
                p = 1.0/(i+1)
                hit += 1
                break
        sum += p

    mrr = sum/len(click_ans)
    hr = hit/len(click_ans)
    print("mrr=" + str(mrr) +"  hr=" + str(hr))"""




