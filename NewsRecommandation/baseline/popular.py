import numpy as np
from process_data import load_file
import const
import pandas as pd
import numpy as np

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


def get_last_clicked(train_set):
    last_clicked = {}
    rows = train_set.shape()[0]
    for i in range(0, rows):
        user_id = train_set[i][0]
        clicked_timestamp = int(train_set[i][2])
        last_clicked.setdefault(user_id, 0)
        if clicked_timestamp > last_clicked[user_id]:
            last_clicked[user_id] = clicked_timestamp

    return last_clicked


def getArticleDict():
    articles_dict = {}
    articles_path = const.RAW_DATA_FOLDER +"articles.csv"
    for line in load_file(articles_path):
        article_info = line.split(',')
        article_id = article_info[0]
        created_at_ts = int(article_info[2])
        articles_dict[article_id] = created_at_ts

    return articles_dict



def is_recall_target(last_clicked_timestamp, art_id, articles_dict, lag_hour_max=27, lag_hour_min=3):
    # 热度文章在用户最后一次点击时刻起，前3小时~27小时内的文章
    lag_max = lag_hour_max *60 * 60 * 1000
    lag_min = lag_hour_min *60 * 60 * 1000


    if articles_dict[art_id] < (last_clicked_timestamp - lag_max):
        return  False
    if articles_dict[art_id] > (last_clicked_timestamp - lag_min):
        return False

    return True

def get_hot_items(user_item_time_dict):
    hot_items = {}
    for _, items_times in user_item_time_dict.items():
        for id_time in items_times:
            article_id = id_time[0]
            hot_items.setdefault(article_id, 0)
            hot_items[article_id] += 1

    return hot_items

def recall_hot_items(user_item_time_dict, articles_dict, topK=10):
    recall = {}
    lag_hour_max = 27
    lag_hour_min = 3
    print("热门商品统计中......")
    hot_items = get_hot_items(user_item_time_dict)
    # hot_items = [(id_1, v_1), (id_2, v_2),......]

    hot_items = sorted(hot_items.items(), key=lambda x : x[1], reverse=True)
    print("热门商品统计完成")
    for user_id, items_times in user_item_time_dict.items():
        last_clicked_timestamp = int(items_times[-1][1])
        clicked_items = get_clicked_items(items_times)
        recommend_items = []
        for id_count in hot_items:
            article_id = id_count[0]
            if article_id in clicked_items:
                continue
            if not is_recall_target(last_clicked_timestamp, article_id, articles_dict, lag_hour_max, lag_hour_min):
                continue
            recommend_items.append(article_id)

            if len(recommend_items) >= topK:
                break

        recall[user_id] = recommend_items
    print("召回完成")
    return recall




def get_clicked_items(items_times):
    clicked_items = []


    for id_time in items_times:
        clicked_items.append(id_time[0])

    return clicked_items







