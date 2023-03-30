import pickle

import numpy as np
import pandas as pd
from lgb_const import DATA_PATH, SAVE_PATH
from collections import defaultdict
def get_user_topic(all_click, articles_info):
    all_click = all_click.merge(articles_info, left_on='click_article_id', right_on='article_id')
    users_topics = {}
    for _, row in all_click.iterrows():
        user_id = int(row['user_id'])
        cate_id = int(row['category_id'])
        users_topics.setdefault(user_id, {})
        users_topics[user_id].setdefault(cate_id, 0)
        users_topics[user_id][cate_id] += 1

    for user, series in users_topics.items():
        max_val = max(series.values())
        min_val = min(series.values())

        if max_val == min_val:
            for cate in series.keys():
                series[cate] = 1
        else:
            for cate in series.keys():
                series[cate] = (series[cate] - min_val) / (max_val - min_val)



    """for user, topics in users_topics.items():
        sort_topic = sorted(topics.items(), key=lambda x: x[1], reverse=True)
        user_topics[user] = sort_topic

    normalize(user_topics)"""

    return users_topics

def normalize(users_topics):
    for user, topics in users_topics.items():
        max_val = max(topics, key=lambda x: x[1])[1]
        min_Val = min(topics, key=lambda x: x[1])[1]
        if max_val == min_Val:
            for i in range(0, len(topics)):
                cate_id, _ = topics[i]
                topics[i] = (cate_id, 1)
        else:
            for i in range(0, len(topics)):
                cate_id, count = topics[i]
                val = (count - min_Val) / (max_val - min_Val) + 0.05
                topics[i] = (cate_id, val)


def save_topic_dict(user_topics):
    path = SAVE_PATH + 'user_topics.pkl'
    pickle.dump(user_topics, open(path, 'wb'), protocol=4)

if __name__ == '__main__':
    click_trn_hist = pd.read_csv(DATA_PATH + 'click_trn_hist_.csv')
    click_val_hist = pd.read_csv(DATA_PATH + 'click_val_hist_.csv')
    click_tst_hist = pd.read_csv(DATA_PATH + 'click_tst_hist_.csv')

    articles_info = pd.read_csv(DATA_PATH + 'articles.csv')

    all_click = click_trn_hist.append(click_val_hist)
    all_click = all_click.append(click_tst_hist)

    print("开始统计......")
    user_topics = get_user_topic(all_click, articles_info)
    print("开始保存......")
    save_topic_dict(user_topics)

    user_topics = pickle.load(open(SAVE_PATH + 'user_topics.pkl', 'rb'))

    i = 0
    for user, t_c in user_topics.items():
        print(user, end= "  ")
        print(t_c[-1], end="  ")
        print(len(t_c))
        i+=1
        if i==20:
            break



