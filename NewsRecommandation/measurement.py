import pickle

import pandas as pd
import numpy as py
from baseline.error_analysis import getTestDict
from collections import defaultdict
def recall2dict(path):
    recall_df = pd.read_csv(path)
    recall_dict = defaultdict(list)

    for _, row in recall_df.iterrows():
        user_id = int(row['user_id'])
        recall_dict[user_id] = []
        art_1 = int(row['article_1'])
        art_2 = int(row['article_2'])
        art_3 = int(row['article_3'])
        art_4 = int(row['article_4'])
        art_5 = int(row['article_5'])
        recall_dict[user_id].append(art_1)
        recall_dict[user_id].append(art_2)
        recall_dict[user_id].append(art_3)
        recall_dict[user_id].append(art_4)
        recall_dict[user_id].append(art_5)

    return recall_dict

def test2dict(path):
    test_df = pd.read_csv(path,)

def recall_precision_f(recall_dict, test_dict):

    recall_rate = 0.0
    precision_rate = 0.0
    len_user = len(recall_dict)
    for user_id, recall_items in recall_dict.items():
        len_recall = len(recall_items)
        hit_count = 0
        if user_id not in test_dict.keys():
            continue
        real_items = test_dict[user_id]
        len_real = len(real_items)
        for item in recall_items:
            if item in real_items:
                hit_count += 1

        if len_real != 0:
            recall_rate += (hit_count * 1.0 / len_real)
        if len_recall != 0:
            precision_rate += (hit_count * 1.0 / len_recall)


    recall = recall_rate/len_user
    precision = precision_rate/len_user

    f = (recall * precision * 2)/(recall + precision)
    return recall, precision, f


def get_mrr_hit(recall_items_dict, tst_last_click):
    hit_count = 0
    score = 0.0
    for user, real_click in tst_last_click.items():
        recall_items = recall_items_dict[user]
        for i in range(0, len(recall_items)):
            if real_click == recall_items[i][0]:
                p = 1.0/(i+1)
                hit_count += 1
                score += p
                break

    len_users = len(tst_last_click)

    mrr = score / len_users
    hit = hit_count / len_users

    return mrr, hit



if __name__ == '__main__':
    """item_recall_path = r'E:/recommend_data/aliyun/result_data/itemcf_baseline_04-11.csv'
    rr_item_recall_path = r'E:/recommend_data/aliyun/result_data/itemcf_related_rule_04-11.csv'
    test_path = r'E:\recommend_data\aliyun\result_data\test_04-11.csv'
    data_columns = ["user_id", "click_article_id", "click_timestamp", "click_environment", "click_deviceGroup",
                    "click_os",
                    "click_country", "click_region", "click_referrer_type"]

    recall_dict = recall2dict(item_recall_path)
    rr_recall_dict = recall2dict(rr_item_recall_path)
    test_df = pd.read_csv(test_path)
    test_array = pd.np.array(test_df)
    test_dict = getTestDict(test_array)
    test_df[data_columns] = test_df[data_columns].applymap(lambda x: int(x))
    r1, p1, f1 = recall_precision_f(recall_dict, test_dict)
    r2, p2, f2 = recall_precision_f(rr_recall_dict, test_dict)

    print("itemcf:      recall="+str(r1)+"    precision="+str(p1) +"    f1="+str(f1))
    print("itemcf_rr:      recall=" + str(r2) + "    precision=" + str(p2) + "    f2=" + str(f2))"""
    """item_recall_items = pickle.load(open('E:/recommend_data/aliyun/lgb/result/itemcf_lgb_.pkl', 'rb'))
    recall_items = {}
    for user, recall_list in item_recall_items.items():

        recall = recall_list[:5]
        recall_items[user] = recall

    click_tst_last = pd.read_csv('E:/recommend_data/aliyun/lgb/data/click_trn_last_.csv')
    real_clicks = {}
    for _, row in click_tst_last.iterrows():
        user = row['user_id']
        real_click = row['click_article_id']
        real_clicks[user] = real_click

    mrr, hr = get_mrr_hit(recall_items, real_clicks)
    print("mrr="+str(mrr) + "  hr="+str(hr))"""

    predict_df = pd.read_csv('E:/recommend_data/aliyun/lgb/result/lgb_ranker_.csv')
    real_df = pd.read_csv('E:/recommend_data/aliyun/lgb/data/click_tst_last_.csv')

    click_hist = pd.read_csv('E:/recommend_data/aliyun/lgb/data/click_tst_hist_.csv')

    hot_users = 0
    users = click_hist['user_id'].unique()


    predict_user_items = {}
    real_user_item = {}

    for _, row in predict_df.iterrows():
        user = int(row['user_id'])
        predict_user_items.setdefault(user, [])
        art_1 = int(row['article_1'])
        art_2 = int(row['article_2'])
        art_3 = int(row['article_3'])
        art_4 = int(row['article_4'])
        art_5 = int(row['article_5'])
        predict_user_items[user].append(art_1)
        predict_user_items[user].append(art_2)
        predict_user_items[user].append(art_3)
        predict_user_items[user].append(art_4)
        predict_user_items[user].append(art_5)

    for _, row in real_df.iterrows():
        user = int(row['user_id'])
        article = int(row['click_article_id'])
        real_user_item[user] = article
    score = 0.0
    hit = 0.0
    for user, article in real_user_item.items():
        predict_items = predict_user_items.get(user, [])
        for i in range(0, len(predict_items)):
            if predict_items[i] == article:
                p = 1.0/(i+1)
                score += p
                hit += 1
                break

    print(len(real_user_item))
    print(len(predict_user_items))

    mrr = score / len(real_user_item)
    hr = hit / len(real_user_item)
    mrr_ = score / len(predict_user_items)
    hr_ = hit / len(predict_user_items)

    print("mrr=" + str(mrr) + "  hr=" + str(hr))
    print("mrr_=" + str(mrr_) + "  hr_=" + str(hr_))


    click_tst_hist = pd.read_csv('E:/recommend_data/aliyun/lgb/data/click_tst_hist_.csv')
    tst_users = click_tst_hist['user_id'].unique()
    last_users = real_df['user_id'].unique()

    print(len(tst_users))
    print(len(last_users))