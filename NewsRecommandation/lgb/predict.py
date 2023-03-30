import pickle
import pandas as pd
import numpy as np
from measurement import get_mrr_hit
from lgb_const import DATA_PATH, SAVE_PATH

click_tst_last = pd.read_csv(DATA_PATH + 'click_tst_last_.csv')

itemcf_recall_25_dict = pickle.load(open(SAVE_PATH + 'item_25_hot_10/itemcf_lgb_.pkl', 'rb'))
cold_start_recall_10_dict = pickle.load(open(SAVE_PATH + 'item_25_hot_10/cold_start.pkl', 'rb'))
multi_recall_25_dict = pickle.load(open(SAVE_PATH + 'item_25_hot_10/final_recall_items_dict.pkl', 'rb'))
click_tst_dict = {}
for _, row in click_tst_last.iterrows():
    user = int(row['user_id'])
    art = int(row['click_article_id'])
    click_tst_dict[user] = art



itemcf_recall_items = {}
for user, item_list in itemcf_recall_25_dict.items():
    top_items_score = item_list[:5]
    top_items = [item for item, val in top_items_score]
    itemcf_recall_items[user] = top_items


cold_start_recall_items = {}
for user, item_list in cold_start_recall_10_dict.items():
    top_items_score = item_list[:5]
    top_items = [item for item, val in top_items_score]
    cold_start_recall_items[user] = top_items

multi_recall_items = {}
for user, item_list in multi_recall_25_dict.items():
    top_items_score = item_list[:5]
    top_items = [item for item, val in top_items_score]
    multi_recall_items[user] = top_items

method_recall_items = {
    'itemcf': itemcf_recall_items,
    'cold_start': cold_start_recall_items,
    'multi_recall': multi_recall_items
}

mrr_hit = {}

for method, user_recall_items in method_recall_items.items():
    len_users = len(click_tst_dict)
    m = 0.0
    hit = 0
    for user, art in click_tst_dict.items():
        recall_items = user_recall_items[user]
        for i in range(0, len(recall_items)):
            if recall_items[i] == art:
                m += 1.0/(i+1)
                hit += 1
                break
    mrr = m / len_users
    hr = hit / len_users
    mrr_hit[method] = [mrr, hr]

for method, rate in mrr_hit.items():
    print(method, end='  ')
    print(rate)



