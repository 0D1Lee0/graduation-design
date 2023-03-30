import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from lgb_const import DATA_PATH, SAVE_PATH


# 依次评估召回的前10, 20, 30, 40, 50个文章中的击中率
def metrics_recall(user_recall_items_dict, trn_last_click_df, topk=5):
    last_click_item_dict = dict(zip(trn_last_click_df['user_id'], trn_last_click_df['click_article_id']))
    user_num = len(user_recall_items_dict)

    for k in range(10, topk + 1, 10):
        hit_num = 0
        for user, item_list in user_recall_items_dict.items():
            # 获取前k个召回的结果
            tmp_recall_items = [x[0] for x in user_recall_items_dict[user][:k]]
            if last_click_item_dict[user] in set(tmp_recall_items):
                hit_num += 1

        hit_rate = round(hit_num * 1.0 / user_num, 5)
        print(' topk: ', k, ' : ', 'hit_num: ', hit_num, 'hit_rate: ', hit_rate, 'user_num : ', user_num)


def combine_recall_results(user_multi_recall_dict, weight_dict=None, topk=25):
    """
    多路召回合并
    :param user_multi_recall_dict: 不同策略召回的文章列表
    :param weight_dict:     不同策略召回的权重
    :param topk:            召回的文章数目
    :return:                多路召回合并后的文章列表
    """
    final_recall_items_dict = {}

    # 对每一种召回结果按照用户进行归一化，方便后面多种召回结果，相同用户的物品之间权重相加
    def norm_user_recall_items_sim(sorted_item_list):
        # 如果冷启动中没有文章或者只有一篇文章，直接返回
        # 基于规则筛选之后就没有文章了
        if len(sorted_item_list) < 2:
            return sorted_item_list

        min_sim = sorted_item_list[-1][1]
        max_sim = sorted_item_list[0][1]

        norm_sorted_item_list = []
        for item, score in sorted_item_list:
            if max_sim > 0:
                norm_score = 1.0 * (score - min_sim) / (max_sim - min_sim) if max_sim > min_sim else 1.0
            else:
                norm_score = 0.0
            norm_sorted_item_list.append((item, norm_score))

        return norm_sorted_item_list

    print('多路召回合并...')
    for method, user_recall_items in tqdm(user_multi_recall_dict.items()):
        print("\n" + method + '...')
        # 若未设置权重，则默认都为1
        if weight_dict == None:
            recall_method_weight = 1
        else:
            recall_method_weight = weight_dict[method]

        for user_id, sorted_item_list in user_recall_items.items():  # 进行归一化
            user_recall_items[user_id] = norm_user_recall_items_sim(sorted_item_list)

        for user_id, sorted_item_list in user_recall_items.items():
            final_recall_items_dict.setdefault(user_id, {})
            for item, score in sorted_item_list:
                final_recall_items_dict[user_id].setdefault(item, 0)
                final_recall_items_dict[user_id][item] += recall_method_weight * score

    final_recall_items_dict_rank = {}
    #控制最终的召回数量
    for user, recall_item_dict in final_recall_items_dict.items():
        final_recall_items_dict_rank[user] = sorted(recall_item_dict.items(), key=lambda x: x[1], reverse=True)[:topk]

    # 将多路召回后的最终结果字典保存到本地
    pickle.dump(final_recall_items_dict_rank, open(os.path.join(SAVE_PATH, 'final_recall_items_dict_40.pkl'), 'wb',),
                protocol=4)

    return final_recall_items_dict_rank


if __name__ == '__main__':
    user_multi_recall_dict = {
        'itemcf_sim_itemcf_recall': {},
        'cold_start_recall': {}
    }
    weight_dict = {
        'itemcf_sim_itemcf_recall': 1.0,
        'cold_start_recall': 0.8
    }
    """
        召回列表：{user1:[(item1, score1), (item2, score2)......]}
    """
    print("读取召回列表......")
    itemcf_recall_dict = pickle.load(open(SAVE_PATH +'itemcf_lgb_40.pkl', 'rb'))
    cold_start_dict = pickle.load(open(SAVE_PATH + 'cold_start.pkl', 'rb'))

    print("len(itemcf)=" + str(len(itemcf_recall_dict)), end='  ')
    print("len(cold_start)=" + str(len(cold_start_dict)))

    user_multi_recall_dict['itemcf_sim_itemcf_recall'] = itemcf_recall_dict
    user_multi_recall_dict['cold_start_recall'] = cold_start_dict

    print("开始合并......")
    combine_recall_results(user_multi_recall_dict, weight_dict)










