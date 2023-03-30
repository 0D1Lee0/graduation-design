import collections

import pandas as pd
from tqdm import tqdm

import const
from baseline import itemcf
import lightgbm

save_path = const.OUTPUT_FOLDER

# 定义常量
train_path = r'E:\recommend_data\aliyun\result_data\train_04-11.csv'
test_path = r'E:\recommend_data\aliyun\result_data\test_04-11.csv'
article_path = r'E:\recommend_data\aliyun\original_data\articles.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

train_df = itemcf.reduce_mem(train_df)
test_df = itemcf.reduce_mem(test_df)



def lgb_racall(click_all_his, articles_info, sim_item_topk=25):
    trn_users = click_all_his['user_id'].unique()
    item_created_time_dict = itemcf.get_item_created_time_dict(articles_info)

    item_topk_click = itemcf.get_item_topk_click(click_all_his, k=50)

    # 获取 用户 - 文章 - 点击时间的字典
    user_item_time_dict = itemcf.get_user_item_time(click_all_his)

    print("计算物品相似度......")
    i2i_rr_sim, i2i_rr_path = itemcf.itemcf_related_rule_sim(click_all_his, item_created_time_dict)
    print("物品相似度计算完成")
    # 定义
    print("召回......")
    user_recall_items_rr_dict = collections.defaultdict(dict)

    for user in tqdm(trn_users):
        user_recall_items_rr_dict[user] = itemcf.item_based_recommend(user, user_item_time_dict, i2i_rr_sim,
                                                          sim_item_topk, sim_item_topk, item_topk_click)

    #itemcf.save_recall_dict(user_recall_items_rr_dict, 'itemcf_lgb')

    return user_recall_items_rr_dict





