import csv
from datetime import datetime
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

import process_data
import const
import itemcf
import collections
import error_analysis
from measurement import recall2dict, recall_precision_f

save_path = const.OUTPUT_FOLDER

# 定义常量
train_path = r'E:\recommend_data\aliyun\result_data\train_04-11.csv'
test_path = r'E:\recommend_data\aliyun\result_data\test_04-11.csv'
article_path = r'E:\recommend_data\aliyun\original_data\articles.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

train_df = itemcf.reduce_mem(train_df)
test_df = itemcf.reduce_mem(test_df)

data_columns = ["user_id", "click_article_id", "click_timestamp", "click_environment", "click_deviceGroup", "click_os",
               "click_country", "click_region", "click_referrer_type"]
train_df[data_columns] = train_df[data_columns].applymap(lambda x : int(x))
test_df[data_columns] = test_df[data_columns].applymap(lambda x : int(x))

article_columns = ['article_id', 'category_id', 'created_at_ts', 'words_count']
article_df = pd.read_csv(article_path)
article_df[article_columns] = article_df[article_columns].applymap(lambda x : int(x))
tst_users = test_df['user_id'].unique()
item_created_time_dict = itemcf.get_item_created_time_dict(article_df)


# 相似文章的数量
sim_item_topk = 25

# 召回文章数量
recall_item_num = 10

# 用户热度补全
item_topk_click = itemcf.get_item_topk_click(train_df, k=50)

# 获取 用户 - 文章 - 点击时间的字典
user_item_time_dict = itemcf.get_user_item_time(train_df)

print("--------------------itemcf--------------------")
print("开始保存 i2i 相似矩阵......")
i2i_sim, i2i_path = itemcf.itemcf_sim(train_df)
print("保存i2i相似矩阵成功")
# 定义
user_recall_items_dict = collections.defaultdict(dict)



"""for user in tqdm(train_df['user_id'].unique()):
    user_recall_items_dict[user] = itemcf.item_based_recommend(user, user_item_time_dict, i2i_sim,
                                                        sim_item_topk, recall_item_num, item_topk_click)

# 将字典的形式转换成df
user_item_score_list = []

for user, items in tqdm(user_recall_items_dict.items()):
    for item, score in items:
        user_item_score_list.append([user, item, score])

recall_df = pd.DataFrame(user_item_score_list, columns=['user_id', 'click_article_id', 'pred_score'])


tst_users = train_df['user_id'].unique()
tst_recall = recall_df[recall_df['user_id'].isin(tst_users)]"""

tst_recall = collections.defaultdict(dict)

for user in tqdm(tst_users):
    tst_recall[user] = itemcf.item_based_recommend(user, user_item_time_dict, i2i_sim,
                                                        sim_item_topk, recall_item_num, item_topk_click)

print(len(tst_recall))
itemcf.save_recall_dict(tst_recall, 'itemcf')

# 将字典的形式转换成df
user_item_score_list = []

for user, items in tqdm(tst_recall.items()):
    for item, score in items:
        user_item_score_list.append([user, item, score])

recall_df = pd.DataFrame(user_item_score_list, columns=['user_id', 'click_article_id', 'pred_score'])
"""print(recall_df.head(10))
recall_df['user_id'] = recall_df['user_id'].astype('int')
recall_df['click_article_id'] = recall_df['click_article_id'].astype('int')
recall_df['pred_score'] = recall_df['pred_score'].astype('float64')"""

print("开始生成提交文件......")
# 生成提交文件
itemcf_path = itemcf.submit(recall_df, topk=5, model_name='itemcf_baseline')

print("生成提交文件成功！")


print("--------------------itemcf_related_rule--------------------")
print("计算物品相似度......")
i2i_rr_sim, i2i_rr_path = itemcf.itemcf_related_rule_sim(train_df, item_created_time_dict)
print("物品相似度计算完成")
# 定义
print("召回......")
user_recall_items_rr_dict = collections.defaultdict(dict)

"""
for user in tqdm(train_df['user_id'].unique()):
    user_recall_items_rr_dict[user] = itemcf.item_based_recommend(user, user_item_time_dict, i2i_rr_sim,
                                                        sim_item_topk, recall_item_num, item_topk_click)

# 将字典的形式转换成df
user_item_score_rr_list = []

for user, items in tqdm(user_recall_items_rr_dict.items()):
    for item, score in items:
        user_item_score_rr_list.append([user, item, score])

recall_rr_df = pd.DataFrame(user_item_score_rr_list, columns=['user_id', 'click_article_id', 'pred_score'])


tst_users = test_df['user_id'].unique()
tst_rr_recall = recall_rr_df[recall_rr_df['user_id'].isin(tst_users)]
"""
tst_rr_recall = collections.defaultdict(dict)
for user in tqdm(tst_users):
    tst_rr_recall[user] = itemcf.item_based_recommend(user, user_item_time_dict, i2i_rr_sim,

                                                                  sim_item_topk, recall_item_num, item_topk_click)

itemcf.save_recall_dict(tst_rr_recall, 'itemcf_related_rule')

user_item_score_rr_list = []
for user, items in tqdm(tst_rr_recall.items()):
    for item, score in items:
        user_item_score_rr_list.append([user, item, score])

recall_rr_df = pd.DataFrame(user_item_score_rr_list, columns=['user_id', 'click_article_id', 'pred_score'])
print("开始生成提交文件......")
# 生成提交文件
itemcf_rr_path = itemcf.submit(recall_rr_df, topk=5, model_name='itemcf_related_rule')
print("生成提交文件成功！")



recall_dict = recall2dict(itemcf_path)
rr_recall_dict = recall2dict(itemcf_rr_path)
test_df = pd.read_csv(test_path)
test_array = pd.np.array(test_df)
test_dict = error_analysis.getTestDict(test_array)
test_df[data_columns] = test_df[data_columns].applymap(lambda x: int(x))
r1, p1, f1 = recall_precision_f(recall_dict, test_dict)
r2, p2, f2 = recall_precision_f(rr_recall_dict, test_dict)

print("itemcf:      recall="+str(r1)+"    precision="+str(p1) +"    f1="+str(f1))
print("itemcf_rr:      recall=" + str(r2) + "    precision=" + str(p2) + "    f2=" + str(f2))
