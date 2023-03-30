import pickle

from tqdm import tqdm

import const
import numpy as np
import pandas as pd
import error_analysis
import itemcf
import process_data
from datetime import datetime
from collections import defaultdict

data_path = const.RAW_DATA_FOLDER
save_path = const.OUTPUT_FOLDER

train_df, test_df = process_data.divide_data(const.RAW_DATA_FOLDER)
# train_df, test_df = process_data.process_data_df(data_path)
print(train_df.dtypes)
test_df = test_df.astype(pd.Int64Dtype())

print("开始保存训练集和测试集......")
train_name = const.OUTPUT_FOLDER + "train" + '_' + datetime.today().strftime('%m-%d') + '.csv'
test_name = const.OUTPUT_FOLDER + "test" + '_' + datetime.today().strftime('%m-%d') + '.csv'
train_df.to_csv(train_name, index=False, header=True)
test_df.to_csv(test_name, index=False, header=True)
print("保存训练集和测试集成功")


print("开始保存i2i相似矩阵......")
article_path = data_path + 'articles.csv'
article_df = pd.read_csv(article_path)
# print(article_df.tail(5))
columns = ['article_id', 'category_id', 'created_at_ts', 'words_count']
article_df[columns] = article_df[columns].applymap(lambda x : int(x))

item_created_time_dict = itemcf.get_item_created_time_dict(article_df)

i2i_sim, sim_path = itemcf.itemcf_related_rule_sim(train_df, item_created_time_dict)
print("保存i2i相似矩阵成功")
# 定义
user_recall_items_dict = defaultdict(dict)

# 获取 用户 - 文章 - 点击时间的字典
user_item_time_dict = itemcf.get_user_item_time(train_df)

# 去取文章相似度
# i2i_sim = pickle.load(open(sim_path, 'rb'))


# 相似文章的数量
sim_item_topk = 10

# 召回文章数量
recall_item_num = 10

# 用户热度补全
item_topk_click = itemcf.get_item_topk_click(train_df, k=50)

print("len(i2i_sim)="+str(len(i2i_sim)))

for user in tqdm(train_df['user_id'].unique()):
    user_recall_items_dict[user] = itemcf.item_based_recommend(user, user_item_time_dict, i2i_sim,
                                                        sim_item_topk, recall_item_num, item_topk_click)

# 将字典的形式转换成df
user_item_score_list = []

for user, items in tqdm(user_recall_items_dict.items()):
    for item, score in items:
        user_item_score_list.append([user, item, score])

recall_df = pd.DataFrame(user_item_score_list, columns=['user_id', 'click_article_id', 'pred_score'])


tst_users = test_df['user_id'].unique()
tst_recall = recall_df[recall_df['user_id'].isin(tst_users)]

print("开始生成提交文件......")
# 生成提交文件
save_name = itemcf.submit(tst_recall, topk=5, model_name='itemcf_related_rule')

print("生成提交文件成功！")

print("开始分析误差......")
test_array = np.array(test_df)
test_dict = error_analysis.getTestDict(test_array)

predeicted_dict = error_analysis.getPredictedDict(save_name)

scores_dict = error_analysis.getScore(predeicted_dict, test_dict, 2)

scores_df = pd.DataFrame.from_dict(scores_dict, orient='index', columns=['score'])
scores_df = scores_df.reset_index()
scores_df = scores_df.rename(columns={'index' : 'user_id'})

scores_name = const.OUTPUT_FOLDER + "error_analysis_itemcf_rr" + '_' + datetime.today().strftime('%m-%d') + '.csv'
scores_df.to_csv(scores_name, index=False, header=True)

score_dict_length = 0
hill_count = 0

for line in process_data.load_file(scores_name):
    score_dict_length = score_dict_length + 1
    score = line.split(',')[1]
    if float(score) > 0:
        hill_count = hill_count + 1

print("The precent of hill is %.2f" % (1.0*hill_count/score_dict_length))