import const
import popular
import numpy as np
import pandas as pd
from process_data import load_file
import error_analysis
from datetime import datetime
train_path = r'E:\recommend_data\aliyun\result_data\train_04-01.csv'
test_path = r'E:\recommend_data\aliyun\result_data\test_04-01.csv'
article_path = r'E:\recommend_data\aliyun\original_data\articles.csv'

def build_train_dict(train_set_path):

    train_set = {}
    for line in load_file(train_set_path):
        info = line.split(',')
        user_id = info[0]
        item_time = (info[1], info[2])
        train_set.setdefault(user_id, [])
        train_set[user_id].append(item_time)

    return train_set

def build_test_dict(test_set_path):
    test_dict = {}

    for line in load_file(test_set_path):
        info = line.split(',')
        user_id = info[0]
        article_id = info[1]
        test_dict.setdefault(user_id, [])
        test_dict[user_id].append(article_id)
    return test_dict

train_set = build_train_dict(train_path)
test_set = build_test_dict(test_path)

article_dict = popular.getArticleDict()

print("开始召回......")
recall = popular.recall_hot_items(train_set, article_dict, 5)
print("召回成功！")
print("开始分析......")
scores_dict = error_analysis.getScore(recall, test_set)

scores_df = pd.DataFrame.from_dict(scores_dict, orient='index', columns=['score'])
scores_df = scores_df.reset_index()
scores_df = scores_df.rename(columns={'index': 'user_id'})
print("开始保存......")
scores_name = const.OUTPUT_FOLDER + "error_analysis_popular" + '_' + datetime.today().strftime('%m-%d') + '.csv'
scores_df.to_csv(scores_name, index=False, header=True)
print("保存成功！")
score_dict_length = 0
hill_count = 0

for line in load_file(scores_name):
    score_dict_length = score_dict_length + 1
    score = line.split(',')[1]
    if float(score) > 0:
        hill_count = hill_count + 1

print("The precent of hill is %.2f" % (1.0*hill_count/score_dict_length))
