import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import gc, os
import time
from datetime import datetime
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
import const
from baseline import error_analysis
from baseline.error_analysis import getTestDict

import measurement

import warnings

from lgb.lgb_const import DATA_PATH, SAVE_PATH

warnings.filterwarnings('ignore')




def submit(recall_df, topk=5, model_name=None):
    recall_df = recall_df.sort_values(by=['user_id', 'pred_score'])
    recall_df['rank'] = recall_df.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')

    # 判断是不是每个用户都有5篇文章及以上
    tmp = recall_df.groupby('user_id').apply(lambda x: x['rank'].max())
    assert tmp.min() >= topk

    del recall_df['pred_score']
    submit = recall_df[recall_df['rank'] <= topk].set_index(['user_id', 'rank']).unstack(-1).reset_index()

    submit.columns = [int(col) if isinstance(col, int) else col for col in submit.columns.droplevel(0)]
    # 按照提交格式定义列名
    submit = submit.rename(columns={'': 'user_id', 1: 'article_1', 2: 'article_2',
                                    3: 'article_3', 4: 'article_4', 5: 'article_5'})

    save_name = SAVE_PATH  + model_name  + '.csv'
    submit.to_csv(save_name, index=False, header=True)
    return save_name

# 排序结果归一化
def norm_sim(sim_df, weight=0.0):
    # print(sim_df.head())
    min_sim = sim_df.min()
    max_sim = sim_df.max()
    if max_sim == min_sim:
        sim_df = sim_df.apply(lambda sim: 1.0)
    else:
        sim_df = sim_df.apply(lambda sim: 1.0 * (sim - min_sim) / (max_sim - min_sim))

    sim_df = sim_df.apply(lambda sim: sim + weight)  # plus one
    return sim_df

if __name__ == '__main__':
    # 重新读取数据的时候，发现click_article_id是一个浮点数，所以将其转换成int类型
    trn_user_item_feats_df = pd.read_csv(SAVE_PATH + 'trn_user_item_feats_df_.csv')
    trn_user_item_feats_df['click_article_id'] = trn_user_item_feats_df['click_article_id'].astype('int64')

    val_user_item_feats_df = pd.read_csv(SAVE_PATH  + 'val_user_item_feats_df_.csv')
    val_user_item_feats_df['click_article_id'] = val_user_item_feats_df['click_article_id'].astype('int64')


    tst_user_item_feats_df = pd.read_csv(SAVE_PATH  + 'tst_user_item_feats_df_.csv')
    tst_user_item_feats_df['click_article_id'] = tst_user_item_feats_df['click_article_id'].astype('int64')

    # 做特征的时候为了方便，给测试集也打上了一个无效的标签，这里直接删掉就行
    del tst_user_item_feats_df['label']

    # 防止中间出错之后重新读取数据
    trn_user_item_feats_df_rank_model = trn_user_item_feats_df.copy()

    val_user_item_feats_df_rank_model = val_user_item_feats_df.copy()

    tst_user_item_feats_df_rank_model = tst_user_item_feats_df.copy()

    print("len_trn=" + str(len(trn_user_item_feats_df_rank_model)), end='  ')
    print("len_val=" + str(len(val_user_item_feats_df_rank_model)), end='  ')
    print("len_tst=" + str(len(tst_user_item_feats_df_rank_model)))
    # 定义特征列
    lgb_cols = ['sim0', 'time_diff0', 'word_diff0', 'sim_max', 'sim_min', 'sim_sum',
                'sim_mean', 'score', 'click_size', 'time_diff_mean', 'active_level',
                'click_environment', 'click_deviceGroup', 'click_os', 'click_country',
                'click_region', 'click_referrer_type', 'user_time_hob1', 'user_time_hob2',
                'words_hbo', 'category_id', 'created_at_ts', 'words_count']

    # 排序模型分组
    trn_user_item_feats_df_rank_model.sort_values(by=['user_id'], inplace=True)
    g_train = trn_user_item_feats_df_rank_model.groupby(['user_id'], as_index=False).count()["label"].values

    val_user_item_feats_df_rank_model.sort_values(by=['user_id'], inplace=True)
    g_val = val_user_item_feats_df_rank_model.groupby(['user_id'], as_index=False).count()["label"].values

    # 排序模型定义
    lgb_ranker = lgb.LGBMRanker(boosting_type='gbdt', num_leaves=15, reg_alpha=0.0, reg_lambda=1,
                                max_depth=4, n_estimators=1000, subsample=0.8, colsample_bytree=0.7, subsample_freq=1,
                                learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=16)
    lgb_ranker.fit(trn_user_item_feats_df_rank_model[lgb_cols], trn_user_item_feats_df_rank_model['label'],
                   group=g_train,
                   eval_set=[(val_user_item_feats_df_rank_model[lgb_cols], val_user_item_feats_df_rank_model['label'])],
                   eval_group=[g_val], eval_at=[1, 2, 3, 4, 5], eval_metric=['ndcg', ], early_stopping_rounds=50, )
    # 模型预测
    tst_user_item_feats_df['pred_score'] = lgb_ranker.predict(tst_user_item_feats_df[lgb_cols],
                                                              num_iteration=lgb_ranker.best_iteration_)

    # 将这里的排序结果保存一份，用户后面的模型融合
    tst_user_item_feats_df[['user_id', 'click_article_id', 'pred_score']].to_csv(SAVE_PATH  + 'lgb_ranker_score.csv',
                                                                                 index=False)

    # 预测结果重新排序, 及生成提交结果
    rank_results = tst_user_item_feats_df[['user_id', 'click_article_id', 'pred_score']]
    rank_results['click_article_id'] = rank_results['click_article_id'].astype(int)
    save_name = submit(rank_results, topk=5, model_name='lgb_ranker_')
    recall_dict = measurement.recall2dict(save_name)
    test_path = DATA_PATH + 'click_tst_last_.csv'
    test_df = pd.read_csv(test_path)
    test_array = np.array(test_df)
    test_dict = error_analysis.getTestDict(test_array)

    predicted_dict = error_analysis.getPredictedDict(save_name)

    scores_dict = error_analysis.getScore(predicted_dict, test_dict, 2)

    scores_df = pd.DataFrame.from_dict(scores_dict, orient='index', columns=['score'])
    scores_df = scores_df.reset_index()
    scores_df = scores_df.rename(columns={'index': 'user_id'})

    data_columns = ["user_id", "click_article_id", "click_timestamp", "click_environment", "click_deviceGroup",
                    "click_os",
                    "click_country", "click_region", "click_referrer_type"]






