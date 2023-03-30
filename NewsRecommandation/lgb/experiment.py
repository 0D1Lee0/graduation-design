import os

import numpy as np
import pandas as pd
import pickle

from tqdm import tqdm

import lightgbm as lgb
from lgb.feature import create_feature
from lgb.lightgbm_rank import submit
from wordvec import recall_dict_2_df, get_user_recall_item_label_df, get_embedding, make_tuple_func
from lgb_const import DATA_PATH, SAVE_PATH

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
    pickle.dump(final_recall_items_dict_rank, open(SAVE_PATH + 'final_recall_items_dict_' + str(topk) + '.pkl', 'wb',),
                protocol=4)

    return final_recall_items_dict_rank

def get_label(click_trn_hist, click_val_hist, click_tst_hist, click_trn_last,
              click_val_last,recall_list_dict, article_info_df, model):
    recall_list_df = recall_dict_2_df(recall_list_dict)
    trn_user_item_label_df, val_user_item_label_df, tst_user_item_label_df = get_user_recall_item_label_df(
        click_trn_hist,
        click_val_hist,
        click_tst_hist,
        click_trn_last,
        click_val_last,
        recall_list_df)
    print("开始保存标签......")
    trn_label_path = SAVE_PATH + 'trn_user_item_label_' + model + '.csv'
    val_label_path = SAVE_PATH + 'val_user_item_label_' + model + '.csv'
    tst_label_path = SAVE_PATH + 'tst_user_item_label_' + model + '.csv'
    trn_user_item_label_df.to_csv(trn_label_path, index=False, header=True)
    val_user_item_label_df.to_csv(val_label_path, index=False, header=True)
    tst_user_item_label_df.to_csv(tst_label_path, index=False, header=True)

    trn_user_item_label_tuples = trn_user_item_label_df.groupby('user_id').apply(make_tuple_func).reset_index()
    trn_user_item_label_tuples_dict = dict(zip(trn_user_item_label_tuples['user_id'], trn_user_item_label_tuples[0]))

    val_user_item_label_tuples = val_user_item_label_df.groupby('user_id').apply(make_tuple_func).reset_index()
    val_user_item_label_tuples_dict = dict(
        zip(val_user_item_label_tuples['user_id'], val_user_item_label_tuples[0]))

    tst_user_item_label_tuples = tst_user_item_label_df.groupby('user_id').apply(make_tuple_func).reset_index()
    tst_user_item_label_tuples_dict = dict(zip(tst_user_item_label_tuples['user_id'], tst_user_item_label_tuples[0]))

    click_all_hist = click_trn_hist.append(click_val_hist)
    click_all_hist = click_all_hist.append(click_tst_hist)

    item_w2v_emb_dict = get_embedding(SAVE_PATH, click_all_hist)

    trn_user_item_feats_df = create_feature(trn_user_item_label_tuples_dict.keys(), trn_user_item_label_tuples_dict, \
                                            click_trn_hist, article_info_df, item_w2v_emb_dict)

    val_user_item_feats_df = create_feature(val_user_item_label_tuples_dict.keys(), val_user_item_label_tuples_dict, \
                                            click_val_hist, article_info_df, item_w2v_emb_dict)

    tst_user_item_feats_df = create_feature(tst_user_item_label_tuples_dict.keys(), tst_user_item_label_tuples_dict, \
                                            click_tst_hist, article_info_df, item_w2v_emb_dict)

    # 保存一份
    trn_user_item_feats_df.to_csv(SAVE_PATH + 'trn_user_item_feats_df' + model + '.csv', index=False)

    val_user_item_feats_df.to_csv(SAVE_PATH + 'val_user_item_feats_df' + model + '.csv', index=False)

    tst_user_item_feats_df.to_csv(SAVE_PATH + 'tst_user_item_feats_df' + model + '.csv', index=False)

    user_info = pd.read_csv(DATA_PATH + 'user_info_.csv')
    print("开始读取特征......")
    trn_user_item_feats_df = pd.read_csv(SAVE_PATH + 'trn_user_item_feats_df' + model + '.csv')
    tst_user_item_feats_df = pd.read_csv(SAVE_PATH + 'tst_user_item_feats_df' + model + '.csv')
    val_user_item_feats_df = pd.read_csv(SAVE_PATH + 'val_user_item_feats_df' + model + '.csv')

    trn_user_item_feats_df = trn_user_item_feats_df.merge(user_info, on='user_id', how='left')
    tst_user_item_feats_df = tst_user_item_feats_df.merge(user_info, on='user_id', how='left')
    val_user_item_feats_df = val_user_item_feats_df.merge(user_info, on='user_id', how='left')
    # 拼上文章特征
    trn_user_item_feats_df = trn_user_item_feats_df.merge(article_info_df, left_on='click_article_id',
                                                          right_on='article_id')

    val_user_item_feats_df = val_user_item_feats_df.merge(article_info_df, left_on='click_article_id',
                                                          right_on='article_id')
    tst_user_item_feats_df = tst_user_item_feats_df.merge(article_info_df, left_on='click_article_id',
                                                          right_on='article_id')

    print("开始分析是否符合兴趣......")
    trn_user_item_feats_df['is_cat_hab'] = trn_user_item_feats_df.apply(
        lambda x: 1 if str(x.category_id) in x.cate_list[1:-1].split(',') else 0, axis=1)

    tst_user_item_feats_df['is_cat_hab'] = tst_user_item_feats_df.apply(
        lambda x: 1 if str(x.category_id) in x.cate_list[1:-1].split(',') else 0, axis=1)

    val_user_item_feats_df['is_cat_hab'] = val_user_item_feats_df.apply(
        lambda x: 1 if str(x.category_id) in x.cate_list[1:-1].split(',') else 0, axis=1)

    # 线下验证
    del trn_user_item_feats_df['cate_list']

    if val_user_item_feats_df is not None:
        del val_user_item_feats_df['cate_list']
    else:
        val_user_item_feats_df = None

    del tst_user_item_feats_df['cate_list']

    del trn_user_item_feats_df['article_id']

    if val_user_item_feats_df is not None:
        del val_user_item_feats_df['article_id']
    else:
        val_user_item_feats_df = None

    del tst_user_item_feats_df['article_id']

    # 训练验证特征
    trn_path = SAVE_PATH + 'trn_user_item_feats_df_' + model + '.csv'
    val_path = SAVE_PATH + 'val_user_item_feats_df_' + model + '.csv'
    tst_path = SAVE_PATH + 'tst_user_item_feats_df_' + model + '.csv'
    trn_user_item_feats_df.to_csv(trn_path, index=False)
    val_user_item_feats_df.to_csv(val_path, index=False)
    tst_user_item_feats_df.to_csv(tst_path, index=False)

    return trn_path, val_path, tst_path


def lgb_rank(model):
    trn_user_item_feats_df = pd.read_csv(SAVE_PATH + 'trn_user_item_feats_df_' + model + '.csv')
    trn_user_item_feats_df['click_article_id'] = trn_user_item_feats_df['click_article_id'].astype('int64')

    val_user_item_feats_df = pd.read_csv(SAVE_PATH + 'val_user_item_feats_df_' + model + '.csv')
    val_user_item_feats_df['click_article_id'] = val_user_item_feats_df['click_article_id'].astype('int64')

    tst_user_item_feats_df = pd.read_csv(SAVE_PATH + 'tst_user_item_feats_df_' + model + '.csv')
    tst_user_item_feats_df['click_article_id'] = tst_user_item_feats_df['click_article_id'].astype('int64')

    del tst_user_item_feats_df['label']

    # 防止中间出错之后重新读取数据
    trn_user_item_feats_df_rank_model = trn_user_item_feats_df.copy()

    val_user_item_feats_df_rank_model = val_user_item_feats_df.copy()

    tst_user_item_feats_df_rank_model = tst_user_item_feats_df.copy()

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
    tst_user_item_feats_df[['user_id', 'click_article_id', 'pred_score']].to_csv(SAVE_PATH + 'lgb_ranker_score' + model + '.csv',
                                                                                 index=False)

    # 预测结果重新排序, 及生成提交结果
    rank_results = tst_user_item_feats_df[['user_id', 'click_article_id', 'pred_score']]
    rank_results['click_article_id'] = rank_results['click_article_id'].astype(int)
    save_name = submit(rank_results, topk=5, model_name='lgb_ranker_' + model)

def recommend(click_trn_hist, click_val_hist, click_tst_hist, click_trn_last,
              click_val_last,recall_list_dict, article_info_df, model):
    get_label(click_trn_hist, click_val_hist, click_tst_hist, click_trn_last,
              click_val_last,recall_list_dict, article_info_df, model)
    lgb_rank(model)

if __name__ == "__main__":
    click_trn_hist = pd.read_csv(DATA_PATH + 'click_trn_hist_.csv')
    click_val_hist = pd.read_csv(DATA_PATH + 'click_val_hist_.csv')
    click_tst_hist = pd.read_csv(DATA_PATH + 'click_tst_hist_.csv')

    click_trn_last = pd.read_csv(DATA_PATH + 'click_trn_last_.csv')
    click_val_last = pd.read_csv(DATA_PATH + 'click_val_last_.csv')

    all_click = click_trn_hist.append(click_val_hist)
    all_click = all_click.append(click_tst_hist)

    article_info = pd.read_csv(DATA_PATH + 'articles.csv')

    itemcf_recall = pickle.load(open(SAVE_PATH + "itemcf_lgb_40.pkl", 'rb'))
    cold_start_recall = pickle.load(open(SAVE_PATH + "cold_start_20.pkl", 'rb'))

    itemcf_recall_10 = {}
    itemcf_recall_25 = {}
    itemcf_recall_40 = itemcf_recall

    cold_start_recall_5 = {}
    cold_start_recall_10 = {}
    cold_start_recall_20 = cold_start_recall

    for user, items in itemcf_recall.items():
        itemcf_recall_10[user] = items[:10]
        itemcf_recall_25[user] = items[:25]

    for user, items in cold_start_recall.items():
        cold_start_recall_5[user] = items[:5]
        cold_start_recall_10[user] = items[:10]

    itemcf_path_10 = SAVE_PATH + 'itemcf_10.pkl'
    itemcf_path_25 = SAVE_PATH + 'itemcf_25.pkl'

    cold_start_path_5 = SAVE_PATH + 'cold_start_5.pkl'
    cold_start_path_10 = SAVE_PATH + 'cold_start_10.pkl'

    pickle.dump(itemcf_recall_10, open(itemcf_path_10, 'wb'), protocol=4)
    pickle.dump(itemcf_recall_25, open(itemcf_path_25, 'wb'), protocol=4)

    pickle.dump(cold_start_recall_5, open(cold_start_path_5, 'wb'), protocol=4)
    pickle.dump(cold_start_recall_10, open(cold_start_path_10, 'wb'), protocol=4)

    itemcf_model_10 = 'itemcf_10'
    itemcf_model_40 = 'itemcf_40'

    cold_start_model_5 = 'cold_start_5'
    cold_start_model_10 = 'cold_start_10'
    cold_start_model_20 = 'cold_start_20'

    print("itemcf......")

    print("itemcf_10......")
    recommend(click_trn_hist, click_val_hist, click_tst_hist, click_trn_last,
              click_val_last, itemcf_recall_10, article_info, itemcf_model_10)
    print("itemcf_40......")
    recommend(click_trn_hist, click_val_hist, click_tst_hist, click_trn_last,
              click_val_last, itemcf_recall_40, article_info, itemcf_model_40)

    print("cold_start......")
    print("cold_start_5......")
    recommend(click_trn_hist, click_val_hist, click_tst_hist, click_trn_last,
              click_val_last, cold_start_recall_5, article_info, cold_start_model_5)
    print("cold_start_10......")
    recommend(click_trn_hist, click_val_hist, click_tst_hist, click_trn_last,
              click_val_last, cold_start_recall_10, article_info, cold_start_model_10)
    print("cold_start_20......")
    recommend(click_trn_hist, click_val_hist, click_tst_hist, click_trn_last,
              click_val_last, cold_start_recall_20, article_info, cold_start_model_20)



