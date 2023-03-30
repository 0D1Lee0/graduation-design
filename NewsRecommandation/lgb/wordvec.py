from collections import defaultdict
from datetime import datetime
import logging
import os
import pickle

import numpy as np
import numpy as pd
import pandas as pd
import const
from tqdm import tqdm
from gensim.models import Word2Vec
from feature import create_feature
from utils import get_hist_and_last_click, get_recall_list
from utils import get_article_info_df, reduce_mem
from lgb_const import DATA_PATH, SAVE_PATH




def train_item_word2vec(click_df, embed_size=64, save_name='item_w2v_emb.pkl', split_char=' '):
    print("开始训练word2vec向量......")
    click_df = click_df.sort_values('click_timestamp')
    # 只有转换成字符串才可以进行训练
    click_df['click_article_id'] = click_df['click_article_id'].astype(str)
    # 转换成句子的形式
    docs = click_df.groupby(['user_id'])['click_article_id'].apply(lambda x: list(x)).reset_index()
    docs = docs['click_article_id'].values.tolist()

    # 为了方便查看训练的进度，这里设定一个log信息

    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

    # 这里的参数对训练得到的向量影响也很大,默认负采样为5
    # 新的gensim函数库版本中, 将size改为vector_size, 将iter改为epochs
    w2v = Word2Vec(docs, vector_size=32, sg=1, window=5, seed=2020, workers=24, min_count=1, epochs=1)

    # 保存成字典的形式
    item_w2v_emb_dict = {k: w2v.wv[k] for k in click_df['click_article_id']}
    pickle.dump(item_w2v_emb_dict, open(SAVE_PATH + 'item_w2v_emb_.pkl', 'wb'))

    return item_w2v_emb_dict


# 可以通过字典查询对应的item的Embedding
def get_embedding(save_path, all_click_df):
    # w2v Embedding是需要提前训练好的
    if os.path.exists(save_path + 'item_w2v_emb.pkl'):
        item_w2v_emb_dict = pickle.load(open(save_path + 'item_w2v_emb_.pkl', 'rb'))
    else:
        item_w2v_emb_dict = train_item_word2vec(all_click_df)


    return item_w2v_emb_dict








# 将召回列表转换成df的形式
def recall_dict_2_df(recall_list_dict):
    df_row_list = []  # [user, item, score]
    print("\n开始将召回列表转换成df......")
    for user, recall_list in tqdm(recall_list_dict.items()):
        for item, score in recall_list:
            df_row_list.append([user, item, score])

    col_names = ['user_id', 'sim_item', 'score']
    recall_list_df = pd.DataFrame(df_row_list, columns=col_names)

    return recall_list_df


# 负采样函数，这里可以控制负采样时的比例, 这里给了一个默认的值
def neg_sample_recall_data(recall_items_df, sample_rate=0.001):
    pos_data = recall_items_df[recall_items_df['label'] == 1]
    neg_data = recall_items_df[recall_items_df['label'] == 0]

    if len(neg_data) != 0:
        print('pos_data_num:', len(pos_data), 'neg_data_num:', len(neg_data), 'pos/neg:', len(pos_data) / len(neg_data))
    else:
        print('pos_data_num:', len(pos_data), 'neg_data_num:', len(neg_data))
    # 分组采样函数
    def neg_sample_func(group_df):
        neg_num = len(group_df)
        sample_num = max(int(neg_num * sample_rate), 1)  # 保证最少有一个
        sample_num = min(sample_num, 5)  # 保证最多不超过5个，这里可以根据实际情况进行选择
        return group_df.sample(n=sample_num, replace=True)

    # 对用户进行负采样，保证所有用户都在采样后的数据中
    neg_data_user_sample = neg_data.groupby('user_id', group_keys=False).apply(neg_sample_func)
    # 对文章进行负采样，保证所有文章都在采样后的数据中
    neg_data_item_sample = neg_data.groupby('sim_item', group_keys=False).apply(neg_sample_func)

    # 将上述两种情况下的采样数据合并
    neg_data_new = neg_data_user_sample.append(neg_data_item_sample)
    # 由于上述两个操作是分开的，可能将两个相同的数据给重复选择了，所以需要对合并后的数据进行去重
    neg_data_new = neg_data_new.sort_values(['user_id', 'score']).drop_duplicates(['user_id', 'sim_item'], keep='last')

    # 将正样本数据合并
    data_new = pd.concat([pos_data, neg_data_new], ignore_index=True)

    return data_new


# 召回数据打标签
def get_rank_label_df(recall_list_df, label_df, is_test=False):
    # 测试集是没有标签了，为了后面代码同一一些，这里直接给一个负数替代
    if is_test:
        recall_list_df['label'] = -1
        return recall_list_df

    label_df = label_df.rename(columns={'click_article_id': 'sim_item'})
    recall_list_df_ = recall_list_df.merge(label_df[['user_id', 'sim_item', 'click_timestamp']], \
                                           how='left', on=['user_id', 'sim_item'])
    recall_list_df_['label'] = recall_list_df_['click_timestamp'].apply(lambda x: 0.0 if np.isnan(x) else 1.0)
    del recall_list_df_['click_timestamp']

    return recall_list_df_


def get_user_recall_item_label_df(click_trn_hist, click_val_hist, click_tst_hist, click_trn_last, click_val_last,
                                  recall_list_df):
    # 获取训练数据的召回列表
    trn_user_items_df = recall_list_df[recall_list_df['user_id'].isin(click_trn_hist['user_id'].unique())]
    # 训练数据打标签
    trn_user_item_label_df = get_rank_label_df(trn_user_items_df, click_trn_last, is_test=False)
    # 训练数据负采样
    trn_user_item_label_df = neg_sample_recall_data(trn_user_item_label_df)


    val_user_items_df = recall_list_df[recall_list_df['user_id'].isin(click_val_hist['user_id'].unique())]
    val_user_item_label_df = get_rank_label_df(val_user_items_df, click_val_last, is_test=False)
    val_user_item_label_df = neg_sample_recall_data(val_user_item_label_df)


    # 测试数据不需要进行负采样，直接对所有的召回商品进行打-1标签
    tst_user_items_df = recall_list_df[recall_list_df['user_id'].isin(click_tst_hist['user_id'].unique())]
    tst_user_item_label_df = get_rank_label_df(tst_user_items_df, None, is_test=True)

    return trn_user_item_label_df, val_user_item_label_df, tst_user_item_label_df


# 将最终的召回的df数据转换成字典的形式做排序特征
def make_tuple_func(group_df):
    row_data = []
    for name, row_df in group_df.iterrows():
        row_data.append((row_df['sim_item'], row_df['score'], row_df['label']))

    return row_data


# 处理空值
def process_nan(df, topk=5):
    """
    处理喜欢文章类别为nan，选用当前数据集中前topk个文章类别处理nan
    :param df: 需要处理的数据集
    :param topk: 返回热门的文章类别数
    :return:
    """
    cate_id_dict = defaultdict(int)
    nan_dict = []
    # np.nan的类型为float
    for index, list in df['cate_list'].items():
        if isinstance(list, float):
            nan_dict.append(index)
        else:
            for l in list:
                cate_id_dict[l] += 1

    hot_cate_id = []
    for id, count in sorted(cate_id_dict.items(), key=lambda x: x[1], reverse=True)[:topk]:
        hot_cate_id.append(id)

    for index in nan_dict:
        df.loc[index, 'cate_list'] = str(hot_cate_id)


if __name__ == '__main__':
    print("开始读取文章信息......")
    article_info_df = get_article_info_df()



    click_trn_hist = pd.read_csv(DATA_PATH + 'click_trn_hist_.csv')
    click_val_hist = pd.read_csv(DATA_PATH + 'click_val_hist_.csv')
    click_tst_hist = pd.read_csv(DATA_PATH + 'click_tst_hist_.csv')
    click_trn_last = pd.read_csv(DATA_PATH + 'click_trn_last_.csv')
    click_val_last = pd.read_csv(DATA_PATH + 'click_val_last_.csv')

    click_all_hist = click_trn_hist.append(click_val_hist)
    click_all_hist = click_all_hist.append(click_tst_hist)
    # 读取召回列表
    recall_list_dict = get_recall_list(SAVE_PATH, 'i2i_itemcf')
    print("读取召回列表......")
    recall_list_df = recall_dict_2_df(recall_list_dict)

    print("开始给训练验证数据打标签......")
    # 给训练验证数据打标签，并负采样（这一部分时间比较久）
    trn_user_item_label_df, val_user_item_label_df, tst_user_item_label_df = get_user_recall_item_label_df(
        click_trn_hist,
        click_val_hist,
        click_tst_hist,
        click_trn_last,
        click_val_last,
        recall_list_df)

    print("开始保存标签......")
    trn_label_path = SAVE_PATH + 'trn_user_item_label_' + '.csv'
    val_label_path = SAVE_PATH + 'val_user_item_label_' + '.csv'
    tst_label_path = SAVE_PATH + 'tst_user_item_label_' + '.csv'
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

    #item_w2v_emb_dict = train_item_word2vec(click_all_hist)"""


    item_w2v_emb_dict = get_embedding(SAVE_PATH, click_all_hist)

    # 获取训练验证及测试数据中召回列文章相关特征
    trn_user_item_feats_df = create_feature(trn_user_item_label_tuples_dict.keys(), trn_user_item_label_tuples_dict, \
                                            click_trn_hist, article_info_df, item_w2v_emb_dict)

    val_user_item_feats_df = create_feature(val_user_item_label_tuples_dict.keys(), val_user_item_label_tuples_dict, \
                                            click_val_hist, article_info_df, item_w2v_emb_dict)

    tst_user_item_feats_df = create_feature(tst_user_item_label_tuples_dict.keys(), tst_user_item_label_tuples_dict, \
                                            click_tst_hist, article_info_df, item_w2v_emb_dict)

    # 保存一份
    trn_user_item_feats_df.to_csv(SAVE_PATH + 'trn_user_item_feats_df.csv', index=False)

    val_user_item_feats_df.to_csv(SAVE_PATH + 'val_user_item_feats_df.csv', index=False)

    tst_user_item_feats_df.to_csv(SAVE_PATH + 'tst_user_item_feats_df.csv', index=False)

    user_info = pd.read_csv(DATA_PATH + 'user_info_.csv')
    print("开始读取特征......")
    trn_user_item_feats_df = pd.read_csv(SAVE_PATH + 'trn_user_item_feats_df.csv')
    tst_user_item_feats_df = pd.read_csv(SAVE_PATH + 'tst_user_item_feats_df.csv')
    val_user_item_feats_df = pd.read_csv(SAVE_PATH + 'val_user_item_feats_df.csv')

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
    trn_user_item_feats_df.to_csv(SAVE_PATH + 'trn_user_item_feats_df_.csv', index=False)
    val_user_item_feats_df.to_csv(SAVE_PATH + 'val_user_item_feats_df_.csv', index=False)
    tst_user_item_feats_df.to_csv(SAVE_PATH + 'tst_user_item_feats_df_.csv', index=False)
