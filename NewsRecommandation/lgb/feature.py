import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from lgb_const import DATA_PATH, SAVE_PATH

def create_feature(users_id, recall_list, click_hist_df, articles_info, articles_emb, N=1):
    """
    基于用户的历史行为做相关特征
    :param users_id: 用户id
    :param recall_list: 对于每个用户召回的候选文章列表
    :param click_hist_df: 用户的历史点击信息
    :param articles_info: 文章信息
    :param articles_emb: 文章的embedding向量, 这个可以用item_content_emb, item_w2v_emb, item_youtube_emb
    :param N: 最近的N次点击  由于A日志里面很多用户只存在一次历史点击， 所以为了不产生空值，默认是1
    """

    # 建立一个二维列表保存结果， 后面要转成DataFrame
    all_user_feas = []
    i = 0
    for user_id in tqdm(users_id):
        # 该用户的最后N次点击
        hist_user_items = click_hist_df[click_hist_df['user_id'] == user_id]['click_article_id'][-N:]

        # 遍历该用户的召回列表
        for rank, (article_id, score, label) in enumerate(recall_list[user_id]):
            # 该文章建立时间, 字数
            a_create_time = articles_info[articles_info['article_id'] == article_id]['created_at_ts'].values[0]
            a_words_count = articles_info[articles_info['article_id'] == article_id]['words_count'].values[0]
            single_user_fea = [user_id, article_id]
            # 计算与最后点击的商品的相似度的和， 最大值和最小值， 均值
            sim_fea = []
            time_fea = []
            word_fea = []
            # 遍历用户的最后N次点击文章
            for hist_item in hist_user_items:
                b_create_time = articles_info[articles_info['article_id'] == hist_item]['created_at_ts'].values[0]
                b_words_count = articles_info[articles_info['article_id'] == hist_item]['words_count'].values[0]
                article_id_ = int(article_id)

                if hist_item or article_id_ not in articles_emb.keys():
                    sim_fea.append(0.0)
                else:
                    # 计算向量内积
                    sim_fea.append(np.dot(articles_emb[hist_item], articles_emb[article_id_]))


                time_fea.append(abs(a_create_time - b_create_time))
                word_fea.append(abs(a_words_count - b_words_count))

            single_user_fea.extend(sim_fea)  # 相似性特征
            single_user_fea.extend(time_fea)  # 时间差特征
            single_user_fea.extend(word_fea)  # 字数差特征
            single_user_fea.extend([max(sim_fea), min(sim_fea), sum(sim_fea), sum(sim_fea) / len(sim_fea)])  # 相似性的统计特征


            single_user_fea.extend([score, rank, label])
            # 加入到总的表中
            all_user_feas.append(single_user_fea)

    # 定义列名
    id_cols = ['user_id', 'click_article_id']
    sim_cols = ['sim' + str(i) for i in range(N)]
    time_cols = ['time_diff' + str(i) for i in range(N)]
    word_cols = ['word_diff' + str(i) for i in range(N)]
    sat_cols = ['sim_max', 'sim_min', 'sim_sum', 'sim_mean']
    user_item_sim_cols = []
    user_score_rank_label = ['score', 'rank', 'label']
    cols = id_cols + sim_cols + time_cols + word_cols + sat_cols + user_item_sim_cols + user_score_rank_label

    # 转成DataFrame
    df = pd.DataFrame(all_user_feas, columns=cols)

    return df



def active_level(all_data, cols):
    """
    制作区分用户活跃度的特征
    :param all_data: 数据集
    :param cols: 用到的特征列
    """
    data = all_data[cols]
    data_copy = data.sort_values(['user_id', 'click_timestamp'])
    user_act = pd.DataFrame(data_copy.groupby('user_id', as_index=False)[['click_article_id', 'click_timestamp']]. \
                            agg({'click_article_id': np.size, 'click_timestamp': {list}}).values,
                            columns=['user_id', 'click_size', 'click_timestamp'])

    # 计算时间间隔的均值
    def time_diff_mean(l):
        if len(l) == 1:
            return 1
        else:
            return np.mean([j - i for i, j in list(zip(l[:-1], l[1:]))])

    user_act['time_diff_mean'] = user_act['click_timestamp'].apply(lambda x: time_diff_mean(x))

    # 点击次数取倒数
    user_act['click_size'] = 1 / user_act['click_size']

    # 两者归一化
    user_act['click_size'] = (user_act['click_size'] - user_act['click_size'].min()) / (
                user_act['click_size'].max() - user_act['click_size'].min())
    user_act['time_diff_mean'] = (user_act['time_diff_mean'] - user_act['time_diff_mean'].min()) / (
                user_act['time_diff_mean'].max() - user_act['time_diff_mean'].min())
    user_act['active_level'] = user_act['click_size'] + user_act['time_diff_mean']

    user_act['user_id'] = user_act['user_id'].astype('int')
    del user_act['click_timestamp']

    return user_act

def get_articles_dict(articles_df):
    articles_dict = {}
    for _, row in articles_df.iterrows():
        article_id = row['article_id']
        category_id = row['category_id']
        created_at_ts = row['created_at_ts']
        words_count = row['words_count']
        articles_dict[article_id] = {}
        articles_dict[article_id]['category_id'] = category_id
        articles_dict[article_id]['created_at_ts'] = created_at_ts
        articles_dict[article_id]['words_count'] = words_count

    return articles_dict


def device_fea(all_data, cols):
    """
    用户的设备特征
    :param all_data: 数据集
    :param cols: 用到的特征列
    :return: 各个用户的设备信息
    """


    user_device_info = all_data[cols]
    # 用众数来表示每个用户的设备信息
    user_device_info = user_device_info.groupby('user_id').agg(lambda x: x.value_counts().index[0]).reset_index()

    return user_device_info

def user_time_hob_fea(all_data, cols, articles, article_col):
    """
    用户的时间习惯
    :param all_data: 数据集
    :param cols:
    :return:
    """
    create_time = []
    for index, row in all_data.iterrows():
        article_id = row['click_article_id']
        create_time.append(articles[article_id][article_col])


    user_time_hob_info = all_data[cols]
    user_time_hob_info[article_col] = create_time

    # 先把时间戳进行归一化
    mm = MinMaxScaler()
    user_time_hob_info['click_timestamp'] = mm.fit_transform(user_time_hob_info[['click_timestamp']])
    user_time_hob_info['created_at_ts'] = mm.fit_transform(user_time_hob_info[['created_at_ts']])

    user_time_hob_info = user_time_hob_info.groupby('user_id').agg('mean').reset_index()

    user_time_hob_info.rename(columns={'click_timestamp': 'user_time_hob1', 'created_at_ts': 'user_time_hob2'},
                              inplace=True)
    return user_time_hob_info


def user_cat_hob_fea(all_data, cols, articles, article_col):
    """
    用户的主题爱好
    :param all_data: 数据集
    :param cols: 用到的特征列
    """
    category = []
    for index, row in all_data.iterrows():
        article_id = row['click_article_id']
        category.append(articles[article_id][article_col])
    user_category_hob_info = all_data[cols]
    user_category_hob_info[article_col] = category
    user_category_hob_info = user_category_hob_info.groupby('user_id').agg({list}).reset_index()

    user_cat_hob_info = pd.DataFrame()
    user_cat_hob_info['user_id'] = user_category_hob_info['user_id']
    user_cat_hob_info['cate_list'] = user_category_hob_info['category_id']

    return user_cat_hob_info

def user_words_hob_fea(all_data, articles, article_col):
    words_count = []
    for _, row in all_data.iterrows():
        article_id = row['click_article_id']
        words = articles[article_id][article_col]
        words_count.append(words)

    user_words_count_hob_info = all_data
    user_words_count_hob_info[article_col] = words_count

    user_wcou_info = user_words_count_hob_info.groupby('user_id')['words_count'].agg('mean').reset_index()
    user_wcou_info.rename(columns={'words_count': 'words_hbo'}, inplace=True)

    return user_wcou_info

def merge_info(all_data, device_cols, user_time_hob_cols, user_category_hob_cols, articles, articles_cols):
    user_act_fea = active_level(all_data, ['user_id', 'click_article_id', 'click_timestamp'])
    print("开始分析用户设备特征......")
    user_device_info = device_fea(all_data, device_cols)
    print("开始分析用户点击时间特征......")
    user_time_hob_info = user_time_hob_fea(all_data, user_time_hob_cols, articles, articles_cols[0])
    print("开始分析用户看的文章类别特征......")
    user_cat_hob_info = user_cat_hob_fea(all_data, user_category_hob_cols, articles, articles_cols[1])
    print("开始分析用户看的文章字数特征......")
    user_wcou_info = user_words_hob_fea(all_data, articles, articles_cols[2])
    user_info = pd.merge(user_act_fea, user_device_info, on='user_id')
    user_info = user_info.merge(user_time_hob_info, on='user_id')
    user_info = user_info.merge(user_cat_hob_info, on='user_id')
    user_info = user_info.merge(user_wcou_info, on='user_id')
    user_last_click_time_info = user_last_click_time_fea(all_data)
    user_info = user_info.merge(user_last_click_time_info, on='user_id')

    user_info.to_csv(DATA_PATH + 'user_info_' + '.csv', index=False)
    print("用户信息保存成功！")

def user_last_click_time_fea(all_data):
    user_last_click_time_dict = {}
    for _, row in all_data.iterrows():
        user_id = row['user_id']
        timestamp = int(row['click_timestamp'])
        user_last_click_time_dict.setdefault(user_id, 0)
        last_time = user_last_click_time_dict[user_id]
        if timestamp > last_time:
            user_last_click_time_dict[user_id] = timestamp
    user_last_click_time_info = pd.DataFrame.from_dict(user_last_click_time_dict, orient='index', columns=['last_click_time'])
    user_last_click_time_info = user_last_click_time_info.reset_index()
    user_last_click_time_info = user_last_click_time_info.rename(columns={'index' : 'user_id'})

    return user_last_click_time_info


if __name__ == '__main__':

    articles_df = pd.read_csv(DATA_PATH + 'articles.csv')
    click_trn_hist = pd.read_csv(DATA_PATH + 'click_trn_hist_.csv')
    click_val_hist = pd.read_csv(DATA_PATH + 'click_val_hist_.csv')
    click_tst_hist = pd.read_csv(DATA_PATH + 'click_tst_hist_.csv')

    click_all_hist = click_trn_hist.append(click_val_hist)
    click_all_hist = click_all_hist.append(click_tst_hist)

    device_cols = ['user_id', 'click_environment', 'click_deviceGroup', 'click_os', 'click_country', 'click_region',
                   'click_referrer_type']
    user_time_hob_cols = ['user_id', 'click_timestamp']
    user_category_hob_cols = ['user_id']
    articles = get_articles_dict(articles_df)
    articles_cols = ['created_at_ts', 'category_id', 'words_count']

    merge_info(click_all_hist, device_cols, user_time_hob_cols, user_category_hob_cols, articles, articles_cols)
