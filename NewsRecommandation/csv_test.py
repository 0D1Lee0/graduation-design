import pandas as pd
from collections import defaultdict
from const import OUTPUT_FOLDER,RAW_DATA_FOLDER
path = r'E:\recommend_data\csv_test.csv'
save_path = OUTPUT_FOLDER
data_path = RAW_DATA_FOLDER


def process_nan(df, topk=5):
    """
    处理喜欢文章类别为nan，选用当前数据集中前topk个文章类别处理nan
    :param df: 需要处理的数据集
    :param topk: 返回热门的文章类别数
    :return:
    """
    cate_id_dict = defaultdict(int)
    nan_dict = []
    #np.nan的类型为float
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







df = pd.read_csv(path)


print(df.dtypes)

df['col3'] = df.apply(lambda x:1 if str(x.col2) in set(x.col1[1:-1].split(',')) else 0, axis=1)
print(df.loc[3]['col1'][1:-1])
print(list(df.loc[2]['col1'][1:-1].split(',')))
print(type(df.loc[3]['col1']))
print(df)
user_info = pd.read_csv(save_path + 'user_info_04-12.csv')
print(user_info.dtypes)

print(type(user_info.loc[0]['user_id']))

merge_df = df.merge(user_info, left_on='col2', right_on='user_id')
#print(merge_df)
#print(merge_df.dtypes)

trn_user_item_feats_df = pd.read_csv(save_path + 'trn_user_item_feats_df.csv', )
trn_user_item_feats_df['click_article_id'] = trn_user_item_feats_df['click_article_id'].astype('int64')
val_user_item_feats_df = pd.read_csv(save_path + 'val_user_item_feats_df.csv')
val_user_item_feats_df['click_article_id'] = val_user_item_feats_df['click_article_id'].astype('int64')
tst_user_item_feats_df = pd.read_csv(save_path + 'tst_user_item_feats_df.csv')
tst_user_item_feats_df['click_article_id'] = tst_user_item_feats_df['click_article_id'].astype('int64')

articles_info = pd.read_csv(data_path + 'articles.csv')
print(articles_info.dtypes)
print(trn_user_item_feats_df.dtypes)

trn_user_item_feats_df = trn_user_item_feats_df.merge(user_info, on='user_id', how='left')
tst_user_item_feats_df = tst_user_item_feats_df.merge(user_info, on='user_id', how='left')
val_user_item_feats_df = val_user_item_feats_df.merge(user_info, on='user_id', how='left')

trn_user_item_feats_df = trn_user_item_feats_df.merge(articles_info, left_on='click_article_id', right_on='article_id')
val_user_item_feats_df = val_user_item_feats_df.merge(articles_info, left_on='click_article_id', right_on='article_id')
tst_user_item_feats_df = tst_user_item_feats_df.merge(articles_info, left_on='click_article_id', right_on='article_id')
"""trn_user_item_feats_df.to_csv(save_path + 'trn_feats_test.csv', index=False)

val_user_item_feats_df.to_csv(save_path + 'val_feats_test.csv', index=False)

tst_user_item_feats_df.to_csv(save_path + 'tst_feats_test.csv', index=False)"""

"""print(trn_user_item_feats_df.loc[0]['cate_list'])
print(type(trn_user_item_feats_df.loc[0]['cate_list']))
print(val_user_item_feats_df.loc[0]['user_id'])
print(val_user_item_feats_df.loc[0]['cate_list'])
print(type(val_user_item_feats_df.loc[0]['cate_list']))
print(tst_user_item_feats_df.loc[0]['cate_list'])
print(type(tst_user_item_feats_df.loc[0]['cate_list']))
print(val_user_item_feats_df.dtypes)"""

print("-----------------------------------------")
print(tst_user_item_feats_df['cate_list'])

process_nan(tst_user_item_feats_df)
"""for i, s in tst_user_item_feats_df['cate_list'].items():
    if isinstance(s, float):
        print(i)"""


trn_user_item_feats_df['is_hob'] = trn_user_item_feats_df.apply(
    lambda x : 1 if str(x.category_id) in x.cate_list[1:-1].split(',') else 0, axis=1
)

val_user_item_feats_df['is_hob'] = val_user_item_feats_df.apply(
    lambda x : 1 if str(x.category_id) in x.cate_list[1:-1].split(',') else 0, axis=1
)


tst_user_item_feats_df['is_hob'] = tst_user_item_feats_df.apply(
    lambda x : 1 if str(x.category_id) in x.cate_list[1:-1].split(',') else 0, axis=1
)

print(trn_user_item_feats_df['is_hob'])
count=0
for v in val_user_item_feats_df['is_hob']:
    if v==1:
        count += 1
print(count)
