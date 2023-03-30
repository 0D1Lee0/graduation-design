import random
from datetime import datetime
import const
from const import RAW_DATA_FOLDER
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from baseline.itemcf import reduce_mem
columns = ["user_id", "click_article_id", "click_timestamp", "click_environment", "click_deviceGroup", "click_os",
               "click_country", "click_region", "click_referrer_type"]
def divide_data(data_path, validation_rate=0.1):
    train_data = []
    test_data = []
    file = data_path+"all_data.csv"

    """for line in load_file(file):
        
        #user_id, click_article_id, click_timestamp, click_environment, click_deviceGroup, click_os, click_country, click_region, click_referrer_type = line.split(",")
        data_list = line.split(',')
        data_list = data_list[1:]
        if (random.random() < pivot):
            train_data.append(data_list)
        else:
            test_data.append(data_list)"""
    with open(file) as f:
        all_data = np.loadtxt(f, str, delimiter=",", skiprows=1)
    #print(all_data.shape[0])
    privot = int(all_data.shape[0] * (1-validation_rate))
    index = np.random.permutation(all_data.shape[0])
    print(index)
    index_list = index.tolist()
    train_index= index_list[0:privot]
    test_index = index_list[privot:]

    for i in train_index:
        train_data.append(all_data[i].tolist()[1:])

    for i in test_index:
        test_data.append(all_data[i].tolist()[1:])



    print(type(index))
    print(index.shape[0])
    print('Split trainingSet and testSet success!')
    print('Ther length of TrainSet = %s' % len(train_data))
    print('Ther length of TestSet = %s' % len(test_data))
    if len(train_data) + len(test_data) == all_data.shape[0]:
        print("match!")

    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    train_df.columns = columns
    test_df.columns = columns

    train_df[columns] = train_df[columns].applymap(lambda x : int(x))
    test_df[columns] = test_df[columns].applymap(lambda x: int(x))
    return train_df, test_df



def process_data_df(data_path, test_size=0.2):
    df = pd.read_csv(data_path + "all_data.csv")
    df_train, df_test = train_test_split(df, test_size)

    return df_train, df_test

def load_file(file):
    with open(file, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:  # 去掉文件第一行的title
                continue
            yield line.strip('\r\n')
    print('Load %s success!' % file)


def trn_val_tst_split(data_path, test_rate=0.1):
    all_data = pd.read_csv(data_path + 'all_data.csv', index_col=0)
    all_data = reduce_mem(all_data)
    all_data = shuffle(all_data)
    all_data = shuffle(all_data)
    all_data = all_data.reset_index(drop=True)



    rows = len(all_data)
    tst_len = int(rows * test_rate)
    train_len = rows - tst_len
    train_data = all_data[0 : train_len]
    test_data = all_data[train_len:]
    val_len = tst_len

    click_trn, click_val, val_ans = trn_val_split(train_data, val_len)
    click_trn_ = click_trn.reset_index(drop=True)
    click_val_ = click_val.reset_index(drop=True)
    val_ans_ = val_ans.reset_index(drop=True)
    test_data_ = test_data.reset_index(drop=True)
    return click_trn_, click_val_, val_ans_, test_data_

# all_click_df指的是训练集
# sample_user_nums 采样作为验证集的用户数量
def trn_val_split(all_click_df, sample_user_nums):
    all_click = all_click_df
    all_user_ids = all_click.user_id.unique()

    # replace=True表示可以重复抽样，反之不可以
    sample_user_ids = np.random.choice(all_user_ids, size=sample_user_nums, replace=False)

    #验证集中的用户不在在训练集中
    click_val = all_click[all_click['user_id'].isin(sample_user_ids)]
    click_trn = all_click[~all_click['user_id'].isin(sample_user_ids)]

    # 将验证集中的最后一次点击给抽取出来作为答案
    click_val = click_val.sort_values(['user_id', 'click_timestamp'])
    val_ans = click_val.groupby('user_id').tail(1)

    click_val = click_val.groupby('user_id').apply(lambda x: x[:-1]).reset_index(drop=True)

    # 去除val_ans中某些用户只有一个点击数据的情况，如果该用户只有一个点击数据，又被分到ans中，
    # 那么训练集中就没有这个用户的点击数据，出现用户冷启动问题，给自己模型验证带来麻烦
    val_ans = val_ans[val_ans.user_id.isin(click_val.user_id.unique())]  # 保证答案中出现的用户再验证集中还有
    click_val = click_val[click_val.user_id.isin(val_ans.user_id.unique())]

    return click_trn, click_val, val_ans


if __name__ == '__main__':
    #train_df, test_df = divide_data(RAW_DATA_FOLDER)
    click_trn, click_val, val_ans, test_data = trn_val_tst_split(const.RAW_DATA_FOLDER)
    click_trn.to_csv(const.OUTPUT_FOLDER + "train" + '_' + datetime.today().strftime('%m-%d') + '.csv',
                     index=False, header=True)
    click_val.to_csv(const.OUTPUT_FOLDER + "val" + '_' + datetime.today().strftime('%m-%d') + '.csv',
                     index=False, header=True)
    test_data.to_csv(const.OUTPUT_FOLDER + "test" + '_' + datetime.today().strftime('%m-%d') + '.csv',
                     index=False, header=True)
    val_ans.to_csv(const.OUTPUT_FOLDER + "ans" + '_' + datetime.today().strftime('%m-%d') + '.csv',
                     index=False, header=True)
    """click_trn = click_trn.loc[:, ~click_trn.columns.str.contains('Unnamed')]
    click_val = click_val.loc[:, ~click_val.columns.str.contains('Unnamed')]
    val_ans = val_ans.loc[:, ~val_ans.columns.str.contains('Unnamed')]
    test_data = test_data.loc[:, ~test_data.columns.str.contains('Unnamed')]"""

    print(click_trn.head(5))
    print(click_val.head(5))
    print(val_ans.head(5))
    print(test_data.head(5))





