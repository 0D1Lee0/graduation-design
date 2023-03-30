
from const import RAW_DATA_FOLDER
import pandas as pd

def get_all_click_df(data_path=RAW_DATA_FOLDER):

    trn_click = pd.read_csv(data_path + 'train_click_log.csv')
    tst_click = pd.read_csv(data_path + 'testA_click_log.csv')

    all_click = trn_click.append(tst_click)

    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    return all_click

def merge(data_path=RAW_DATA_FOLDER):
    print("分别导入数据......")
    mergeDf = get_all_click_df(data_path)
    print("导出合并数据......")
    mergeDf.to_csv(data_path+"all_data.csv")
    print("导出成功！")


merge(RAW_DATA_FOLDER)
