import pandas as pd
from itemcf import get_user_item_time

article_path = r'E:\recommend_data\aliyun\original_data\articles.csv'
user_item_time_path = r'E:\recommend_data\aliyun\result_data\train_04-01.csv'
article_df = pd.read_csv(article_path)
user_item_time_df = pd.read_csv(user_item_time_path)
article_df = article_df.astype(int)
user_item_time_df = user_item_time_df.astype(pd.Int64Dtype())
print(article_df.dtypes)
print(user_item_time_df.dtypes)

user_item_time_dict = get_user_item_time(user_item_time_df)

