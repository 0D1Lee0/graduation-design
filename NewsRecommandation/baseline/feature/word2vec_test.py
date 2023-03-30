from word2vec import get_embedding, get_article_info_df
from baseline.user_frature import create_feature
import const

data_path = const.RAW_DATA_FOLDER
save_path = const.OUTPUT_FOLDER

if __name__ == '__main__':
    article_info = get_article_info_df()
