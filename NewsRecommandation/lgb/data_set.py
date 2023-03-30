import pickle

import pandas as pd

from lgb_const import DATA_PATH, SAVE_PATH

click_trn_hist = pd.read_csv(DATA_PATH + 'click_trn_hist_.csv')
click_trn_last = pd.read_csv(DATA_PATH + 'click_trn_last_.csv')

click_val_hist = pd.read_csv(DATA_PATH + 'click_val_hist_.csv')
click_val_last = pd.read_csv(DATA_PATH + 'click_val_last_.csv')

click_tst_hist = pd.read_csv(DATA_PATH + 'click_tst_hist_.csv')
click_tst_last = pd.read_csv(DATA_PATH + 'click_tst_last_.csv')

print("len(trn_hist)=" + str(len(click_trn_hist['user_id'].unique())), end="  ")
print("len(trn_last)=" + str(len(click_trn_last['user_id'].unique())))

print("len(val_hist)=" + str(len(click_val_hist['user_id'].unique())), end="  ")
print("len(val_last)=" + str(len(click_val_last['user_id'].unique())))

print("len(tst_hist)=" + str(len(click_tst_hist['user_id'].unique())), end="  ")
print("len(tst_last)=" + str(len(click_tst_last['user_id'].unique())))

trn_user_item_feats_df = pd.read_csv(SAVE_PATH + 'trn_user_item_feats_df.csv')
tst_user_item_feats_df = pd.read_csv(SAVE_PATH + 'tst_user_item_feats_df.csv')
val_user_item_feats_df = pd.read_csv(SAVE_PATH + 'val_user_item_feats_df.csv')

print("len(trn_feats)=" + str(len(trn_user_item_feats_df)))
print("len(val_feats)=" + str(len(val_user_item_feats_df)))
print("len(tst_feats)=" + str(len(tst_user_item_feats_df)))

trn_user_item_feats_df_ = pd.read_csv(SAVE_PATH + 'trn_user_item_feats_df.csv')
tst_user_item_feats_df_ = pd.read_csv(SAVE_PATH + 'tst_user_item_feats_df.csv')
val_user_item_feats_df_ = pd.read_csv(SAVE_PATH + 'val_user_item_feats_df.csv')

print("len(trn_feats_)=" + str(len(trn_user_item_feats_df_)))
print("len(val_feats_)=" + str(len(val_user_item_feats_df_)))
print("len(tst_feats_)=" + str(len(tst_user_item_feats_df_)))

all_click = pd.read_csv(DATA_PATH + 'all_data.csv')
print("len(all_users)=" + str(len(all_click['user_id'].unique())))

trn_hist_users = click_trn_hist['user_id'].unique()
trn_last_users = click_trn_last['user_id'].unique()

val_hist_users = click_val_hist['user_id'].unique()
val_last_users = click_val_last['user_id'].unique()

tst_hist_users = click_tst_hist['user_id'].unique()
tst_last_users = click_tst_last['user_id'].unique()

print("len(trn_hist_users)=" + str(len(trn_hist_users)), end='  ')
print("len(trn_last_users)=" + str(len(trn_last_users)))

print("len(val_hist_users)=" + str(len(val_hist_users)), end='  ')
print("len(val_last_users)=" + str(len(val_last_users)))

print("len(tst_hist_users)=" + str(len(tst_hist_users)), end='  ')
print("len(tst_last_users)=" + str(len(tst_last_users)))

final_recall_items_dict = pickle.load(open(SAVE_PATH +'final_recall_items_dict.pkl', 'rb'))
print(len(final_recall_items_dict))