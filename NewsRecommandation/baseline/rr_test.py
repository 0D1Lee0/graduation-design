from datetime import datetime

import pandas as pd

import const
import error_analysis
import numpy as np

from baseline import process_data

predicted_path = r'E:\recommend_data\aliyun\result_data\itemcf_related_rule_04-05.csv'
test_path = r'E:\recommend_data\aliyun\result_data\test_04-05.csv'


predicted_dict =error_analysis.getPredictedDict(predicted_path)

with open(test_path) as f:
    test_array = np.loadtxt(test_path, str, delimiter=",", skiprows=1)

test_dict = error_analysis.getTestDict(test_array)

scores_dict = error_analysis.getScore(predicted_dict, test_dict, 2)

scores_df = pd.DataFrame.from_dict(scores_dict, orient='index', columns=['score'])
scores_df = scores_df.reset_index()
scores_df = scores_df.rename(columns={'index' : 'user_id'})

scores_name = const.OUTPUT_FOLDER + "error_analysis_itemcf_rr" + '_' + datetime.today().strftime('%m-%d') + '.csv'
scores_df.to_csv(scores_name, index=False, header=True)

score_dict_length = 0
hill_count = 0

for line in process_data.load_file(scores_name):
    score_dict_length = score_dict_length + 1
    score = line.split(',')[1]
    if float(score) > 0.0:
        hill_count = hill_count + 1

print(hill_count)
print(score_dict_length)
print("The precent of hill is %.2f" % (1.0*hill_count/score_dict_length))