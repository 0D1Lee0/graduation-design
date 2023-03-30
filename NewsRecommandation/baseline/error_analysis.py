import numpy as np
import pandas as pd
import csv
from datetime import datetime
from baseline.process_data import load_file
import const
def getPredictedDict(file):

    with open(file) as f:
        predicted_array = np.loadtxt(file, str, delimiter=",", skiprows=1)

    predicted_dict = {}

    rows = predicted_array.shape[0]

    for i in range(0, rows):
        user_predicted = predicted_array[i]
        user_id = int(user_predicted[0])
        articles = user_predicted[1:]
        articles_int = []
        for article in articles:
            articles_int.append(int(article))
        predicted_dict[user_id] = articles_int

    return predicted_dict

def getTestDict(test_array):
    test_dict = {}
    rows = test_array.shape[0]

    for i in range(rows):
        user_predicted = test_array[i]
        user_id = int(user_predicted[0])
        article_id = int(user_predicted[1])
        test_dict.setdefault(user_id, [])
        test_dict[user_id].append(article_id)

    return test_dict



def getScore(predicted_dict, test_dict, digit=2):
    scores_dict = {}
    predicted_user = list(predicted_dict.keys())
    for user_id, items in test_dict.items():
        #print("type(user_id): " + str(type(user_id)))
        #print("type(key[0]): "+ str(type(predicted_user[0])))
        if user_id not in predicted_user:
            continue
        predicted_items = predicted_dict[user_id]
        score_sum = 0.0
        for index in range(0, len(predicted_items)):
            article_id = predicted_items[index]
            #print("type(article_id): " + str(type(article_id)))
            #print("type(items[0]): "+ str(type(items[0])))
            if article_id in items:
                score_sum += (1.0 / (index + 1))

        scores_dict[user_id] = round(score_sum, digit)

    return scores_dict

if __name__ == "__main__":

    test_file = r'E:\recommend_data\aliyun\result_data\test_04-01.csv'
    predicted_file = r'E:\recommend_data\aliyun\result_data\itemcf_baseline_04-01.csv'

    with open(test_file) as f:
        test_array = np.loadtxt(f, str, delimiter=",", skiprows=1)

    """with open(predicted_file) as f:
        predicted_array = np.loadtxt(f, str, delimiter=",", skiprows=1)"""



    test_dict = getTestDict(test_array)
    predicted_dict = getPredictedDict(predicted_file)

    scores_dict = getScore(predicted_dict, test_dict)

    scores_df = pd.DataFrame.from_dict(scores_dict, orient='index', columns=['score'])
    scores_df = scores_df.reset_index()
    scores_df = scores_df.rename(columns={'index': 'user_id'})

    scores_name = const.OUTPUT_FOLDER + "error_analysis" + '_' + datetime.today().strftime('%m-%d') + '.csv'
    scores_df.to_csv(scores_name, index=False, header=True)

    score_dict_length = 0
    hill_count = 0

    for line in load_file(scores_name):
        score_dict_length = score_dict_length + 1
        score = line.split(',')[1]
        if float(score) > 0:
            hill_count = hill_count + 1

    print("The precent of hill is %.2f" % (1.0*hill_count/score_dict_length))