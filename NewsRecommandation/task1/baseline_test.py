import numpy as np
import csv

# -*- coding=utf-8 -*-

predicted_data = r'E:\recommend_data\aliyun\result_data\itemcf_baseline_03-26.csv'
readl_data = r'E:\recommend_data\aliyun\original_data\testA_click_log.csv'

with open(predicted_data) as f:
    predicted = np.loadtxt(f, str, delimiter=",", skiprows=1)   #二维数组

print(type(predicted))
print(predicted[0][0])
print(predicted[0][1:])
print(predicted.shape[0])
real = {}

with open(readl_data) as f:
    f_csv = csv.reader(f)

    header = next(f_csv)

    for row in f_csv:
        user = row[0]
        newsId = row[1]
        real.setdefault(user, [])
        real[user].append(newsId)

scores = []
index = 0
row = predicted.shape[0]
column = predicted.shape[1]

for i in range(0, 10):
    user_predicted = predicted[i]
    score = 0
    user_real = real[user_predicted[0]]
    print(user_real)
    for j in range(1, column):
        print(user_predicted[j])
        if user_predicted[j] in user_real:
            print("hill!")
            point = 1/j
            score = score + point
    scores.append(score)
    print("score:"+str(score))
    print("--------------------------")

print(len(scores))
