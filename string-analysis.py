# for each dataset the average/median/percentile 90 distance between the entity names in the two languages

import Levenshtein
from include.Config import Config
import re
import numpy as np
import pickle
from collections import Counter
import matplotlib.pyplot as plt

# could not remove comas, as variants...
lowbound = 0; highbound = 17880

inf1 = open(Config.e1)
id2name1_test = dict()
for i1, line in enumerate(inf1):
    strs = line.strip().split('\t')
    wordline = strs[1].split('/')[-1].lower().replace('(','').replace(')','')
    wordline = re.sub("[\s+\.\!\/_,$%^*_\-(+\"\')]+|[+—?【】“”！，。？、~@#￥%……&*（）]+'", "",wordline)
    if i1>=lowbound and i1<highbound:
        id2name1_test[i1] = wordline
print(len(id2name1_test))

inf2 = open(Config.e2)
id2name2_test = dict()
for i1, line in enumerate(inf2):
    strs = line.strip().split('\t')
    wordline = strs[1].split('/')[-1].lower().replace('(','').replace(')','')
    wordline = re.sub("[\s+\.\!\/_,$%^*_\-(+\"\')]+|[+—?【】“”！，。？、~@#￥%……&*（）]+'", "",wordline)
    if i1>=lowbound and i1<highbound:
        id2name2_test[i1] = wordline
print(len(id2name2_test))

overallscores = []
coun = 0
for item in range(lowbound, highbound):
    # print(item)
    if item in id2name1_test:
        name1 = id2name1_test[item]
        name2 = id2name2_test[item]
        if name1 != name2:
            coun += 1
            print(name1 + '\t' + name2)
        overallscores.append(Levenshtein.distance(name1, name2))
print(coun)

print(np.array(overallscores))
print('ave: ' + str(np.average(np.array(overallscores))))
print('median: ' + str(np.median(np.array(overallscores))))
print('percentile 90: ' + str(np.percentile(np.array(overallscores), 90)))
print('percentile 10: ' + str(np.percentile(np.array(overallscores), 10)))

# overallscores = []
# coun = 0
# for item in range(lowbound, highbound):
#     # print(item)
#     name1 = id2name1_test[item]
#     name2 = id2name2_test[item]
#     overallscores.append(Levenshtein.distance(name1, name2))

# pickle.dump(overallscores, open(Config.store  + '/pos_distances.pkl', 'wb'))
# overallscores = pickle.load(open(Config.store   + '/pos_distances.pkl', 'rb'))
# num_Count = Counter(overallscores)
# print(num_Count)
# x1 = list(num_Count.keys())
# x1new = []
# for it in x1:
#     if it <= 15:
#         x1new.append(it)
# x1 = x1new
# x1.sort()
# y1 = [num_Count[item]/np.sum(np.array(list(num_Count.values()))) *100 for item in x1]
#
# import matplotlib.pyplot as plt
# fig = plt.figure()#figsize=(7,7)
# plt.plot(x1, y1)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.xlabel('Levenshtein distance', fontsize=15)
# plt.ylabel('Percentage', fontsize=15)
# # plt.legend(loc=0)
# plt.show()
# fig.savefig('data/wd_imdb/dbpfb.pdf', format='pdf')




# overallscores = []
# for item in range(lowbound, highbound):
#     print(item)
#     # print(item)
#     name1 = id2name1_test[item]
#     scores = []
#     for item1 in range(lowbound, item):
#         name2 = id2name2_test[item1]
#         overallscores.append(Levenshtein.distance(name1, name2))
#
#     for item1 in range(item+1, highbound):
#         name2 = id2name2_test[item1]
#         overallscores.append(Levenshtein.distance(name1, name2))
#     # break
#
# pickle.dump(overallscores, open(Config.store  + '/neg_distances.pkl', 'wb'))
#
# overallscores = pickle.load(open(Config.store   + '/neg_distances.pkl', 'rb'))
# num_Count = Counter(overallscores)
# print(num_Count)
# x1 = list(num_Count.keys())
# x1.sort()
# y1 = [num_Count[item] / 10499 for item in x1]
# import matplotlib.pyplot as plt
# plt.plot(x1, y1)
# plt.xlabel('Levenshtein distance')
# plt.ylabel('Number')
# # plt.legend(loc=0)
# plt.show()


# # lans = ['zh_en', 'ja_en', 'fr_en', 'en_fr', 'en_de', 'dbp_yg', 'dbp_wd']
# # labels = ['ZH_EN', 'JA_EN', 'FR_EN', 'EN_FR', 'EN_DE', 'DBP_YG', 'DBP_WD']
#
# # lans = ['dbp_wd', 'dbp_yg']
# # labels = ['DBP_WD', 'DBP_YG']
#
# lans = ['wd_imdb', 'wd_imdb/Title', 'dbp_wd', ]
# labels = ['WD_IMDB', 'WD_IMDB (film title)', 'DBP_WD', ]
# markers = ['^','^' ,'s',]
#
# # markers = ['^', '^','^','s','s','s','s']
# # colors = [(254/255,67/255,101/255), (252/255,157/255,154/255), (249/255,205/255,173/255), (200/255,200/255,169/255), (131/255,175/255,155/255), '#00CED1', 'b']
# xxx = []
# yyy = []
# for lan in lans:
#     overallscores = pickle.load(open('data/' + lan   + '/pos_distances.pkl', 'rb'))
#     num_Count = Counter(overallscores)
#     print(num_Count)
#     x1 = list(num_Count.keys())
#     x1new = []
#     for it in x1:
#         if it <= 15:
#             x1new.append(it)
#     x1new.sort()
#     y1 = [num_Count[item] / np.sum(np.array(list(num_Count.values()))) *100 for item in x1new]
#     xxx.append(x1new)
#     yyy.append(y1)
#
# fig = plt.figure()#figsize=(20,8)
# for i in range(len(xxx)):
#     plt.plot(xxx[i], yyy[i], label=labels[i], marker =markers[i])
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.xlabel('Levenshtein distance', fontsize=15)
# plt.ylabel('Percentage', fontsize=15)
# plt.legend(fontsize=12)
# plt.show()
# # fig.savefig('111.pdf', format='pdf')
# fig.savefig('data/wd_imdb/all.pdf', format='pdf')