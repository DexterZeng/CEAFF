import Levenshtein
import re
import numpy as np
from include.Load import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lan", type=str, default="zh_en", help="which dataset?")
    args = parser.parse_args()

    lan = args.lan

    testnum = 10500
    nelinenum = 10500
    valinum = 900
    if lan == 'dbp_fb':
        testnum = 17880
        nelinenum = 25542
        valinum = 1532

    ILL = loadfile('data/' + lan + '/ill_ent_ids', 2)
    illL = len(ILL)
    test = ILL[:testnum]
    test0 = [str(item[0]) for item in test]
    test1 = [str(item[1]) for item in test]

    vali = ILL[testnum:testnum+valinum]
    vali0 = [str(item[0]) for item in vali]
    vali1 = [str(item[1]) for item in vali]
    train = ILL[testnum+valinum:]

    inf1 = open('data/' + lan + '/ent_ids_1')
    id2name1 = dict()
    for i1, line in enumerate(inf1):
        strs = line.strip().split('\t')
        wordline = strs[1].split('/')[-1].lower().replace('(','').replace(')','')
        wordline = re.sub("[\s+\.\!\/_,$%^*_\-(+\"\')]+|[+—?【】“”！，。？、~@#￥%……&*（）]+'", "",wordline)
        id2name1[strs[0]] = wordline
    print(len(id2name1))

    if lan == 'dbp_fb':
        inf2 = open('data/' + lan + '/ent_ids_2_name')
    else:
        inf2 = open('data/' + lan + '/ent_ids_2')

    id2name2 = dict()
    for i1, line in enumerate(inf2):
        strs = line.strip().split('\t')
        wordline = strs[1].split('/')[-1].lower().replace('(','').replace(')','')
        wordline = re.sub("[\s+\.\!\/_,$%^*_\-(+\"\')]+|[+—?【】“”！，。？、~@#￥%……&*（）]+'", "",wordline)
        id2name2[strs[0]] = wordline
    print(len(id2name2))

    names1 = []
    for item in vali0:
        names1.append(id2name1[item])

    names2 = []
    for item in vali1:
        names2.append(id2name2[item])

    overallscores = []
    for item in names1:
        scores = []
        for item1 in names2:
            scores.append(Levenshtein.ratio(item, item1))
        overallscores.append(scores)

    # print(np.array(overallscores))
    np.save('./data/'+ lan + '/string_mat_vali.npy', np.array(overallscores))

    names1 = []
    for item in test0:
        names1.append(id2name1[item])

    names2 = []
    for item in test1:
        names2.append(id2name2[item])

    overallscores = []
    for item in names1:
        scores = []
        for item1 in names2:
            scores.append(Levenshtein.ratio(item, item1))
        overallscores.append(scores)

    # print(np.array(overallscores))
    np.save('./data/'+ lan + '/string_mat_test.npy', np.array(overallscores))