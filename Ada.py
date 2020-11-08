import tensorflow as tf
from include.ModelOri import build_SE, training
from include.Load import *
import warnings
import json
import scipy
from scipy import spatial
import time
import copy
from utils import *
import argparse
import os
from sklearn import preprocessing

warnings.filterwarnings("ignore")

'''
Follow the code style of GCN-Align:
https://github.com/1049451037/GCN-Align
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


seed = 12306
np.random.seed(seed)
tf.set_random_seed(seed)

def make_print_to_file(fileName, path='./'):
    import sys
    import os
    import sys
    import datetime

    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8',)

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    sys.stdout = Logger(fileName + '.log', path=path)
    print(fileName.center(60,'*'))

def getsim_matrix(se_vec, test_pair, method):
    if method == 'cosine':
        se_vec = preprocessing.normalize(se_vec)
        Lvec = np.array([se_vec[e1] for e1, e2 in test_pair])
        Rvec = np.array([se_vec[e2] for e1, e2 in test_pair])
        aep = 1 - np.matmul(Lvec, Rvec.T)
    else:
        Lvec = np.array([se_vec[e1] for e1, e2 in test_pair])
        Rvec = np.array([se_vec[e2] for e1, e2 in test_pair])
        aep = scipy.spatial.distance.cdist(Lvec, Rvec, metric=method)

    if method == 'cityblock': # !!! notice that it is the max~
        aep = aep / aep.max()
    return aep

def get_hits_ma(sim, top_k=(1, 10)):
    pairs = []
    top_lr = [0] * len(top_k)
    mrr_sum_l = 0
    coun1 = 0
    for i in range(sim.shape[0]):
        # print(sim[i, :].argmin())
        pairs.append([i, sim[i, :].argmin()])
        rank = sim[i, :].argsort()
        # print(sim[i, :])
        rank_index = np.where(rank == i)[0][0]
        if rank_index == 0: coun1 += 1
        mrr_sum_l = mrr_sum_l + 1.0 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    msg = 'Hits@1:%.3f, Hits@10:%.3f, MRR:%.3f' % (top_lr[0] / len(sim), top_lr[1] / len(sim), mrr_sum_l / len(sim))
    print(msg)
    print(coun1)
    return pairs

def get_hits_ma_rec(sim, ouf, top_k=(1, 10)):
    top_lr = [0] * len(top_k)
    mrr_sum_l = 0
    coun1 = 0
    for i in range(sim.shape[0]):
        rank = sim[i, :].argsort()
        # print(sim[i, :])
        rank_index = np.where(rank == i)[0][0]
        if rank_index == 0:
            coun1 += 1
        else:
            ouf.write(str(i) + '\t' + str(np.argmin(sim[i, :])+10500) +  '\n')
        mrr_sum_l = mrr_sum_l + 1.0 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    msg = 'Hits@1:%.3f, Hits@10:%.3f, MRR:%.3f' % (top_lr[0] / len(sim), top_lr[1] / len(sim), mrr_sum_l / len(sim))
    print(msg)
    print(coun1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lan", type=str, default="dbp_fb", help="which dataset?") #ja_en en_fr

    args = parser.parse_args()

    class Config:
        # language = 'dbp_wd_15k_V1' # dbp_wd_15k_V1 | en_fr_15k_V1 | wk3l_60k/en_de
        language = args.lan  # zh_en | ja_en | fr_en ||en_fr _15k_V1　en_de _15k_V1 dbp_wd _15k_V1 dbp_yg _15k_V1 wd_imdb dbp_fb
        e1 = 'data/' + language + '/ent_ids_1'
        # e2name = 'data/' + language + '/ent_ids_2_golden'
        e2 = 'data/' + language + '/ent_ids_2'
        kg1 = 'data/' + language + '/triples_1'
        kg2 = 'data/' + language + '/triples_2'
        ill = 'data/' + language + '/ref_ent_ids'
        store = 'data/' + language + '/'

        epochs = 300
        dim = 300
        se_dim = 300
        act_func = tf.nn.relu
        alpha = 0.1
        beta = 0.3
        # gamma = 1.0  # margin based loss
        gamma = 3.0  # margin based loss
        # k = 125  # number of negative samples for each positive one
        k = 25
        # seed = 3  # 30% of seeds
        seed = 0.3
        epochs_se = 300
        beta = 0.3

    make_print_to_file('logs/' + Config.language, path='./')

    t = time.time()

    e = len(set(loadfile(Config.e1, 1)) | set(loadfile(Config.e2, 1)))
    ILL = loadfile(Config.ill, 2)
    illL = len(ILL)
    testnum = 10500
    nelinenum = 10500
    valinum = 900
    if Config.language == 'dbp_fb':
        testnum = 17880
        nelinenum = 25542
        valinum = 1532
    test = ILL[:testnum]
    vali = ILL[testnum:testnum+valinum]
    train = ILL[testnum+valinum:]
    train_array = np.array(train)

    KG1 = loadfile(Config.kg1, 3)
    KG2 = loadfile(Config.kg2, 3)

    M1 = np.zeros((testnum, testnum))
    M2 = np.zeros((testnum, testnum))
    for item in KG1:
        if item[0] < testnum and item[2] < testnum:
            M1[item[0], item[2]] = 1

    for item in KG2:
        if item[0] - nelinenum < testnum and item[2] - nelinenum < testnum:
            M2[item[0] - nelinenum, item[2] - nelinenum] = 1

    M1_vali = np.zeros((valinum, valinum))
    M2_vali = np.zeros((valinum, valinum))
    for item in KG1:
        if item[0] < testnum + valinum and item[2] < testnum + valinum and item[0] > testnum and item[2] > testnum:
            M1_vali[item[0]-testnum, item[2]-testnum] = 1

    for item in KG2:
        if item[0] - nelinenum < testnum+ valinum and item[2] - nelinenum < testnum+ valinum and item[0] - nelinenum> testnum and item[2] - nelinenum> testnum:
            M2_vali[item[0] - testnum - nelinenum, item[2] - testnum - nelinenum] = 1

    method = "braycurtis" # cosine,braycurtis,cityblock euclidean\
    ###########################################structure############################################
    # # output_layer, loss = build_SE(Config.se_dim, Config.act_func, Config.gamma, Config.k, e, train_array, KG1 + KG2)
    # # se_vec, J = training(output_layer, loss, 25, Config.epochs_se, train_array, e, Config.k)
    # # np.save('./data/' + Config.language + '/se_vec_test.npy', se_vec)

    # se_vec = np.load('./data/' + Config.language + '/se_vec_test.npy')
    # # se_vec = preprocessing.normalize(se_vec)
    # aep = getsim_matrix(se_vec, test, method)
    # np.save('./data/' + Config.language + '/' + 'test' + '/stru_mat_test-' + method + '.npy', aep)
    # aep_vali = getsim_matrix(se_vec, vali, method)
    # np.save('./data/' + Config.language + '/' + 'vali' + '/stru_mat_vali-' + method + '.npy', aep_vali)

    aep = np.load('./data/' + Config.language + '/' + 'test' + '/stru_mat_test-' + method + '.npy')
    # aep = preprocessing.normalize(aep)
    print(aep)
    aep_vali = np.load('./data/' + Config.language + '/' + 'vali' + '/stru_mat_vali-' + method + '.npy')

    # ###########################################semantic############################################
    # if  Config.language == 'en_fr' or Config.language == 'en_de':
    #     nepath = './data/' + Config.language + '/name_vec_cpm_1.txt'
    #     ne_vec = loadNe(nepath)
    #     ne_vecold = copy.deepcopy(ne_vec)
    # elif Config.language == 'dbp_wd' or Config.language == 'dbp_yg' or Config.language == 'dbp_fb':
    #     nepath = './data/' + Config.language + '/name_vec_ftext.txt'
    #     ne_vec = loadNe(nepath)
    #     ne_vecold = copy.deepcopy(ne_vec)
    # else:
    #     with open(file='./data/' + Config.language + '/' + Config.language.split('_')[0] + '_vectorList.json',
    #               mode='r', encoding='utf-8') as f:
    #         embedding_list = json.load(f)
    #         print(len(embedding_list), 'rows,', len(embedding_list[0]), 'columns.')
    #         ne_vec = np.array(embedding_list)
    #
    # ne_vec = preprocessing.normalize(ne_vec)
    # aep_n = getsim_matrix(ne_vec, test, method)
    #
    # # if Config.language == 'en_fr' or Config.language == 'en_de':
    # #     aep_nnew = copy.deepcopy(aep_n)
    # #     counter = 0; totoal = 0
    # #     for e1, e2 in test:
    # #         if sum(ne_vecold[e1]) == 0:
    # #             totoal +=1
    # #             aep_nnew[counter] = np.ones(len(aep_n))
    # #         if sum(ne_vecold[e2]) == 0:
    # #             aep_nnew[:, counter] = np.ones(len(aep_n))
    # #         counter += 1
    # #     print(totoal)
    # #     aep_n = aep_nnew
    # np.save('./data/' + Config.language + '/' + 'test' + '/name_mat_test-' + method + '.npy', aep_n)
    #
    # aep_n_vali = getsim_matrix(ne_vec, vali, method)
    # # if Config.language == 'en_fr' or Config.language == 'en_de':
    # #     aep_n_valinew = copy.deepcopy(aep_n_vali)
    # #     counter = 0; totoal = 0
    # #     for e1, e2 in vali:
    # #         if sum(ne_vecold[e1]) == 0:
    # #             totoal +=1
    # #             aep_n_valinew[counter] = np.ones(len(aep_n_vali))
    # #         if sum(ne_vecold[e2]) == 0:
    # #             aep_n_valinew[:, counter] = np.ones(len(aep_n_vali))
    # #         counter += 1
    # #     print(totoal)
    # #     aep_n_vali = aep_n_valinew
    # np.save('./data/' + Config.language + '/' + 'vali' + '/name_mat_vali-' + method + '.npy', aep_n_vali)

    # if '_V1' in Config.language:
    #     if method == 'braycurtis':
    #         aep_n = np.load('./data/' + Config.language + '/name_mat_train_new-' + method + '.npy')
    #     else:
    #         aep_n = np.load('./data/' + Config.language + '/name_mat_train-' + method + '.npy')
    #
    # else:
    aep_n = np.load('./data/' + Config.language + '/' + 'test' + '/name_mat_test-' + method + '.npy')
    # aep_n = preprocessing.normalize(aep_n)
    print(aep_n)

    aep_n_vali = np.load('./data/' + Config.language + '/' + 'vali' + '/name_mat_vali-' + method + '.npy')
    # print(aep_n)

    ###########################################string############################################
    strsim = np.load('./data/' + Config.language + '/string_mat_train.npy')
    str_sim_vali = np.load('./data/' + Config.language + '/string_mat_vali.npy')
    # print(len(strsim))
    # print(str_sim)

    strsim = 1 - strsim
    str_sim = strsim[:testnum, :testnum]
    # str_sim = preprocessing.normalize(str_sim)
    str_sim_vali = 1 - str_sim_vali


    #######################switch between validation and testing!!
    directiory = 'test' # vali test
    print('***************************' + directiory + '***************************')
    if directiory == 'vali':
        aep = aep_vali
        aep_n = aep_n_vali
        str_sim = str_sim_vali
        M1 = M1_vali
        M2 = M2_vali
    get_hits_ma(aep)
    get_hits_ma(aep_n)
    get_hits_ma(str_sim)


    # for Learning-to-rank training! validation is used for training here, considering the overfitting of stru on the original training set!!!
    #
    # ouf = open('./lambdamart/traindbpfb.txt', 'w')
    # negnum = 10
    # for i in range(len(aep)):
    #     pos_stru = aep[i][i]
    #     pos_name = aep_n[i][i]
    #     pos_string = str_sim[i][i]
    #     ouf.write('1 qid:' + str(i) + ' 1:' + str(pos_stru)+ ' 2:' + str(pos_name)+ ' 3:' + str(pos_string) + '\n')
    #     rank_stru = aep[i, :].argsort()[:10]
    #     rank_name = aep_n[i, :].argsort()[:10]
    #     rank_string = str_sim[i, :].argsort()[:10]
    #     total_negs = np.concatenate([rank_stru, rank_name, rank_string])
    #     total_negs = np.unique(total_negs)
    #     total_negs = total_negs[total_negs!=i]
    #     # if i in total_negs:
    #     #     np.delete(total_negs, i)
    #
    #     # a = np.arange(len(aep))
    #     # np.delete(a, i)
    #     # negs = np.random.choice(a, size=negnum, replace=False)
    #     for neg in total_negs:
    #         neg_stru = aep[i][neg]
    #         neg_name = aep_n[i][neg]
    #         neg_string = str_sim[i][neg]
    #         ouf.write(
    #             '0 qid:' + str(i) + ' 1:' + str(neg_stru) + ' 2:' + str(neg_name) + ' 3:' + str(neg_string) + '\n')

    # ouf = open('./lambdamart/testzhen.txt', 'w')
    # # recall = 0
    # for i in range(len(aep)):
    #     pos_stru = aep[i][i]
    #     pos_name = aep_n[i][i]
    #     pos_string = str_sim[i][i]
    #     ouf.write('1 qid:' + str(i) + ' 1:' + str(pos_stru) + ' 2:' + str(pos_name) + ' 3:' + str(pos_string) + '\n')
    #     # a = np.arange(len(aep))
    #     # np.delete(a, i)
    #     # negs = np.random.choice(a, size=negnum, replace=False)
    #     rank_stru = aep[i, :].argsort()[:10]
    #     rank_name = aep_n[i, :].argsort()[:10]
    #     rank_string = str_sim[i, :].argsort()[:10]
    #     total_negs = np.concatenate([rank_stru, rank_name, rank_string])
    #     total_negs = np.unique(total_negs)
    #     total_negs = total_negs[total_negs != i]
    #     # if i in total_negs:
    #     #     np.delete(total_negs, i)
    #     for neg in total_negs:
    #         neg_stru = aep[i][neg]
    #         neg_name = aep_n[i][neg]
    #         neg_string = str_sim[i][neg]
    #         ouf.write(
    #             '0 qid:' + str(i) + ' 1:' + str(neg_stru) + ' 2:' + str(neg_name) + ' 3:' + str(neg_string) + '\n')


    # # ###########################################calculate weight + fusion############################################
    # weight_stru = 0.3
    # weight_text = 0.3
    # weight_string = 0.3
    theta1 = 0.0 # gap

    # theta_1 = 0.05
    # theta_2 = 0.1
    theta_1 = 0
    theta_2 = 0.5
    import pickle

    aep_r = aep.T
    aep_n_r = aep_n.T
    str_sim_r = str_sim.T

    # confi_stru, top_scores_stru = obtain_confi(aep,aep_r,theta1)
    # confi_text, top_scores_text = obtain_confi(aep_n,aep_n_r,theta1)
    # confi_string, top_scores_string = obtain_confi(str_sim,str_sim_r,theta1)
    #
    # outf1 = open(Config.store + directiory  + '/confi_stru-' + method +'.pkl', 'wb')
    # pickle.dump(confi_stru, outf1)
    # outf1 = open(Config.store + directiory  + '/top_scores_stru-' + method +'.pkl', 'wb')
    # pickle.dump(top_scores_stru, outf1)
    #
    # outf1 = open(Config.store + directiory  + '/confi_name-' + method +'.pkl', 'wb')
    # pickle.dump(confi_text, outf1)
    # outf1 = open(Config.store + directiory  + '/top_scores_name-' + method +'.pkl', 'wb')
    # pickle.dump(top_scores_text, outf1)
    #
    # outf1 = open(Config.store + directiory  + '/confi_string-' + method +'.pkl', 'wb')
    # pickle.dump(confi_string, outf1)
    # outf1 = open(Config.store + directiory  + '/top_scores_string-' + method +'.pkl', 'wb')
    # pickle.dump(top_scores_string, outf1)

    outf1 = open(Config.store + directiory  + '/confi_stru-' + method +'.pkl', 'rb')
    confi_stru = pickle.load(outf1)
    outf1 = open(Config.store + directiory  + '/top_scores_stru-' + method +'.pkl', 'rb')
    top_scores_stru = pickle.load(outf1)

    outf1 = open(Config.store + directiory  + '/confi_name-' + method +'.pkl', 'rb')
    confi_text = pickle.load(outf1)
    outf1 = open(Config.store + directiory  + '/top_scores_name-' + method +'.pkl', 'rb')
    top_scores_text = pickle.load(outf1)

    outf1 = open(Config.store + directiory  + '/confi_string-' + method +'.pkl', 'rb')
    confi_string = pickle.load(outf1)
    outf1 = open(Config.store + directiory  + '/top_scores_string-' + method +'.pkl', 'rb')
    top_scores_string = pickle.load(outf1)

    weight_stru, weight_text, weight_string = cal_weight(theta_1, theta_2, confi_stru, confi_text, confi_string,
                                                         top_scores_stru, top_scores_text, top_scores_string)
    aep_fuse = (aep * weight_stru + aep_n * weight_text + str_sim * weight_string)

    # weight_stru, weight_text = cal_weight_2(theta_1, theta_2, confi_stru, confi_text, top_scores_stru, top_scores_text)
    # aep_fuse = (aep * weight_stru + aep_n * weight_text)

    # weight_stru, weight_text = cal_weight_2(theta_1, theta_2, confi_stru, confi_string, top_scores_stru, top_scores_string)
    # aep_fuse = (aep * weight_stru + str_sim * weight_text)

    # weight_stru, weight_text = cal_weight_2(theta_1, theta_2, confi_text, confi_string, top_scores_text, top_scores_string)
    # aep_fuse = (aep_n * weight_stru + str_sim * weight_text)

    aep_fuse_ori = copy.deepcopy(aep_fuse)
    aep_fuse_r = aep_fuse.T
    pairs = get_hits_ma(aep_fuse)

    # without collective analysis
    # ouf = open(Config.store + 'results-woc.txt', 'w')
    # for item in pairs:
    #     ouf.write(str([item[0]]) + '\t' + str([item[1]]) + '\n')

    # outf = open('withoutl.txt', 'w')
    # get_hits_ma_rec(aep_fuse, outf)

    # print()
    # # iterative to obtain the results
    # # iterative(vali, aep_fuse)

    if directiory == 'test':
        dic_row = {i: i for i in range(len(test))}
        dic_col = {i: i for i in range(len(test))}
    else:
        dic_row = {i: i + testnum for i in range(len(vali))}
        dic_col = {i: i + testnum for i in range(len(vali))}

    # determine the results iteratively...
    total_matched = 0
    total_true = 0
    aep_fuse_new = copy.deepcopy(aep_fuse)
    aep_fuse_r_new = copy.deepcopy(aep_fuse_r)
    matchedpairs = {}

    # while total_matched < len(test):
    for _ in range(2):
        aep_fuse_new, aep_fuse_r_new, matched, matched_true, dic_row, dic_col, M1, M2, mmtached \
            = ite1(aep_fuse_new, aep_fuse_r_new, dic_row, dic_col,M1,M2,matchedpairs)
        print(len(matchedpairs))
        total_matched += matched
        total_true += matched_true
        print('Total Match ' + str(total_matched))
        print('Total Match True ' + str(total_true))
        if matched == 0: break
    # print(float(total_true)/10500)
    print("End of pre-treatment...\n")

    print("total time elapsed: {:.4f} s".format(time.time() - t))

    new2old_row, new2old_col = dic_row, dic_col
    aep_fuse_new = 1 - aep_fuse_new

    leftids, rightids, newindex, cans, scores = evarerank(aep_fuse_new, new2old_row, new2old_col)

    leftids = np.array(leftids)
    rightids = np.array(rightids)
    # print(leftids)
    # print(rightids)
    # print(M1)
    # print(M2)

    directiory = 'test'

    fillup = '-2'
    pickle.dump(matchedpairs, open('./data/' + Config.language + '/' + directiory  + '/matchedp-iterl' + fillup + '-' + method + '.pkl', 'wb'))
    # np.save('./data/' + Config.language + '/' + directiory  + '/m1-iterl' + fillup + '-' + method + '.npy', M1)
    # np.save('./data/' + Config.language + '/' + directiory  + '/m2-iterl' + fillup + '-' + method + '.npy', M2)
    np.save('./data/' + Config.language + '/' + directiory  + '/newindex-iterl' + fillup + '-' + method + '.npy', newindex)
    np.save('./data/' + Config.language + '/' + directiory  + '/cans-iterl' + fillup + '-' + method + '.npy', cans)
    np.save('./data/' + Config.language + '/' + directiory  + '/scores-iterl' + fillup + '-' + method + '.npy', scores)
    np.save('./data/' + Config.language + '/' + directiory  + '/leftids-iterl' + fillup + '-' + method + '.npy', leftids)
    np.save('./data/' + Config.language + '/' + directiory  + '/rightids-iterl' + fillup + '-' + method + '.npy', rightids)








