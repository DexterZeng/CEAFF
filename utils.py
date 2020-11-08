import copy
import numpy as np

def obtain_confi(aep, aep_r,theta1):
    aep_rank = copy.deepcopy(aep)
    results_stru = aep_rank.argsort(axis = 1)[:,0]
    print(results_stru)
    aep_rank.sort(axis = 1)
    top_scores_stru = aep_rank[:,0]
    second_top_scores_stru = aep_rank[:, 1]
    print(top_scores_stru)
    left2right = {i: results_stru[i] for i in range(len(results_stru))}

    aep_rank_r = copy.deepcopy(aep_r)
    results_stru_r = aep_rank_r.argsort(axis = 1)[:,0]
    print(results_stru_r)
    aep_rank_r.sort(axis = 1)
    top_scores_stru_r = aep_rank_r[:,0]
    second_top_scores_stru_r = aep_rank[:, 1]
    print(top_scores_stru_r)
    right2left = {i: results_stru_r[i] for i in range(len(results_stru_r))}

    confident = dict()
    #allconfi = 0
    for item in left2right:
        if right2left[left2right[item]] == item:
            if second_top_scores_stru[item] - top_scores_stru[item] >= theta1:
                if second_top_scores_stru_r[left2right[item]] - top_scores_stru_r[left2right[item]] >= theta1:
            #allconfi += 1
            #if top_scores_stru[item] != 1.0:
                    confident[item] = left2right[item]
    #print(confident)
    #print('Real Confi, including 1: ' + str(allconfi))
    print('Confi: ' + str(len(confident)))
    correct = 0
    for i in confident:
        if confident[i] == i:
            correct += 1
    print('Correct: ' + str(correct))
    return confident, top_scores_stru


def macthedbynew(item, confi, confi_1, confi_2):
    a = 0.333
    b = 0.5
    if item in confi_1 and item in confi_2:
        if confi[item] == confi_1[item] == confi_2[item]:
            weight = a
        elif confi[item] == confi_1[item] or confi[item] == confi_2[item]:
            weight = b
        else:
            weight = 1
    elif item in confi_1 and item not in confi_2:
        if confi[item] == confi_1[item]:
            weight = b
        else:
            weight = 1
    elif item in confi_2 and item not in confi_1:
        if confi[item] == confi_2[item]:
            weight = b
        else:
            weight = 1
    else:
        weight = 1
    return weight


def cal_weight(theta_1, theta_2, confi_stru, confi_text, confi_string, top_scores_stru, top_scores_text,
               top_scores_string):
    weight_stru = 0
    coun1 = 0
    for item in confi_stru:
        if top_scores_stru[item] > theta_1:
            weight_stru += macthedbynew(item, confi_stru, confi_text, confi_string)
        else:
            #if weight_stru!=0:
                weight_stru += theta_2
                coun1 += 1
    print('Stru Unqualified: ' + str(coun1))

    coun2 = 0
    weight_text = 0
    for item in confi_text:
        if top_scores_text[item] > theta_1:
            weight_text += macthedbynew(item, confi_text, confi_stru, confi_string)
        else:
            #if weight_text!=0:
                weight_text += theta_2
                coun2 += 1
    print('Text Unqualified: ' + str(coun2))

    coun3 = 0
    weight_string = 0
    for item in confi_string:
        if top_scores_string[item] > theta_1:
            weight_string += macthedbynew(item, confi_string, confi_stru, confi_text)
        else:
            #if weight_string!=0:
                weight_string += theta_2
                coun3 += 1
    print('String Unqualified: ' + str(coun3))

    # if '_V1' in Config.language:
    #     weight_text = weight_text*0.8#(10500.0-1343.0*2)/10500.0

    print('stru w: ' + str(weight_stru))
    print('text w: ' + str(weight_text))
    print('string w: ' + str(weight_string))
    print(len(confi_string))
    weight_stru = float(weight_stru) /len(confi_stru)
    weight_text = float(weight_text) / len(confi_text)
    weight_string = float(weight_string) / len(confi_string)

    w_s = float(weight_stru) / (weight_stru + weight_text + weight_string)
    w_t = float(weight_text) / (weight_stru + weight_text + weight_string)
    w_st = float(weight_string) / (weight_stru + weight_text + weight_string)

    print('stru w: ' + str(w_s))
    print('text w: ' + str(w_t))
    print('string w: ' + str(w_st))

    return w_s, w_t, w_st


def macthedby2(item, confi, confi_1):
    if item in confi_1:
        if confi[item] == confi_1[item]:
            weight = 0.5
        else:
            weight = 1
    else:
        weight = 1
    return weight

def cal_weight_2(theta_1, theta_2, confi_stru, confi_text, top_scores_stru, top_scores_text):
    wei1 = 0
    wei2 = 0
    coun1 = 0
    for item in confi_stru:
        if top_scores_stru[item] > theta_1:
            wei1 += macthedby2(item, confi_stru, confi_text)
        else:
            wei1 += theta_2
            coun1 += 1
    print('Stru Unqualified: ' + str(coun1))
    #print(wei1)

    coun1 = 0
    for item in confi_text:
        if top_scores_text[item] > theta_1:
            wei2 += macthedby2(item, confi_text, confi_stru)
        else:
            wei2 += theta_2
            coun1 += 1
    print('Text Unqualified: ' + str(coun1))
    #print(wei2)
    #name

    # if '_V1' in language:
    #     wei2 = wei2*0.8#(10500.0-1343.0*2)/10500.0
    wei1 = wei1/float(len(confi_stru))
    wei2 = wei2/float(len(confi_text))
    w_s = float(wei1)/(wei1 + wei2)
    w_t = float(wei2)/(wei1 + wei2)


    print('w1: ' + str(w_s))
    print('w2: ' + str(w_t))

    return w_s, w_t

def ite(aep_fuse, aep_fuse_r, dic_row, dic_col, counn):
    aep_fuse_rank = copy.deepcopy(aep_fuse)
    aep_fuse_rank = aep_fuse_rank
    results_stru = aep_fuse_rank.argsort(axis = 1)[:,0]
    #print(results_stru)
    aep_fuse_rank.sort(axis = 1)
    top_scores_stru = aep_fuse_rank[:,0]
    # print(top_scores_stru)
    #second_top_scores_stru = aep_fuse_rank[:, 1]
    #print(top_scores_stru)
    left2right = {i: results_stru[i] for i in range(len(results_stru))}

    aep_fuse_rank_r = copy.deepcopy(aep_fuse_r)
    aep_fuse_rank_r = aep_fuse_rank_r
    results_stru_r = aep_fuse_rank_r.argsort(axis = 1)[:,0]
    #print(results_stru_r)
    aep_fuse_rank_r.sort(axis = 1)
    top_scores_stru_r = aep_fuse_rank_r[:,0]
    #second_top_scores_stru_r = aep_fuse_rank_r[:, 1]
    #print(top_scores_stru_r)
    right2left = {i: results_stru_r[i] for i in range(len(results_stru_r))}

    if counn ==0:
        thetaa = 0.1
    elif counn ==1:
        thetaa = 0.25
    elif counn == 2:
        thetaa = 0.5 # fr 0.3
    else:
        thetaa = 1
    confident = dict()
    row = []
    col = []
    for item in left2right:
        if right2left[left2right[item]] == item:
            if top_scores_stru[item] <= thetaa:  # 0.4->0
            # print(top_scores_stru[item] - second_top_scores_stru[item])
            # if top_scores_stru[item] - second_top_scores_stru[item] >= theta1:
            #     if top_scores_stru_r[left2right[item]] - second_top_scores_stru_r[left2right[item]] >= theta1:
                    confident[item] = left2right[item]
                    row.append(item)
                    col.append(left2right[item])
    #print(confident)
    print('Confi in fuse: ' + str(len(confident)))
    correct = 0
    for i in confident:
        if dic_col[confident[i]] == dic_row[i]:
            correct += 1
    print('Correct in fuse: ' + str(correct))

    # after removal, need to define a mapping function to map column/rows indexes to the origional indexes
    newind_row = 0
    new2old_row = dict()
    newind_col = 0
    new2old_col = dict()
    for item in range(aep_fuse.shape[0]):
        if item not in row:
            new2old_row[newind_row] = dic_row[item] # dic_row item not just item # item is one-hop map while dic_row...
            newind_row += 1
        if item not in col:
            new2old_col[newind_col] = dic_col[item]
            newind_col += 1
    #print(len(new2old_row))
    #print(len(new2old_col))

    aep_fuse_new = np.delete(aep_fuse, row, axis=0)
    aep_fuse_new = np.delete(aep_fuse_new, col, axis=1)
    aep_fuse_r_new = aep_fuse_new.T

    return aep_fuse_new, aep_fuse_r_new, len(confident), correct, new2old_row, new2old_col

def iterative(vali, aep_fuse):
    dic_row = {i: i for i in range(len(vali))}
    dic_col = {i: i for i in range(len(vali))}
    total_matched = 0
    total_true = 0
    aep_fuse_new = copy.deepcopy(aep_fuse)
    aep_fuse_r_new = copy.deepcopy(aep_fuse.T)
    # for _ in range(2):
    counn = 0
    while total_matched < len(vali):
        aep_fuse_new, aep_fuse_r_new, matched, matched_true, dic_row, dic_col \
            = ite(aep_fuse_new, aep_fuse_r_new, dic_row, dic_col, counn)
        total_matched += matched
        total_true += matched_true
        print('Total Match ' + str(total_matched))
        print('Total Match True ' + str(total_true))
        if matched == 0: break
        # print(dic_row)
        counn += 1
        if counn == 3:
            # left = list(dic_row.values())
            # results = aep_fuse_ori.argsort(axis=1)[:, 0]
            # correct = 0
            # for i in left:
            #     rightid = results[i]
            #     if i == rightid:
            #         correct += 1
            # print("maximum strategy " + str(correct))
            # print()
            aep_fuse = aep_fuse_new
            print(aep_fuse.shape)
            aep_rank = copy.deepcopy(aep_fuse)
            results = aep_rank.argsort(axis=1)[:, 0]
            correct = 0
            leftids = []
            for i in range(len(dic_row)):
                leftid = dic_row[i]
                leftids.append(leftid)
                rightid = dic_col[results[i]]
                if leftid == rightid:
                    correct += 1
            print("maximum strategy " + str(correct))
            print()
    print(float(total_true) / len(aep_fuse))


def ite1(aep_fuse, aep_fuse_r, dic_row, dic_col,M1,M2,mmtached):
    aep_fuse_rank = copy.deepcopy(aep_fuse)
    aep_fuse_rank = aep_fuse_rank
    results_stru = aep_fuse_rank.argsort(axis = 1)[:,0]
    #print(results_stru)
    aep_fuse_rank.sort(axis = 1)
    top_scores_stru = aep_fuse_rank[:,0]
    #second_top_scores_stru = aep_fuse_rank[:, 1]
    #print(top_scores_stru)
    left2right = {i: results_stru[i] for i in range(len(results_stru))}

    aep_fuse_rank_r = copy.deepcopy(aep_fuse_r)
    aep_fuse_rank_r = aep_fuse_rank_r
    results_stru_r = aep_fuse_rank_r.argsort(axis = 1)[:,0]
    #print(results_stru_r)
    aep_fuse_rank_r.sort(axis = 1)
    top_scores_stru_r = aep_fuse_rank_r[:,0]
    #second_top_scores_stru_r = aep_fuse_rank_r[:, 1]
    #print(top_scores_stru_r)
    right2left = {i: results_stru_r[i] for i in range(len(results_stru_r))}

    theta1 = 0.00
    confident = dict()
    row = []
    col = []
    for item in left2right:
        if right2left[left2right[item]] == item:
            # if top_scores_stru[item] > 0:  # 0.4->0
            # print(top_scores_stru[item] - second_top_scores_stru[item])
            # if top_scores_stru[item] - second_top_scores_stru[item] >= theta1:
            #     if top_scores_stru_r[left2right[item]] - second_top_scores_stru_r[left2right[item]] >= theta1:
                    confident[item] = left2right[item]
                    row.append(item)
                    col.append(left2right[item])
    #print(confident)
    print('Confi in fuse: ' + str(len(confident)))
    correct = 0
    for i in confident:
        mmtached[dic_row[i]]=dic_col[confident[i]]
        if dic_col[confident[i]] == dic_row[i]:
            correct += 1
    print('Correct in fuse: ' + str(correct))

    # after removal, need to define a mapping function to map column/rows indexes to the origional indexes
    newind_row = 0
    new2old_row = dict()
    newind_col = 0
    new2old_col = dict()
    for item in range(aep_fuse.shape[0]):
        if item not in row:
            new2old_row[newind_row] = dic_row[item] # dic_row item not just item # item is one-hop map while dic_row...
            newind_row += 1
        if item not in col:
            new2old_col[newind_col] = dic_col[item]
            newind_col += 1
    #print(len(new2old_row))
    #print(len(new2old_col))

    aep_fuse_new = np.delete(aep_fuse, row, axis=0)
    aep_fuse_new = np.delete(aep_fuse_new, col, axis=1)
    aep_fuse_r_new = aep_fuse_new.T

    M1 = np.delete(M1, row, axis=0)
    M1 = np.delete(M1, row, axis=1)

    M2 = np.delete(M2, col, axis=0)
    M2 = np.delete(M2, col, axis=1)

    return aep_fuse_new, aep_fuse_r_new, len(confident), correct, new2old_row, new2old_col, M1,M2, mmtached

def evarerank(aep_fuse_new, new2old_row, new2old_col):
    ### reindex according to difficulty
    aep_fuse = aep_fuse_new
    # print(aep_fuse)
    print(aep_fuse.shape)
    aep_rank = copy.deepcopy(aep_fuse)
    aep_rank = -aep_rank

    results = aep_rank.argsort(axis=1)[:, 0]
    correct = 0
    leftids = []
    pairs = []
    for i in range(len(new2old_row)):
        leftid = new2old_row[i]
        leftids.append(leftid)
        rightid = new2old_col[results[i]]
        if leftid == rightid:
            correct += 1
        pairs.append([leftid, rightid])

    print("maximum strategy " + str(correct))
    print()
    # check multiple
    mul = dict()
    for item in pairs:
        if item[1] not in mul:
            mul[item[1]] = 1
        else:
            mul[item[1]] += 1

    multiple = []
    mulsource = 0
    for item in mul:
        if mul[item] > 1:
            multiple.append([item,mul[item]])
            mulsource += mul[item]
    print("multiple source entities are aligned to the same target entities!")
    print('source ' + str(mulsource))
    print('target ' + str(len(multiple)))
    print()


    rightids = []
    for i in range(len(new2old_col)):
        rightid = new2old_col[i]
        rightids.append(rightid)

    count = 0
    for left in leftids:
        if left in rightids:
            count += 1
    print("total " + str(count))

    top_ids = aep_rank.argsort(axis=1)[:, :10]  # top_ids seem to be the positions, instead of the actual ids...
    wrongsx = []
    wrongsy = []
    truePositions = []
    top10 = 0
    for ii in range(len(leftids)):
        pos = np.where(np.array(rightids) == leftids[ii])[0]
        if len(pos) > 0:
            pos = pos[0]
            truePositions.append(pos)
        else:
            pos = -1
        if pos in top_ids[ii]:
            top10 += 1
            if pos != top_ids[ii][0]:
                wrongsx.append(ii)
                wrongsy.append(pos)

    #print(truePositions)
    print("top10 " + str(top10))

    aep_rank.sort(axis=1)
    top_scores = -aep_rank[:, :10]
    scores = -aep_rank[:, 0]
    # truescores = aep_fuse[wrongsx, wrongsy]
    # gap = scores[wrongsx] - truescores
    # print(len(gap))
    # print(np.mean(gap))

    # print(scores)
    newindex = (-scores).argsort()
    # print(newindex)
    cans = top_ids[newindex, :]
    scores = top_scores[newindex, :]
    # print(cans)
    # print(scores)

    return leftids, rightids, newindex, cans, scores