import tensorflow as tf
from include.Load import *
import warnings
import time
import argparse
import copy
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
seed = 12306
#np.random.seed(seed)
#tf.set_random_seed(seed)

def make_print_to_file(fileName, path='./'):
    '''
    path， it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    '''
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

    #fileName = datetime.datetime.now().strftime('day'+'%Y_%m_%d')
    sys.stdout = Logger(fileName + '.log', path=path)
    print(fileName.center(60,'*'))

class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        #self.s = tf.placeholder(tf.float32, [n_features, 3], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=n_features,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

            #self.acts_prob = tf.transpose(tf.nn.softmax(self.acts_prob, axis=0, name='act_prob'))

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        #print(probs)
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=n_features,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lan", type=str, default="zh_en", help="which dataset?")
    parser.add_argument("--round", type=int, default=1, help="multiple rounds of results")

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

    fillup = '-4'
    method = 'braycurtis' #euclidean braycurtis cityblock cosine

    make_print_to_file('logs/' + Config.language + 'lr', path='./')


    # M1 = np.load('./data/' + Config.language + '/m1-iterl' + fillup + '-'+ method +'.npy')
    # M2 = np.load('./data/' + Config.language + '/m2-iterl' + fillup + '-'+ method +'.npy')
    # norm = np.max(np.sum(M2, axis=-1).squeeze())
    import pickle

    directiory = 'test'

    matchedpairs = pickle.load(open('./data/' + Config.language + '/' + directiory  + '/matchedp-iterl' + fillup + '-' + method + '.pkl', 'rb'))
    newindex = np.load('./data/' + Config.language + '/' + directiory  + '/newindex-iterl' +fillup + '-'+ method +'.npy')
    cans = np.load('./data/' + Config.language + '/' + directiory  + '/cans-iterl' +fillup + '-'+ method +'.npy')
    scores = np.load('./data/' + Config.language + '/' + directiory  + '/scores-iterl' +fillup + '-'+ method +'.npy')
    leftids = np.load('./data/' + Config.language + '/' + directiory  + '/leftids-iterl' +fillup + '-'+ method +'.npy')
    rightids = np.load('./data/' + Config.language + '/' + directiory  + '/rightids-iterl' +fillup + '-'+ method +'.npy')

    # print(newindex)
    # print(cans)
    # print(scores)
    # print(leftids)
    # print(rightids)

    KG1 = loadfile(Config.kg1, 3)
    KG2 = loadfile(Config.kg2, 3)

    testnum = 10500
    nelinenum = 10500
    valinum = 900
    if Config.language == 'dbp_fb':
        testnum = 17880
        nelinenum = 25542
        valinum = 1532

    M1 = np.zeros((testnum, testnum))
    M2 = np.zeros((testnum, testnum))
    for item in KG1:
        if item[0] < testnum and item[2] < testnum:
            M1[item[0], item[2]] = 1
    # print(M1)
    # print(np.sum(M1, axis=-1))
    for item in KG2:
        if item[0] - nelinenum < testnum and item[2] - nelinenum < testnum:
            M2[item[0] - nelinenum, item[2] - nelinenum] = 1
    # print(M2)
    # print(np.sum(M2, axis=-1))

    M1_vali = np.zeros((valinum, valinum))
    M2_vali = np.zeros((valinum, valinum))
    for item in KG1:
        if item[0] < testnum + valinum and item[2] < testnum + valinum and item[0] > testnum and item[2] > testnum:
            M1_vali[item[0] - testnum, item[2] - testnum] = 1
    # print(M1_vali)
    # print(np.sum(M1_vali, axis=-1))
    for item in KG2:
        if item[0] - nelinenum < testnum + valinum and item[2] - nelinenum < testnum + valinum and item[0] - nelinenum > testnum and \
                item[2] - nelinenum > testnum:
            M2_vali[item[0] - testnum - nelinenum, item[2] - testnum - nelinenum] = 1
    # print(M2_vali)
    # print(np.sum(M2_vali, axis=-1))

    if directiory == 'vali':
        M1 = M1_vali
        M2 = M2_vali

    # norm = np.max(np.sum(M2[rightids], axis=-1).squeeze())
    norm = 1

    OUTPUT_GRAPH = False
    GAMMA = 0.9  # reward discount in TD error
    LR_A = 0.001  # learning rate for actor
    LR_C = 0.002  # learning rate for critic
    truncNum = 10

    N_F = truncNum
    N_A = truncNum
    sess = tf.Session()
    actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
    critic = Critic(sess, n_features=N_F,
                    lr=LR_C)  # we need a good teacher, so the teacher should learn faster than the actor

    sess.run(tf.global_variables_initializer())
    entNum = len(rightids)
    t = time.time()
    epoch = 500
    fig_loss = np.zeros([epoch])
    fig_accuracy = np.zeros([epoch])

    highest = 0
    RECORD = []
    for i_episode in range(epoch):
        print(i_episode)
        golScoreWhole = np.array([1.0]*entNum)
        trueacts = []
        ids = []

        idl2r = matchedpairs
        # for i in range(10500):
        #     if i not in leftids:
        #         idl2r[i] = i

        if directiory == 'vali':
            adj = np.where(M1[leftids[newindex[0]] - testnum] == 1)[0]
        else:
            adj = np.where(M1[leftids[newindex[0]]] == 1)[0]

        if len(adj) > 0:
            rids = []
            for id in adj:
                if id in idl2r:
                    rids.append(idl2r[id])
            # radj = np.argwhere(rightids == rids)
            if directiory == 'vali':
                Ms = M2[rightids[cans[0]] - testnum]
            else:
                Ms = M2[rightids[cans[0]]]
            Ms = Ms[:, rids]
            cohScore = np.sum(Ms, axis=-1).squeeze()/norm
            # print(cohScore)
        else:
            cohScore = np.array([0.0] * truncNum)

        golScore = golScoreWhole[cans[0]]
        locScore = scores[0]
        #observation = np.concatenate((locScore*golScore, cohScore))
        observation = locScore*golScore + cohScore
        # observation = locScore + cohScore

        for i in range(len(leftids)):
            # print(i)
            #index = np.argwhere(leftids == newindex[i])[0] # 在leftids中第几个 序列编码
            #index =newindex[i]  # 在leftids中第几个 序列编码

            #print(observation.shape)
            #observation = np.concatenate((locScore, golScore))
            #observation = locScore*golScore
            action = actor.choose_action(observation)
            trueaction = cans[i][action]
            trueacts.append(trueaction)

            idl2r[leftids[newindex[i]]] = rightids[trueaction]

            ids.append(i)
            golScoreWhole[trueaction] = -1
            # print(action)
            # print(locScore + golScore)
            #golScore[golScore==0] = -1 #-5
            #reward = (locScore*golScore)[action] #+ golScore[action]
            # if golScore[action] == -1:
            #     gs = 0
            # else:
            #     gs = 1
            # print(cohScore)

            # abla  reward = (locScore*golScore +cohScore)[action] # + golScore[action]
            reward = (locScore*golScore +cohScore)[action]  # + golScore[action]

            if i == len(leftids) - 1: break
            golScore_ = golScoreWhole[cans[i+1]]
            locScore_ = scores[i+1]
            if directiory == 'vali':
                adj = np.where(M1[leftids[newindex[i + 1]]-testnum] == 1)[0]
            else:
                adj = np.where(M1[leftids[newindex[i+1]]] == 1)[0]

            if len(adj) > 0:
                rids = []
                for id in adj:
                    if id in idl2r:
                        rids.append(idl2r[id])
                # radj = np.argwhere(rightids == rids)
                if directiory == 'vali':
                    Ms = M2[rightids[cans[i+1]] - testnum]
                else:
                    Ms = M2[rightids[cans[i + 1]]]
                Ms = Ms[:, rids]
                cohScore_ = np.sum(Ms, axis=-1).squeeze()/norm
                #print(cohScore_)
            else:
                cohScore_ = np.array([0.0] * truncNum)

            # observation = np.concatenate((locScore, golScore))
            # observation_ = locScore_ * golScore_
            #observation_ = np.concatenate((locScore_*golScore_, cohScore_))

            observation_ = locScore_ * golScore_ + cohScore_
            # observation_ = locScore_ + cohScore_

            td_error = critic.learn(observation, reward, observation_)  # gradient = grad[r + gamma * V(s_) - V(s)]
            actor.learn(observation, action, td_error)  # true_gradient = grad[logPi(s,a) * td_error]
            golScore = golScore_
            locScore = locScore_
            cohScore = cohScore_
            observation = observation_
            #golScoreWhole[trueaction] = golScoreWhole[trueaction] - 1
            #golScore[action] = golScore[action] - 1
            #golScore[action] = -1
            # print(golScore)

        #actions = RL.ep_as
        print(len(trueacts))
        truth = np.where(rightids[trueacts] == leftids[newindex[ids].tolist()])

        print(len(truth[0]))
        RECORD.append(len(truth[0]))
        if len(truth[0]) > highest:
            highest = len(truth[0])
        print('highest ' + str(highest))
        print()

        fig_accuracy[i_episode] = len(truth[0])

        if (i_episode+1) % epoch == 0:
            print("total time elapsed: {:.4f} s".format(time.time() - t))
            # np.save('./data/' + Config.language + '/RLresults' + fillup + '-' + method + '.npy',
            #         np.array(rightids[trueacts]))

    # IMPORTANT!!! WRITE FILES
    RECORD = np.array(RECORD)
    # np.save('./data/' + Config.language + '/' + directiory  + '/corrects' + str(args.round) + '.npy', RECORD)
    print(np.average(RECORD[-20:]))
    # # for the final round, write down the results:
    #
    # mul = dict()
    # ouf = open('./data/' + Config.language + '/' + directiory  + '/rl-results'+ str(args.round) +'.txt', 'w')
    # for i in range(len(trueacts)):
    #     ouf.write(str(leftids[newindex[ids].tolist()][i]) + '\t' + str(rightids[trueacts][i]) + '\n')
    #     if rightids[trueacts][i] not in mul:
    #         mul[rightids[trueacts][i]] = 1
    #     else:
    #         mul[rightids[trueacts][i]] += 1
    #
    # multiple = []
    # mulsource = 0
    # for item in mul:
    #     if mul[item] > 1:
    #         multiple.append([item, mul[item]])
    #         mulsource += mul[item]
    # print("multiple source entities are aligned to the same target entities!")
    # print('source ' + str(mulsource))
    # print('target ' + str(len(multiple)))
    # print()

    # import matplotlib.pyplot as plt
    # plt.plot(np.arange(epoch), fig_accuracy, label="Truth")
    # plt.legend(loc=0)
    # plt.show()