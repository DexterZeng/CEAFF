import math
from .Init import *
import copy
import numpy as np
from include.Load import *
import json

def func(KG):
	head = {}
	cnt = {}
	for tri in KG:
		if tri[1] not in cnt:
			cnt[tri[1]] = 1
			head[tri[1]] = set([tri[0]])
		else:
			cnt[tri[1]] += 1
			head[tri[1]].add(tri[0])
	r2f = {}
	for r in cnt:
		r2f[r] = len(head[r]) / cnt[r]
	return r2f


def ifunc(KG):
	tail = {}
	cnt = {}
	for tri in KG:
		if tri[1] not in cnt:
			cnt[tri[1]] = 1
			tail[tri[1]] = set([tri[2]])
		else:
			cnt[tri[1]] += 1
			tail[tri[1]].add(tri[2])
	r2if = {}
	for r in cnt:
		r2if[r] = len(tail[r]) / cnt[r]
	return r2if


def get_mat(e, KG):
	r2f = func(KG)
	r2if = ifunc(KG)
	du = [1] * e
	for tri in KG:
		if tri[0] != tri[2]:
			du[tri[0]] += 1
			du[tri[2]] += 1
	M = {}
	for tri in KG:
		if tri[0] == tri[2]:
			continue
		if (tri[0], tri[2]) not in M:
			M[(tri[0], tri[2])] = math.sqrt(math.sqrt(math.sqrt(math.sqrt(r2if[tri[1]]))))
		else:
			M[(tri[0], tri[2])] += math.sqrt(math.sqrt(math.sqrt(math.sqrt(r2if[tri[1]]))))
		if (tri[2], tri[0]) not in M:
			M[(tri[2], tri[0])] = math.sqrt(math.sqrt(math.sqrt(math.sqrt(r2f[tri[1]]))))
		else:
			M[(tri[2], tri[0])] += math.sqrt(math.sqrt(math.sqrt(math.sqrt(r2f[tri[1]]))))
	for i in range(e):
		M[(i, i)] = 1
	return M, du


# get a sparse tensor based on relational triples
def get_sparse_tensor(e, KG):
	print('getting a sparse tensor...')
	M, du = get_mat(e, KG)
	ind = []
	val = []
	for fir, sec in M:
		ind.append((sec, fir))
		val.append(M[(fir, sec)] / math.sqrt(du[fir]) / math.sqrt(du[sec]))
	M = tf.SparseTensor(indices=ind, values=val, dense_shape=[e, e])
	return M


# add a layer
def add_diag_layer(inlayer, dimension, M, act_func, dropout=0.0, init=ones):
	inlayer = tf.nn.dropout(inlayer, 1 - dropout)
	print('adding a layer...')
	w0 = init([1, 300])
	tosum = tf.sparse_tensor_dense_matmul(M, tf.multiply(inlayer, w0))
	if act_func is None:
		return tosum
	else:
		return act_func(tosum)


def add_full_layer(inlayer, dimension_in, dimension_out, M, act_func, dropout=0.0, init=glorot):
	inlayer = tf.nn.dropout(inlayer, 1 - dropout)
	print('adding a layer...')
	w0 = init([dimension_in, dimension_out])
	tosum = tf.sparse_tensor_dense_matmul(M, tf.matmul(inlayer, w0))
	if act_func is None:
		return tosum
	else:
		return act_func(tosum)


# se input layer
def get_se_input_layer(e, dimension):
	# with open(file='./data/'+ Config.language + '/ja' + '_vectorList.json', mode='r', encoding='utf-8') as f:
	# 	embedding_list = json.load(f)
	# input_embeddings = tf.convert_to_tensor(embedding_list)
	# ent_embeddings = tf.Variable(input_embeddings)
	#
	# nepath = './RSN-full-datasets/'+ Config.language + '/mapping/0_3/name_vec_cpm_3.txt'
	# ne_vec = loadNe(nepath)
	# ne_vec = ne_vec.astype(np.float32)
	# input_embeddings = tf.convert_to_tensor(ne_vec)
	# ent_embeddings = tf.Variable(input_embeddings)

	print('adding the se input layer...')
	ent_embeddings = tf.Variable(tf.truncated_normal([e, dimension], stddev=1.0 / math.sqrt(e)))
	print(ent_embeddings)
	print(type(ent_embeddings))
	return tf.nn.l2_normalize(ent_embeddings, 1)


# get loss node
def get_loss(outlayer, ILL, gamma, k):
	print('getting loss...')
	left = ILL[:, 0]
	right = ILL[:, 1]
	t = len(ILL)
	left_x = tf.nn.embedding_lookup(outlayer, left)
	right_x = tf.nn.embedding_lookup(outlayer, right)
	A = tf.reduce_sum(tf.abs(left_x - right_x), 1)
	neg_left = tf.placeholder(tf.int32, [t * k], "neg_left")
	neg_right = tf.placeholder(tf.int32, [t * k], "neg_right")
	neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
	neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
	B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
	C = - tf.reshape(B, [t, k])
	D = A + gamma
	L1 = tf.nn.relu(tf.add(C, tf.reshape(D, [t, 1])))
	neg_left = tf.placeholder(tf.int32, [t * k], "neg2_left")
	neg_right = tf.placeholder(tf.int32, [t * k], "neg2_right")
	neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
	neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
	B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
	C = - tf.reshape(B, [t, k])
	L2 = tf.nn.relu(tf.add(C, tf.reshape(D, [t, 1])))
	return (tf.reduce_sum(L1) + tf.reduce_sum(L2)) / (2.0 * k * t)

def highway(layer1, layer2, dimension):
    kernel_gate = glorot([dimension, dimension])
    bias_gate = zeros([dimension])
    transform_gate = tf.matmul(layer1, kernel_gate) + bias_gate
    transform_gate = tf.nn.sigmoid(transform_gate)
    carry_gate = 1.0 - transform_gate
    return transform_gate * layer2 + carry_gate * layer1


def build_SE(dimension, act_func, gamma, k, e, ILL, KG):
	tf.reset_default_graph()
	input_layer = get_se_input_layer(e, dimension)
	M = get_sparse_tensor(e, KG)

	hidden_layer = add_diag_layer(input_layer, dimension, M, act_func, dropout=0.0)
	#hidden_layer = highway(input_layer, hidden_layer, dimension)

	output_layer = add_diag_layer(hidden_layer, dimension, M, None, dropout=0.0)
	#output_layer = highway(hidden_layer, output_layer, dimension)

	loss = get_loss(output_layer, ILL, gamma, k)
	return output_layer, loss

def training(output_layer, loss, learning_rate, epochs, ILL, e, k):
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)  # optimizer can be changed
	print('initializing...')
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)
	print('running...')
	J = []
	t = len(ILL)
	ILL = np.array(ILL)
	L = np.ones((t, k)) * (ILL[:, 0].reshape((t, 1)))
	neg_left = L.reshape((t * k,))
	L = np.ones((t, k)) * (ILL[:, 1].reshape((t, 1)))
	neg2_right = L.reshape((t * k,))

	for i in range(epochs):
		if i % 10 == 0:
			neg2_left = np.random.choice(e, t * k)
			neg_right = np.random.choice(e, t * k)
		sess.run(train_step, feed_dict={"neg_left:0": neg_left,
										"neg_right:0": neg_right,
										"neg2_left:0": neg2_left,
										"neg2_right:0": neg2_right})
		if (i + 1) % 20 == 0:
			th = sess.run(loss, feed_dict={"neg_left:0": neg_left,
										   "neg_right:0": neg_right,
										   "neg2_left:0": neg2_left,
										   "neg2_right:0": neg2_right})
			J.append(th)
			print('%d/%d' % (i + 1, epochs), 'epochs...')
	outvec = sess.run(output_layer)
	sess.close()
	return outvec, J


def combine(dimension, act_func, gamma, k, e, ILL, KG):
	tf.reset_default_graph()
	input_layer = get_se_input_layer(e, dimension)
	M = get_sparse_tensor(e, KG)
	hidden_layer = add_diag_layer(input_layer, dimension, M, act_func, dropout=0.0)
	output_layer = add_diag_layer(hidden_layer, dimension, M, None, dropout=0.0)
	#loss = get_loss(output_layer, ILL, gamma, k)
	print('getting loss...')

	# left = ILL[:, 0]
	# right = ILL[:, 1]
	# t = len(ILL) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	t = 4500
	left = tf.placeholder(tf.int32, [t], "left")
	right = tf.placeholder(tf.int32, [t], "right")

	left_x = tf.nn.embedding_lookup(output_layer, left)
	right_x = tf.nn.embedding_lookup(output_layer, right)
	A = tf.reduce_sum(tf.abs(left_x - right_x), 1)
	neg_left = tf.placeholder(tf.int32, [t * k], "neg_left")
	neg_right = tf.placeholder(tf.int32, [t * k], "neg_right")
	neg_l_x = tf.nn.embedding_lookup(output_layer, neg_left)
	neg_r_x = tf.nn.embedding_lookup(output_layer, neg_right)
	B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
	C = - tf.reshape(B, [t, k])
	D = A + gamma
	L1 = tf.nn.relu(tf.add(C, tf.reshape(D, [t, 1])))
	neg_left = tf.placeholder(tf.int32, [t * k], "neg2_left")
	neg_right = tf.placeholder(tf.int32, [t * k], "neg2_right")
	neg_l_x = tf.nn.embedding_lookup(output_layer, neg_left)
	neg_r_x = tf.nn.embedding_lookup(output_layer, neg_right)
	B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
	C = - tf.reshape(B, [t, k])
	L2 = tf.nn.relu(tf.add(C, tf.reshape(D, [t, 1])))
	loss = (tf.reduce_sum(L1) + tf.reduce_sum(L2)) / (2.0 * k * t)

	learning_rate = 25
	epochs = 400

	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)  # optimizer can be changed
	print('initializing...')
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)
	print('running...')
	J = []
	ILL_ori = copy.deepcopy(ILL)

	trainall = []
	for curri in range(4):
		ILL = ILL_ori[curri*1125: (curri+1) * 1125]
		for iii in range(10):
			trainall.extend(ILL)

	for kkk in range(10):
		ILL = trainall[kkk*4500: (kkk+1)*4500]
		t = len(ILL)
		ILL = np.array(ILL)
		L = np.ones((t, k)) * (ILL[:, 0].reshape((t, 1)))
		neg_left = L.reshape((t * k,))
		L = np.ones((t, k)) * (ILL[:, 1].reshape((t, 1)))
		neg2_right = L.reshape((t * k,))
		left = ILL[:, 0]
		right = ILL[:, 1]

		if kkk % 10 == 0:
			neg2_left = np.random.choice(e, t * k)
			neg_right = np.random.choice(e, t * k)
			sess.run(train_step, feed_dict={"left:0": left,
											"right:0": right,
											"neg_left:0": neg_left,
											"neg_right:0": neg_right,
											"neg2_left:0": neg2_left,
											"neg2_right:0": neg2_right})
		if (kkk + 1) % 20 == 0:
			th = sess.run(loss, feed_dict={"left:0": left,
										"right:0": right,
										   "neg_left:0": neg_left,
										   "neg_right:0": neg_right,
										   "neg2_left:0": neg2_left,
										   "neg2_right:0": neg2_right})
			J.append(th)
			print('%d/%d' % (kkk + 1, epochs), 'epochs...')

	for i in range(400):
		# for curri in range(2):
		# 	ILL = ILL_ori[curri*2250: (curri+1) * 2250]
			ILL = copy.deepcopy(ILL_ori)
			t = len(ILL)
			ILL = np.array(ILL)
			left = ILL[:, 0]
			right = ILL[:, 1]
			#print(left)
			L = np.ones((t, k)) * (ILL[:, 0].reshape((t, 1)))
			neg_left = L.reshape((t * k,))
			#print(neg_left)
			L = np.ones((t, k)) * (ILL[:, 1].reshape((t, 1)))
			neg2_right = L.reshape((t * k,))

			if i % 10 == 0:
				neg2_left = np.random.choice(e, t * k)
				neg_right = np.random.choice(e, t * k)
				sess.run(train_step, feed_dict={"left:0": left,
												"right:0": right,
												"neg_left:0": neg_left,
												"neg_right:0": neg_right,
												"neg2_left:0": neg2_left,
												"neg2_right:0": neg2_right})
			if (i + 1) % 20 == 0:
				th = sess.run(loss, feed_dict={"left:0": left,
											"right:0": right,
											   "neg_left:0": neg_left,
											   "neg_right:0": neg_right,
											   "neg2_left:0": neg2_left,
											   "neg2_right:0": neg2_right})
				J.append(th)
				print('%d/%d' % (i + 1, epochs), 'epochs...')

	# for curri in range(4):
	# 	ILL = ILL_ori[curri*1125: (curri+1) * 1125]
	# 	t = len(ILL)
	# 	ILL = np.array(ILL)
	# 	L = np.ones((t, k)) * (ILL[:, 0].reshape((t, 1)))
	# 	neg_left = L.reshape((t * k,))
	# 	L = np.ones((t, k)) * (ILL[:, 1].reshape((t, 1)))
	# 	neg2_right = L.reshape((t * k,))
	# 	left = ILL[:, 0]
	# 	right = ILL[:, 1]
    #
	# 	for i in range(50):
	# 		if i % 10 == 0:
	# 			neg2_left = np.random.choice(e, t * k)
	# 			neg_right = np.random.choice(e, t * k)
	# 		sess.run(train_step, feed_dict={"left:0": left,
	# 										"right:0": right,
	# 										"neg_left:0": neg_left,
	# 										"neg_right:0": neg_right,
	# 										"neg2_left:0": neg2_left,
	# 										"neg2_right:0": neg2_right})
	# 		if (i + 1) % 20 == 0:
	# 			th = sess.run(loss, feed_dict={"left:0": left,
	# 										"right:0": right,
	# 										   "neg_left:0": neg_left,
	# 										   "neg_right:0": neg_right,
	# 										   "neg2_left:0": neg2_left,
	# 										   "neg2_right:0": neg2_right})
	# 			J.append(th)
	# 			print('%d/%d' % (i + 1, epochs), 'epochs...')

	outvec = sess.run(output_layer)
	sess.close()

	return outvec, J
