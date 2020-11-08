import tensorflow as tf
class Config:
	#language = 'dbp_wd_15k_V1' # dbp_wd_15k_V1 | en_fr_15k_V1 | wk3l_60k/en_de
	language = 'fr_en' # zh_en | ja_en | fr_en ||en_fr _15k_V1　en_de _15k_V1 dbp_wd _15k_V1 dbp_yg _15k_V1 wd_imdb dbp_fb
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
	#gamma = 1.0  # margin based loss
	gamma = 3.0  # margin based loss
	#k = 125  # number of negative samples for each positive one
	k=25
	#seed = 3  # 30% of seeds
	seed = 0.3
	epochs_se = 300
	beta = 0.3
