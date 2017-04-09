import numpy as np 
import tensorflow as tf 
import tensorflow.contrib.slim as slim 

def create_network(inputs=None, layer_dim=None, scope=None):
	#TODO: add activation function options
	# inputs = tf.placeholder(shape=input_shape, dtype=tf.float32)
	for i, layer_size in enumerate(layer_dim):
		if i == 0:
			net = slim.fully_connected(inputs, layer_size, scope=scope+'/fc-'+str(i))
		else:
			net = slim.fully_connected(net, layer_size, scope=scope+'/fc-'+str(i))
	trainable_variables = []
	for i in range(len(layer_dim)):
		for item in slim.get_variables(scope+'/fc-'+str(i)):
			trainable_variables.append(item)

	return net, trainable_variables

def create_input_placeholder(input_shape=None):
	inputs = tf.placeholder(shape=input_shape, dtype=tf.float32)
	return inputs 

def compute_jacobian(f, x, f_dim): #FIXME: The problem is with sess - bypass it 
	# jacobian = tf.stack([tf.gradients(f[0][i], x)[0][0] for i in range(sess.run(tf.shape(f)[1]))])
	jacobian = tf.stack([tf.gradients(f[0][i], x) for i in range(f_dim)])

	return jacobian

def main():
	N_s = 4 
	N_a = 2 
	policy_layer_dim = [10,N_a]
	forward_model_layer_dim = [20,N_s]
	reward_layer_dim = [30,1]
	BATCHSIZE = 10

	dummy_state = np.random.random([BATCHSIZE,N_s])
	dummy_action = np.random.random([BATCHSIZE,N_a])
	dummy_target_action = np.random.random([BATCHSIZE,N_a])
	dummy_target_next_state = np.random.random([BATCHSIZE,N_s])
	dummy_target_reward = np.random.random()

	policy_in = create_input_placeholder([None, N_s])
	policy, policy_params = create_network(policy_in, policy_layer_dim, scope='policy')
	forward_model_in = create_input_placeholder([None, N_s+N_a])
	forward_model, forward_model_params = create_network(forward_model_in, forward_model_layer_dim, scope='forward_model')
	reward_in = create_input_placeholder([None, N_s+N_a])
	reward, reward_params = create_network(reward_in, reward_layer_dim, scope='reward')

	init = tf.global_variables_initializer()
	
	sess = tf.Session()

	sess.run(init)
	print sess.run(tf.shape(reward), {reward_in:np.hstack([dummy_state,dummy_action])})

	# Jacobians of network outputs wrt inputs
	#FIXME: not right
	# grad_policy_stateaction = tf.gradients(policy, policy_in)
	# sess.run(grad_policy_stateaction,{policy_in:dummy_state})

	# grad_reward_stateaction = compute_jacobian(reward, reward_in, 1)
	# grad_reward_state, grad_reward_action = grad_reward_stateaction[:,:N_s], grad_reward_stateaction[:,N_s:]

	# grad_policy_state = compute_jacobian(policy, policy_in, sess)
	
	# grad_forward_model_stateaction = compute_jacobian(forward_model, forward_model_in, sess)
	# grad_forward_model_state, grad_forward_model_action = grad_forward_model_stateaction[:,:N_s], grad_forward_model_stateaction[:,N_s:]

	# r_sa = sess.run(tf.shape(grad_reward_stateaction), feed_dict={reward_in:np.hstack([dummy_state, dummy_action])})
	# f_sa = sess.run(tf.shape(grad_forward_model_stateaction), feed_dict={forward_model_in:np.hstack([dummy_state, dummy_action])})
	# p_s = sess.run(tf.shape(grad_policy_state), feed_dict={policy_in:dummy_state})

	# print r_sa, f_sa, p_s


	# # Trainable Parameters

	# # print("Policy parameters:",[sess.run(tf.shape(policy_param)) for policy_param in policy_params])
	# # print("Forward model parameters:", [sess.run(tf.shape(forward_model_param)) for forward_model_param in forward_model_params])
	# # print("Reward parameters:", [sess.run(tf.shape(reward_param)) for reward_param in reward_params])



	# # Defining some dummy losses
	# policy_loss = tf.nn.l2_loss(policy-dummy_target_action)
	# forward_model_loss = tf.nn.l2_loss(forward_model-dummy_target_next_state)
	# reward_loss = tf.nn.l2_loss(reward-dummy_target_reward)

	# # print(sess.run(policy_loss, {policy_in:dummy_state}))
	# # print(sess.run(forward_model_loss, {forward_model_in:np.hstack([dummy_state, dummy_action])}))
	# # print(sess.run(reward_loss, {reward_in:np.hstack([dummy_state, dummy_action])}))
	

	# # Applying gradients
	# dummy_state = dummy_state+10
	# policy_gradients = tf.gradients(policy_loss, policy_params)
	# pg_val = sess.run(policy_gradients, {policy_in:dummy_state})
	# optimizer = tf.train.GradientDescentOptimizer(0.01)
	# print(sess.run(policy_params, {policy_in:dummy_state}))
	# trainer = optimizer.apply_gradients(zip(policy_gradients, policy_params))
	# sess.run(trainer, {policy_in:dummy_state})
	# import IPython; IPython.embed()



	# train_policy = optimizer.minimize(policy_loss, var_list=policy_params)	
	# sess.run(train_policy, {policy_in:dummy_state})



if __name__=='__main__':
	main()