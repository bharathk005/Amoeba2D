import aworld as w
import eyes as e
import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt




def basic_control():
	while world.running:
		
		#e.see_amoeba_world(0,50,world.displayX,world.displayY)
		reward,window_dat,box_dir = world.run_frame()
		x,y = world.objects[0].body.position
		world.force_dir = world.n_dir 
		

		# temp = pygame.display.set_mode((world.displayX, world.displayY))
		# temp.blit(pygame.surfarray.make_surface(window_dat),(0,0) )
		# pygame.display.update()
		# if x > 450 and world.n_dir == 1:
		# 	plt.imshow(window_dat)
		# 	plt.show()



class CreateNN():
	def __init__(self,ip_size,action_size,LR,name):
		self.ip_size = ip_size
		self.action_size = action_size
		self.LR = LR
		self.name = name

		with tf.variable_scope(name):
			self.inputs = tf.placeholder(tf.float32,[None,ip_size],name = "input")
			self.actions = tf.placeholder(tf.float32,[None,action_size],name = "action")
			self.disc_ep_reward = tf.placeholder(tf.float32,[None,],name = "disconted_rewards")

			self.mean_reward = tf.placeholder(tf.float32,name = "mean_reward")

			with tf.name_scope(name + "FC1"):
				fc1 = tf.contrib.layers.fully_connected(input = self.inputs,
					                                     num_outputs = self.ip_size * 2,
					                                     activation_fn = tf.nn.relu,
					                                     weights_initializer = tf.contrib.layers.xavier_initialer())
			with tf.name_scope(name + "FC2"):
				fc2 = tf.contrib.layers.fully_connected(input = fc1,
					                                     num_outputs = self.ip_size * 2,
					                                     activation_fn = tf.nn.relu,
					                                     weights_initializer = tf.contrib.layers.xavier_initialer())

			with tf.name_scope(name + "FC3"):
				fc3 = tf.contrib.layers.fully_connected(input = fc2,
					                                     num_outputs = self.action_size,
					                                     activation_fn = tf.nn.relu,
					                                     weights_initializer = tf.contrib.layers.xavier_initialer())

			with tf.name_scope(name + "FC4"):
				fc4 = tf.contrib.layers.fully_connected(input = fc3,
					                                     num_outputs = self.action_size,
					                                     activation_fn = None,
					                                     weights_initializer = tf.contrib.layers.xavier_initialer())

	 		with tf.name_scope(name + "softmax"):
	 			action_prob = tf.nn.softmax(fc4)

	 		with tf.name_scope(name + "loss"):
	 			prob_log = tf.nn.softmax_cross_entropy_with_logits(logits = fc4,labels = self.actions)
	 			loss = tf.reduce_mean(prob_log * self.disc_ep_reward)

	 		with tf.name_scope(name + "train"):
	  			train_op = tf.train.AdamOptimizer(self.LR).minimize(loss)




def trainNN():
	network = CreateNN()
	rewards_vec = []
	total_rewards = 0
	max_reward = 0
	episode_states, episode_actions,episode_rewards = [],[],[]
	with tf.Session() as sess:
		sess.run(tf.global_vaiables_initializer())
		world = w.CreateWorld()

		for episode in range(TOTAL_EPISODES):
			reward_for_episode = 0
			





if __name__ == '__main__':
	basic_control()
	print('THE END')
