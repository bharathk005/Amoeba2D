import aworld as w
import eyes as e
import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt

import time
TOTAL_EPISODES = 50
LR = 0.01
GAMMA = 0.95

def basic_control():
	world = w.CreateWorld()
	world_dat = world.reset()
	done = True
	action = 1
	last_time = time.time()
	while True:
		
		#e.see_amoeba_world(0,50,world.displayX,world.displayY)
		reward,state,done = world.run_frame(action)
		print(state.shape)
		x,y = world.objects[0].body.position
		action = int((world.n_dir +1)/2)
		if world.running == False:
			window_dat = world.reset()
			#print(np.__config__.show())
			print('loop took {} seconds'.format(time.time()-last_time))	
			last_time = time.time()
		

		# temp = pygame.display.set_mode((world.displayX, world.displayY))
		# temp.blit(pygame.surfarray.make_surface(window_dat),(0,0) )
		# pygame.display.update()
		# plt.imshow(window_dat)
		# plt.show()

def discount_and_normalize_rewards(episode_rewards):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * GAMMA + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative
    
    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)
    if std == 0:
    	discounted_episode_rewards = discounted_episode_rewards
    else:
    	discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)
    
    return discounted_episode_rewards

class CreateNN():
	def __init__(self,ip_size,action_size,LR,name):
		self.ip_size = ip_size
		self.action_size = action_size
		self.LR = LR
		self.name = name

		with tf.name_scope(name):
			self.inputs = tf.placeholder(tf.float32,[None,*ip_size],name = "inputs")
			self.actions = tf.placeholder(tf.float32,[None,action_size],name = "actions")
			self.disc_ep_reward = tf.placeholder(tf.float32,[None,],name = "disc_ep_reward")

			self.mean_reward = tf.placeholder(tf.float32,name = "mean_reward")

			with tf.name_scope(name + " con1"):
				self.con1 = tf.layers.conv2d(inputs = self.inputs,
	                                         filters = 32,
	                                         kernel_size = [20,10],
	                                         strides = [10,5],
	                                         padding = "VALID",
	                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
	                                         name = "con1")
				self.con1_batchnorm = tf.layers.batch_normalization(self.con1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm1')
				self.con1_out = tf.nn.elu(self.con1_batchnorm, name="con1_out")


			with tf.name_scope(name + " con2"):
				self.con2 = tf.layers.conv2d(inputs = self.con1_out,
	                                         filters = 64,
	                                         kernel_size = [8,4],
	                                         strides = [4,2],
	                                         padding = "VALID",
	                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
	                                         name = "con2")
				self.con2_batchnorm = tf.layers.batch_normalization(self.con2,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm2')
				self.con2_out = tf.nn.elu(self.con2_batchnorm, name="con2_out")

			with tf.name_scope(name + " con3"):
				self.con3 = tf.layers.conv2d(inputs = self.con2_out,
	                                         filters = 128,
	                                         kernel_size = [4,4],
	                                         strides = [2,2],
	                                         padding = "VALID",
	                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
	                                         name = "con3")
				self.con3_batchnorm = tf.layers.batch_normalization(self.con3,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm3')
				self.con3_out = tf.nn.elu(self.con3_batchnorm, name="con3_out")

			with tf.name_scope(name + " flat_n_dense"):
				self.flater = tf.layers.flatten(self.con3_out)
				self.fc = tf.layers.dense(inputs = self.flater, units = 512, activation = tf.nn.elu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc1")
				self.output = tf.layers.dense(inputs = self.fc, 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units = 3, 
                                        activation=None)


			with tf.name_scope(name + "-softmax"):
				self.action_prob = tf.nn.softmax(self.fc4)

			with tf.name_scope(name + "-loss"):
				self.prob_log = tf.nn.softmax_cross_entropy_with_logits(logits = self.fc4,labels = self.actions)
				self.loss = tf.reduce_mean(self.prob_log * self.disc_ep_reward)

			with tf.name_scope(name + "-train"):
				self.train_op = tf.train.AdamOptimizer(self.LR).minimize(self.loss)




def trainNN():
	world = w.CreateWorld()
	state = world.reset()

	network = CreateNN(state.shape,world.action_size,LR,"FirstNN")
	complete_rewards = []
	total_rewards = 0
	max_reward = 0
	episode_states, episode_actions,episode_rewards = [],[],[]
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		

		for episode in range(TOTAL_EPISODES):
			reward_for_episode = 0
			state = world.reset()

			while True:
				actions = sess.run(network.action_prob, feed_dict = {network.inputs:state.reshape(1,*state.shape) })
				print('****************************************')
				print(actions.shape)
				choice = np.argmax(actions)
				action = world.action_space[int(choice)]
				reward,next_state,done = world.run_frame(action)

				episode_rewards.append(reward)
				episode_states.append(state)
				episode_actions.append(actions)
				if done:
					reward_for_episode = np.sum(episode_rewards)
					complete_rewards.append(reward_for_episode)

					max_reward = np.amax(complete_rewards)

					print("******************************")
					print("Epi: ", episode)
					print("Epi Reward: ", reward_for_episode)
					print("Max Reward: ", max_reward)
					dis_ep_reward = discount_and_normalize_rewards(episode_rewards)
					loss,_ = sess.run([network.loss,network.train_op],feed_dict={network.inputs: episode_states,network.actions: np.vstack(np.array(episode_actions)),network.disc_ep_reward: dis_ep_reward})

					episode_states, episode_actions,episode_rewards = [],[],[]
					break

				state = next_state






if __name__ == '__main__':
	#basic_control()
	trainNN()
	print('THE END')
