import aworld as w
import eyes as e
import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt


TOTAL_EPISODES = 50
LR = 0.01
GAMMA = 0.95

def basic_control():
	world = w.CreateWorld()
	done = True
	action = 1
	while done:
		
		#e.see_amoeba_world(0,50,world.displayX,world.displayY)
		reward,state,done = world.run_frame(action)
		x,y = world.objects[0].body.position
		action = int((world.n_dir +1)/2)
		if world.running == False:
			window_dat = world.reset()
			print(window_dat.shape)
		

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
    discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)
    
    return discounted_episode_rewards

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
				self.fc1 = tf.contrib.layers.fully_connected(input = self.inputs,
					                                     num_outputs = self.ip_size * 2,
					                                     activation_fn = tf.nn.relu,
					                                     weights_initializer = tf.contrib.layers.xavier_initialer())
			with tf.name_scope(name + "FC2"):
				self.fc2 = tf.contrib.layers.fully_connected(input = self.fc1,
					                                     num_outputs = self.ip_size * 2,
					                                     activation_fn = tf.nn.relu,
					                                     weights_initializer = tf.contrib.layers.xavier_initialer())

			with tf.name_scope(name + "FC3"):
				self.fc3 = tf.contrib.layers.fully_connected(input = self.fc2,
					                                     num_outputs = self.action_size,
					                                     activation_fn = tf.nn.relu,
					                                     weights_initializer = tf.contrib.layers.xavier_initialer())

			with tf.name_scope(name + "FC4"):
				self.fc4 = tf.contrib.layers.fully_connected(input = self.fc3,
					                                     num_outputs = self.action_size,
					                                     activation_fn = None,
					                                     weights_initializer = tf.contrib.layers.xavier_initialer())

			with tf.name_scope(name + "softmax"):
				self.action_prob = tf.nn.softmax(self.fc4)

			with tf.name_scope(name + "loss"):
				self.prob_log = tf.nn.softmax_cross_entropy_with_logits(logits = self.fc4,labels = self.actions)
				self.loss = tf.reduce_mean(self.prob_log * self.disc_ep_reward)

			with tf.name_scope(name + "train"):
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
		sess.run(tf.global_vaiables_initializer())
		

		for episode in range(TOTAL_EPISODES):
			reward_for_episode = 0
			state = world.reset()

			while True:
				actions = sess.run(network.action_prob, feed_dict = {network.inputs:state})
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
					loss,_ = sess.run([network.loss,network.train_op],feed_dict={network.inputs: np.vstack(np.array(episode_states)),network.actions: np.vstack(np.array(episode_actions)),network.disc_ep_reward: dis_ep_reward})

					episode_states, episode_actions,episode_rewards = [],[],[]
					break

				state = next_state






if __name__ == '__main__':
	basic_control()
	print('THE END')
