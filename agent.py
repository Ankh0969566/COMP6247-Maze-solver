import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from memory_data import Memory
import os



"""
This class is used to implement the algorithms in Deep Q learning
Use a 3 layers full connect neural network
"""


str = "H:\DLwork\DQN_work\\"
class Deep_Q_net(nn.Module):
	def __init__(self, n_feature, n_hidden, n_actions):
		super(Deep_Q_net, self).__init__()

		self.device = torch.device('cuda:0')
		self.hidden1 = nn.Linear(n_feature, n_hidden)
		self.hidden2 = nn.Linear(n_hidden,n_hidden)
		self.output = nn.Linear(n_hidden,n_actions)

		self.to(self.device)

	def save_net(self):
		print('Saving network ...')
		torch.save(self.state_dict(), str+'maze_model.pt')
	
	def load_net(self): 
		print('Load saves ...')
		self.load_state_dict(torch.load('H:\DLwork\DQN_work\maze_model.pt'))

	def forward(self, x):

		out = self.hidden1(x)
		out = F.relu(out)
		out = self.hidden2(out)
		out = F.relu(out)
		out = self.output(out)
		return out

"""
n_actions: 5 actions the agent can do
n_features: Dimension of input neural network
learning_rate: the learning rate alpha
gamma_rate: epsion-greedy 
q_target_replace: After centen steps, updata the target network
memory_size: the size of the memory
batch_size : the batch size used to sample from memory
"""

class DeepQNetwork():
	def __init__(self, n_action=5, n_features=18, n_hidden=20, learning_rate=0.01, gamma_rate=0.9, epsilon_greedy=0.9,
				q_target_replace=200, memory_size=500, batch_size=32, greedy_flag = False):
		self.n_action = n_action
		self.n_features = n_features
		self.n_hidden = n_hidden
		self.learning_rate = learning_rate
		self.gamma_rate = gamma_rate
		self.q_target_replace = q_target_replace
		self.memory_size = memory_size
		self.batch_size = batch_size
		
		self.test_flag = False
		self.no_fire = False

		self.greedy_flag = greedy_flag
		self.epsilon_final = epsilon_greedy
		self.epsilon = 0.2

		# total learning step
		self.step_num = 0


		# memory
		self.memory_data = Memory(self.memory_size,n_features)
		#Mse lose
		self.loss_function = nn.MSELoss()

		self.device = torch.device('cuda:0')

		self.loss_recording = []
		self.build_NN()

		
	def get_loss_record(self):
		return self.loss_recording

	def build_NN(self):
		#build eval network and target network
		self.q_eval_net = Deep_Q_net(self.n_features, self.n_hidden, self.n_action)
		self.q_target_net = Deep_Q_net(self.n_features, self.n_hidden, self.n_action)
		self.optimizer = torch.optim.Adam(self.q_eval_net.parameters(), lr=self.learning_rate)
        #RMSprop

	#store the experience into memory
	def store_memory(self, state, action, reward, state_):

		state = state.flatten()
		state_ = state_.flatten()

		self.memory_data.save_memory(state, action, reward, state_)


	#Choose action base on ϵ-greedy policy
	def choose_action(self, observation):
				
		observation = torch.as_tensor(observation,dtype=torch.float32)
		observation = torch.flatten(observation).to(self.device)
		
		if np.random.uniform() < self.epsilon:
			actions_value = self.q_eval_net.forward(observation)
			action = torch.argmax(actions_value).item()

		else:
			action = np.random.randint(0, self.n_action)
		return action



	def learn(self):

		# Update the target parameters
		if self.step_num % self.q_target_replace == 0:
			self.q_target_net.load_state_dict(self.q_eval_net.state_dict())
			if self.test_flag:
				print("Target paremeter upload")
				
		# sample batch memory from all memory
		batch_memory_data = self.memory_data.batch_sample(self.memory_size,self.batch_size)


		#Extraction of two states in the experience
		memory_state = batch_memory_data[:, -self.n_features:]
		memory_state_ = batch_memory_data[:, :self.n_features]


		q_next = self.q_target_net(torch.Tensor(memory_state_).to(self.device))
		q_old = self.q_eval_net(torch.Tensor(memory_state).to(self.device))

		


		q_new = torch.zeros_like(q_old).to(self.device)
		
		batch_index = np.arange(self.batch_size, dtype=np.int32)
		"""
		from 0 to n_feature-1 is state 
		the memory [n_feature] is action 
		the memory [n_feature+1] is reward. 
		"""
		
		eval_act_index = batch_memory_data[:, self.n_features].astype(int)
		#take out reward in batch
		reward = torch.Tensor(batch_memory_data[:, self.n_features+1]).to(self.device)

		
		#use Q_target_network calculate the new Q value
		q_new[batch_index, eval_act_index] = reward + self.gamma_rate*torch.max(q_next, 1)[0]
		#loss function
		loss = self.loss_function(q_old, q_new)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		
		
		self.epsilon = self.epsilon_final
		
		self.loss_recording.append(loss.cpu().item())
		self.step_num += 1

	def plot_cost(self,epoch):
		plt.plot(np.arange(len(self.loss_recording)), self.loss_recording)
		plt.ylabel('Cost')
		plt.xlabel('training steps')
		plt.savefig(str+'pic-{}.png'.format(epoch%20 + 1)) 
		plt.close()


	def save_model(self):
		self.q_eval_net.save_net()
		self.q_eval_net.save_net()
	def load_model(self):
		self.q_eval_net.load_net()
		self.q_eval_net.load_net()

	