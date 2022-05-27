import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from memory_data import Memory
import os

"""
This class is used to implement the algorithms in Deep Q learning
Use a CNN 
"""

str = "H:\DLwork\DQN_work\\"
class Deep_Q_net(nn.Module):
	def __init__(self, n_feature, n_hidden, n_actions):
		super(Deep_Q_net, self).__init__()

		self.device = torch.device('cuda:0')
		self.conv1 = nn.Conv2d(3, 6, 1)
		self.fc1 = nn.Linear(54,32) #为什么是54 呢
		self.fc2 = nn.Linear(32, n_actions)

		self.to(self.device)
	def save_net(self):
		print('Saving network ...')
		torch.save(self.state_dict(), str+'maze_model.pt')
	
	def load_net(self): # file
		print('Load saves ...')
		self.load_state_dict(torch.load('H:\DLwork\DQN_work\maze_model.pt'))

	def forward(self, x):
		

		out = self.conv1(x)
		out = F.relu(out)

		out = out.view(out.size()[0], -1)
		out = self.fc1(out)
		out = F.relu(out)
		out = self.fc2(out)
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

class CNNDeepQNetwork():
	def __init__(self, n_actions=5, n_features=18, n_hidden=20, learning_rate=0.01, gamma_rate=0.9, epsilon_greedy=0.9,
				q_target_replace=200, memory_size=500, batch_size=32, greedy_flag = False):
		self.n_actions = n_actions
		self.n_features = n_features
		self.n_hidden = n_hidden
		self.learning_rate = learning_rate
		self.gamma_rate = gamma_rate
		self.q_target_replace = q_target_replace
		self.memory_size = memory_size
		self.batch_size = batch_size
		
		self.test_flag = False
		self.no_fire = True

		self.greedy_flag = greedy_flag
		self.epsilon_final = epsilon_greedy
		self.epsilon = 0.2

		# total learning step
		self.step_num = 0

		# memory
		self.memory_data = Memory(self.memory_size,n_features)
		#Mse lose
		self.loss_function = nn.MSELoss()

		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

		self.loss_recording = []
		self.build_NN()

		
	def loss_record_reset(self):
		self.loss_recording = []

	def build_NN(self):
		#build eval network and target network
		self.q_eval_network = Deep_Q_net(self.n_features, self.n_hidden, self.n_actions).to(self.device)
		self.q_target_network = Deep_Q_net(self.n_features, self.n_hidden, self.n_actions).to(self.device)
		self.optimizer = torch.optim.Adam(self.q_eval_network.parameters(), lr=self.learning_rate)



	def store_memory(self, state, action, reward, state_):

		state = state.flatten()
		state_ = state_.flatten()

		self.memory_data.save_memory(state, action, reward, state_)



	#Choose action base on ϵ-greedy policy
	def choose_action(self, observation):
		

		observation = torch.as_tensor(observation,dtype=torch.float32).to(self.device)
		
		if np.random.uniform() < self.epsilon:
			#transefer the observation into correct form for CNN
			if self.no_fire:
				observation = observation[:,:,[0,2,3]]
			observation = observation.transpose(0,2).to(self.device)
			observation = observation.unsqueeze(0)


			actions_value = self.q_eval_network(observation)
			action = torch.argmax(actions_value).item()

		else:
			action = np.random.randint(0, self.n_actions)
		return action



	def learn(self):

		# Update the target parameters
		if self.step_num % self.q_target_replace == 0:
			self.q_target_network.load_state_dict(self.q_eval_network.state_dict())
			if self.test_flag:
				print("Target paremeter upload")
				
		# sample batch memory from all memory
		batch_memory_data = self.memory_data.batch_sample(self.memory_size,self.batch_size)


		#Extraction of two states in the experience
		memory_state = batch_memory_data[:, -self.n_features:]

		#This part used to transefer the observation into correct form for CNN
		################
		if self.no_fire:
			m_1 = np.zeros((memory_state.shape[0],3,3,3)) 
			index = 0
			for i in memory_state:
				i = np.reshape(i,(3,3,4))
				i = i[:,:,[0,2,3]]
				i = i.transpose(2,0,1)
				m_1[index] = i
				index+=1
		else:
			m_1 = np.zeros((memory_state.shape[0],4,3,3))
			index = 0
			for i in memory_state:

				i = np.reshape(i,(3,3,4))
				i = i.transpose(2,0,1)
				m_1[index] = i
				index+=1

		memory_state_ = batch_memory_data[:, :self.n_features]
		if self.no_fire:
			m_2 = np.zeros((memory_state_.shape[0],3,3,3))
			index_2 = 0
			for i in  memory_state_:
				i = np.reshape(i,(3,3,4))
				i = i[:,:,[0,2,3]]
				i = i.transpose(2,0,1)
				m_2[index_2] = i
				index_2+=1
		else:
			m_2 = np.zeros((memory_state_.shape[0],4,3,3))
			index_2 = 0
			for i in  memory_state_:
				i = np.reshape(i,(3,3,4))
				i = i.transpose(2,0,1)
				m_2[index_2] = i
				index_2+=1
		#################

		q_next = self.q_target_network(torch.Tensor(m_2).to(self.device))
		q_old = self.q_eval_network(torch.Tensor(m_1).to(self.device))


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
		q_new[batch_index, eval_act_index] = reward + self.gamma_rate*torch.max(q_next, 1)[0].to(self.device)
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
		self.q_eval_network.save_net()
		self.q_eval_network.save_net()
	def load_model(self):
		self.q_eval_network.load_net()
		self.q_eval_network.load_net()

	