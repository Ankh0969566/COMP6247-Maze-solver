import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


str = "H:\DLwork\DQN_work\\"
class QLearningTable:
	def __init__(self, n_action, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
		self.n_action = [0,1,2,3,4]
		self.learning_rate = learning_rate
		self.gamma_rate = reward_decay


		self.e_greedy = e_greedy
		self.q_table = pd.DataFrame(columns=self.n_action, dtype=np.int)

	def check_state(self, state):
		if state not in self.q_table.index:
			self.q_table = self.q_table.append(pd.Series([0]*len(self.n_action),index=self.q_table.columns,name=state))

	def choose_action(self, observation):
		self.check_state(observation)
		if np.random.uniform() < self.e_greedy:
			state_action = self.q_table.loc[observation,:]
			action = np.random.choice(state_action[state_action==np.max(state_action)].index)
		else:
			action = np.random.choice(self.n_action)
		return action

	def save_model(self):
		pd.DataFrame(self.q_table).to_csv(str + "Q_table.csv", mode='a')
	

	def store_memory(self, state, action, reward, state_):
		
		return 0

	
	def plot_cost(self,epoch):
		plt.plot(np.arange(len(self.loss_recording)), self.loss_recording)
		plt.ylabel('Cost')
		plt.xlabel('training steps')
		plt.savefig(str+'pic-{}.png'.format(epoch%20 + 1)) 
		plt.close()


	def learn(self, s, a, r, s_):
		self.check_state(s_)
		q_predict = self.q_table.loc[s, a]
		if s_ != 'terminal':
			q_target = r + self.gamma_rate*self.q_table.loc[s_, :].max()
		else:
			q_target = r
		self.q_table.loc[s, a] += self.learning_rate*(q_target-q_predict)