
from sre_parse import expand_template
import time
from read_maze import get_local_maze_information
from read_maze import load_maze
import matplotlib.pyplot as plt
import numpy as np
from maze_env import Maze
from environment import Environment
from agent import DeepQNetwork
from CNNagent import CNNDeepQNetwork
import pandas as pd
import sys
from Q_agent import QLearningTable

str = "H:\DLwork\DQN_work\\"
class Logger(object):
    def __init__(self, filename='H:\DLwork\DQN_work\\default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass


sys.stdout = Logger('H:\DLwork\DQN_work\\default.log', sys.stdout)




def run():

    episodes = 100000

    n_action = 5
    #Dimension of input neural network
    n_feature = 2 #Use postion

    # n_feature = 36 #Use wall fire and agent postion 
    #Number of hidden neurons
    n_hidden =32
    pre_postion = (1,1)
    canv = Maze()

    agent = DeepQNetwork(n_action, n_feature,n_hidden,
        learning_rate=0.001,
        gamma_rate=0.9,
        epsilon_greedy=0.8,
        q_target_replace=200,
        memory_size=2000000,
        batch_size=20,
        greedy_flag=True
        )
    env = Environment()

    print('...starting...')
    


    for train_episodes in range(episodes):

    #Initializing the map
        time_start_epoch=time.time()
        observation = env.reset()


        


        canv.set_dynamic(env.get_original_around, env.get_actor_postion, [])
        done = False
        #Initialize the maze environment
        last_epoch_postion = pre_postion
        
        # observation = np.zeros((2,1))

        score = 0
        step = 0
        step_epoch = []
        ##Random explore
        # while not done:
        #     canv.maze_run(env.get_original_around, env.get_actor_postion, env.get_actor_path)

        #     action = np.random.randint(0,4)
        #     observation_, reward, done = env.step(action, score)            
        #     agent.store_memory(observation, action,reward, observation_)
            
        # print("finish explore")


        
        same_po = 0
        done = False
        while not done:

            canv.maze_run(env.get_original_around, env.get_actor_postion, env.get_actor_path)
            #Choose action base on epsion-greedy
            action = agent.choose_action(observation)

            observation_, reward, done = env.step(action, score)

            #If agent has not changed position for too long
            if env.get_actor_postion == pre_postion:
                same_po +=1
            else: 
                same_po = 0
            if same_po>500000:
                done = True
                print("Stay too long")



#Out put
            obs_origin = env.get_original_around
            wall = []
            fire = []
            for i in range(3):
                for j in range(3):
                    if obs_origin[i][j][0] == 0:
                        wall.append([i,j])
                    if obs_origin[i][j][1] >0: 
                        fire.append([i,j])   

            # print("Time",step,"actor position",env.get_actor_postion,"action",action)
            # print("Path",env.get_actor_path)
            # print("Observation")
            # print("Wall",wall)
            # print("fire",fire)
            # print(" ")
            score += reward
            



            #record position in last step
            pre_postion = env.get_actor_postion
            #store into memory
            agent.store_memory(observation, action,reward, observation_)

            if (step>0) and (step%1 ==0):
                agent.learn()

            observation = observation_

            step += 1

            if step%200 ==0:
                agent.save_model()
                # print(step," ", agent.get_loss_record())
                agent.plot_cost(step)
                # time_end_epoch=time.time()
                # print('time cost',time_start_epoch-time_end_epoch,'s')
                # pd.DataFrame(env.get_actor_path).to_csv(str + "agent_path.csv", mode='a')
                # pd.DataFrame(M.q_table.table).to_csv(dir + "q_table.csv")

        

        step_epoch.append(step)
        time_end_epoch=time.time()
        print('time cost',time_start_epoch-time_end_epoch,'s')
        pd.DataFrame(env.get_actor_path).to_csv(str + "agent_path.csv", mode='a')     
        plot_epoch_step(train_episodes,step_epoch)

def plot_epoch_step(epoch,step_epoch):
    plt.plot(np.arange(len(step_epoch)), step_epoch)
    plt.ylabel('Steps')
    plt.xlabel('Epoch')
    plt.savefig(str+'pic-{}.png'.format(epoch%20 + 1)) 
    plt.close()
if __name__ == '__main__':
    run()





