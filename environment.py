

"""environment Class 
Environment class return observation and state reward. 
Realization of agent movement and obstruction by walls and flames
"""

from read_maze import get_local_maze_information
import numpy as np

# Set agent action list 
global action_dict
"""
Action Space
0 - stay
1 - up
2 - down
3 - left
4 - right
"""
action_dict = { 0:(0,0),     
                1:(-1,0),
                2:(1,0),
                3:(0,-1),
                4:(0,1),
              } 
#0 = stay , 1 = up 2 = down 3 = left 4 = right

global rewards_dict

#set the agent rewards list
# rewards_dict = {"onwards": -0.05,  #move towards the goal
#               "backwards":-0.05,  #move away from the goal
#               "visited":-0.35,   #visit the postion in path
#               "block":-0.35,  #Agent can not move
#               "fire":-0.05,  #Agent obstructed by flame
#               "wall":-0.9,  #Agent obstructed by wall
#               "unvisit":-0.05, #visit new postion
#               "stay":-0.9, #agent chooses to stay without being blocked
#               'end':10  #goal   
#               }



rewards_dict = {"onwards": -0,  #move towards the goal
              "backwards":-0,  #move away from the goal
              "visited":-0,   #visit the postion in path
              "block":-0,  #Agent can not move
              "fire":-0,  #Agent obstructed by flame
              "wall":-0,  #Agent obstructed by wall
              "unvisit":-0, #visit new postion
              "stay":-0, #agent chooses to stay without being blocked
              'end':10  #goal   
              }

class Environment:
    def __init__(self):


        self.actor_postion = (1, 1)
        self.pre_postion = (1,1)
        self.step_num = 0
        self.obstacle_num = 0
        self.stay_num = 0
        self.visited_num = 0

        
        
        self.actor_path = [self.actor_postion]
        self.observe_environment 
        
        #original observation
        self.observe_original = []
        #obersivation with actor location
        self.observe_actor = []

        self.postion_reward_flage = False

        self.maze_times = np.zeros((200,200))

    
    def reset(self):

        self.step_num = 0
        self.obstacle_num = 0
        self.stay_num = 0
        self.visited_num = 0
        # self.actor_postion = self.last_epoch_postion
        self.actor_postion = (1,1)
        self.pre_action = 0
        self.same_ac = 0
        self.actor_path.clear() #clear path
        self.actor_path = [self.actor_postion]


        self.observe_environment

        # return self.observe_original
        return self.observe_actor


    @property

    def observe_environment(self):
        x, y = self.actor_postion
        the_postion = np.zeros((2,1))
        the_postion[0] = x
        the_postion[1] = y


        self.actor_path.append(self.actor_postion)


        around = get_local_maze_information(x, y)  
        self.observe_original = around
        
        #Construct a 3*3 coordinate matrix with the current position as the center
        actor_loccation_around = np.zeros((3, 3, 2), dtype=int)
        for i in range(3):
            for j in range(3):
                actor_loccation_around[i][j][0] = i+x-1
                actor_loccation_around[i][j][1] = j+y-1




        # self.observe_actor=np.dstack((actor_loccation_around,self.observe_original))  #Use wall and agent postion

        self.observe_actor = the_postion


        #only use the agent postion as neural network input

        # return self.observe_original
        return self.observe_actor
        return the_postion


    @property
    def get_original_around(self):
        return self.observe_original
    @property
    def get_actor_postion(self):
        return self.actor_postion
    @property
    def get_actor_path(self):
        return self.actor_path


    def vist_maze(self,postion):
        
        return self.maze_times[postion[0]][postion[1]] 

    def manha_reward(self):
        new_p = self.actor_postion
        old_p = self.pre_postion

        end_p = (199,199)
        old_dis = end_p[0]-old_p[0] + end_p[1]-old_p[1]
        new_dis = end_p[0]-new_p[0] + end_p[1]-new_p[1]
        if old_dis > new_dis:
            reward = (-0.002) * (end_p[0]-new_p[0]) + (-0.001) * (end_p[1]-new_p[1]) + 2
        else:
            reward = -1
        vistis_time = self.vist_maze(new_p)
        if vistis_time !=0:
            reward = -3 - 0.2 * vistis_time
        # reward = 0  #If use the first reward function , set this to 0
        return reward
        

    """
    5*5 matrix with reward in current postion
    [[0.  0.  0.  0.  0. ]
    [0.  0.5 0.3 0.2 0.2]
    [0.  0.3 0.7 0.5 0.4]
    [0.  0.2 0.5 0.8 0.6]
    [0.  0.2 0.4 0.6 0.8]]
    Referring to this matrix, assign a reward to the 199*199 maze
    """

    
    def postion_reward(self,actor_position):
        maze_reward = np.zeros((200,200))
        self.aim_reward = 20
        for i in range(maze_reward.shape[0]):
            for j in range(maze_reward.shape[1]):
                if i == j:
                    maze_reward[i][j] = self.aim_reward
                    step = (self.aim_reward)/(i+1)
                    for index in range(i,-1,-1):
                        maze_reward[index][j] = int(step*index)
                        maze_reward[i][index] = int(step*index)
            
        postion_reward = maze_reward[actor_position]

        
        if self.postion_reward_flage:
            return postion_reward
        else:  #Do not use this reward 
            return 0 


    def step(self, action, score):

        
        self.step_num += 1 # increase the time

        global action_dict 
        global rewards_dict 

        #move action
        x_move, y_move = action_dict[action] 

        # waste too much time. stop the game

        # if score < -200000:
        #     print('Waste too much time')
        #     end_flag = True
        #     return self.observe_environment, -1., end_flag # terminate


       
        x, y = self.actor_postion
        prior_env = get_local_maze_information(x, y)


        #Check if actor was blocked
        block_flag = True
        for i in prior_env:
            for j in i:
                if j[0] == 1 and j[1] == 0: 
                    block_flag = False


        test_flag  = False
        end_flag = False

        #check if the agent performs the same action continuously
        if self.pre_action == action:
            self.same_ac +=1
        else:
            self.same_ac = 0

        if(self.same_ac>5000):
            end_flag = True
            print("Same action too much")

        self.pre_action = action

        
        cur_x, cur_y = (1 + x_move, 1 + y_move) 
        #The agent was blocked
        if action == 0 and block_flag: 
            
            the_reward = self.manha_reward()+rewards_dict['block']+self.postion_reward(self.actor_postion)
            if test_flag:
                print("block")
            return self.observe_environment,the_reward , end_flag
        #penalise no reseaon stay
        elif action == 0: 
            self.stay_num += 1
            
            the_reward = self.manha_reward()+rewards_dict['block']+self.postion_reward(self.actor_postion)
            if test_flag:
                print("stay")
            return self.observe_environment, the_reward, end_flag

       
        # check wall
        if prior_env[cur_x][cur_y][0] == 0: 
            self.obstacle_num += 1
            
            the_reward = self.manha_reward()+rewards_dict['block']+self.postion_reward(self.actor_postion)
            if test_flag:
                print("wall")
            return self.observe_environment, the_reward, end_flag
        # check fire
        if prior_env[cur_x][cur_y][1] > 0: 
            self.obstacle_num += 1

            the_reward = self.manha_reward()+rewards_dict['block']+self.postion_reward(self.actor_postion)
            if test_flag:
                print("fire")
            return self.observe_environment,the_reward, end_flag


        #acotr move sucessful
        self.actor_postion = (x + x_move, y + y_move) 
        self.maze_times[self.actor_postion[0]][self.actor_postion[1]] +=1

        # actor reach  goal
        if self.actor_postion == (199, 199):
            end_flag = True
            
            the_reward = self.manha_reward()+rewards_dict['block']+self.postion_reward(self.actor_postion)
            self.pre_postion = self.actor_postion


            if test_flag:
                print("end")
            return self.observe_environment, the_reward, end_flag
        #agent have not arrive the postion
        if self.actor_postion not in self.actor_path:
            
            the_reward = self.manha_reward()+rewards_dict['block']+self.postion_reward(self.actor_postion)
            self.pre_postion = self.actor_postion
            if test_flag:
                print("unvisit")
            return self.observe_environment, the_reward,end_flag
        # actoe has arrived the postion
        if self.actor_postion in self.actor_path:
            self.visited_num += 1

            the_reward = self.manha_reward()+rewards_dict['block']+self.postion_reward(self.actor_postion)
            self.pre_postion = self.actor_postion
            if test_flag:
                print("visted")
            return self.observe_environment, the_reward, end_flag

        # actor move towards the goal
        if x_move > 0 or y_move > 0:
            
            the_reward = self.manha_reward()+rewards_dict['block']+self.postion_reward(self.actor_postion)
            self.pre_postion = self.actor_postion
            if test_flag:
                print("onwards")
            return self.observe_environment, the_reward, end_flag
    
        # actor has no choice but move away from goal

        the_reward = self.manha_reward()+rewards_dict['block']+self.postion_reward(self.actor_postion)
        self.pre_postion = self.actor_postion
        if test_flag:
                print("backwards")
        return self.observe_environment, the_reward, end_flag
