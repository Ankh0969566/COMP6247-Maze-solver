
import numpy as np



"""
This class implements memory in the experience replay mechanism
"""

class Memory():
    def __init__(self, memory_size,n_features):

        #Memory bank size
        self.memory_size = memory_size
        self.n_features = n_features
        self.relpace_index = 0
        self.memory_dataset = np.zeros((self.memory_size, self.n_features*2+2))
        self.test_flag = False
    def get_memory(self):
        return self.memory_dataset
        
    def save_memory(self,state, action, reward, state_):

        new_memory = np.hstack((state, [action, reward], state_))
		# replace the old memory with new memory when the memory is full
        index = self.relpace_index % self.memory_size
            


        self.memory_dataset[index, :] = new_memory 
        self.relpace_index += 1
#Sample from the memory
    def batch_sample(self,memory_size,batch_size):
        #Insufficient number of experiences to extract all memories
        if self.relpace_index > memory_size:
            sample_index = np.random.choice(memory_size, size = batch_size)
        #Sufficient number of experiences, extracted by batch
        else:
            sample_index = np.random.choice(self.relpace_index, size = batch_size)
        batch_memory = self.memory_dataset[sample_index, :]    
        return batch_memory


