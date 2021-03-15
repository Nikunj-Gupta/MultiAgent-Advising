import gym 
import math, random, os, numpy as np 
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F 
from torch.autograd import Variable

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor
Tensor = FloatTensor


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Network(nn.Module):
    def __init__(self, layer_dims): # layer_dims must include input and output dims too. 
        super(Network, self).__init__() 
        self.linears = nn.ModuleList([nn.Linear(layer_dims[i], layer_dims[i+1]) for i in range(len(layer_dims)-1)]) 

    def forward(self, x):
        for i in range(len(self.linears)-1):
            x = F.relu(self.linears[i](x)) 
            x = self.linears[i+1](x)         
        return x 

class DQN: 
    def __init__(self, state_dim, action_dim, config=None, use_teacher=False, teacher_model=None, advice_threshold=0.1): 
        if not config: 
            config = {
                # hyper parameters
                "EPS_START": 1.0,  # e-greedy threshold start value 
                "EPS_END": 0.05,  # e-greedy threshold end value 
                "EPS_DECAY": 2000,  # e-greedy threshold decay 
                "GAMMA": 0.95,  # Q-learning discount factor 
                "LR": 0.001,  # NN optimizer learning rate 
                "HIDDEN_LAYER": [64, 64],  # NN hidden layers' sizes 
                "BATCH_SIZE": 64  # Q-learning batch size 
            } 
        self.config = config 

        self.use_teacher = use_teacher 
        if self.use_teacher: 
            self.teacher = DQN(state_dim, action_dim) 
            self.teacher.load_model(teacher_model) 
            print("Teacher loaded!") 
            self.advice_threshold = advice_threshold 
            print("Teacher Advice Threshold set to: ", self.advice_threshold) 

        self.EPS_START = self.config["EPS_START"]
        self.EPS_END = self.config["EPS_END"] 
        self.EPS_DECAY = self.config["EPS_DECAY"] 
        self.GAMMA = self.config["GAMMA"] 
        self.LR = self.config["LR"] 
        self.HIDDEN_LAYER = self.config["HIDDEN_LAYER"] 
        self.BATCH_SIZE = self.config["BATCH_SIZE"] 
        
        layer_dims = [state_dim] 
        layer_dims.extend(self.HIDDEN_LAYER) 
        layer_dims.append(action_dim) 
       
        self.model = Network(layer_dims=layer_dims) 
        print("\nModel:\n") 
        print(self.model) 
        # for name, param in self.model.named_parameters(): 
        #     if param.requires_grad: 
        #         print(name, param.data) 
       
        self.memory = ReplayMemory(10000) 

        self.optimizer = optim.Adam(self.model.parameters(), self.LR)
        self.steps_done = 0 
 
    def save_in_memory(self, state, action, next_state, reward): 
        self.memory.push((FloatTensor([state]),
                     action,  # action is already a tensor
                     FloatTensor([next_state]),
                     FloatTensor([reward])))

    def select_action(self, state, train=True): 
        state = FloatTensor([state]) 
        sample = random.random()
        self.eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if train: 
            if sample > self.eps_threshold:
                if self.use_teacher: 
                    """ 
                    Importance Advising 
                    --> Asking for Teacher's advice when the student is not confident enough among the actions it can choose 
                    
                    """
                    diff = np.array(self.model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[0])[0] - \
                        np.array(self.model(Variable(state, volatile=True).type(FloatTensor)).data.min(1)[0])[0] 
                    if diff < self.advice_threshold: 
                        return self.teacher.model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
                return self.model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
            else:
                return LongTensor([[random.randrange(2)]])
        else:
            return self.model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1) 
    
    def learn(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        # random transition batch is taken from experience replay memory
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

        batch_state = Variable(torch.cat(batch_state))
        batch_action = Variable(torch.cat(batch_action))
        batch_reward = Variable(torch.cat(batch_reward).unsqueeze(-1))
        batch_next_state = Variable(torch.cat(batch_next_state))

        # current Q values are estimated by NN for all actions
        current_q_values = self.model(batch_state).gather(1, batch_action)
        # expected Q values are estimated from actions which gives maximum Q value
        max_next_q_values = self.model(batch_next_state).detach().max(1)[0]
        expected_q_values = batch_reward + (self.GAMMA * max_next_q_values.unsqueeze(-1))

        # loss is measured from error between current and newly expected Q values
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)

        # backpropagation of loss to NN
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 
    
    def save_model(self, path, expname, episode): # path: with expname 
        if not os.path.exists(os.path.join(path, expname)): os.makedirs(os.path.join(path, expname)) 
        torch.save(self.model.state_dict(), os.path.join(path, expname, "checkpoint"+str(episode)+"--model.pth")) 

    def load_model(self, path): 
        print("Model loaded") 
        self.model.load_state_dict(torch.load(path)) 


def botPlay():
    state = env.reset() 
    frames = []
    while True:
        frame = env.render(mode='rgb_array')
        frames.append(frame)
        action = select_action(FloatTensor([state]))
        next_state, reward, done, _ = env.step(action[0, 0].item())

        state = next_state

        if done:
            break

    clip = ImageSequenceClip(frames, fps=20)
    clip.write_gif('test2.gif', fps=20)
