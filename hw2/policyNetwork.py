import torch
import torch.nn as nn
import numpy as np
import gym
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.distributions import MultivariateNormal
from tensorboardX import SummaryWriter
torch.manual_seed(0)


class Vnetwork(nn.Module):
    def __init__(self,input_size):
        super(Vnetwork,self).__init__()
        self.main=nn.Sequential(
            nn.Linear(input_size,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )
    def forward(self, x):
        return self.main(x)


class PolicyNetwork(nn.Module):
    def __init__(self,input_size,out_put_size,is_discrete=False):
        super(PolicyNetwork,self).__init__()
        self.is_discrete=is_discrete
        if(is_discrete==False):
            self.main=nn.Sequential(
                nn.Linear(input_size,128),
                nn.Tanh(),
                nn.Linear(128,out_put_size),
                nn.Tanh()
            )
            self.sigma=Variable(torch.Tensor([-2]),requires_grad=True)
        else:
            self.main = nn.Sequential(
                nn.Linear(input_size, 32,bias=False),
                nn.ReLU(),
                nn.Linear(32, out_put_size,bias=False),
                nn.Softmax(dim=1)
            )
        self.log_prob_list=[]



    def sampleAction(self, x):
        if(self.is_discrete==True):
            x=self.main(x)
            x=Categorical(x)
            ac=x.sample()
            self.log_prob_list.append(x.log_prob(ac))
            return ac
        else:
            x=self.main(x)
            if(self.sigma.item()<-3):
                self.sigma = Variable(torch.Tensor([-3]), requires_grad=True)
            sigma=torch.exp(self.sigma)
            #ac=x+torch.exp(self.sigma)*torch.randn(x.size()[0],x.size()[1])
            x=MultivariateNormal(x,sigma*torch.eye(x.size()[1]))
            ac=x.sample()
            self.log_prob_list.append(x.log_prob(ac))
            return ac


    def forward(self, x):
        return self.main(x)

    def save(self,path):
        torch.save(self.state_dict(),path)

    def restore(self,path):
        self.load_state_dict(torch.save(path))


class agent(object):
    def __init__(self,env_name,gamma,max_step,batch_size):
        self.env=gym.make(env_name)
        self.gamma=gamma
        self.max_step=min(max_step,self.env.spec.max_episode_steps)
        self.discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.obv_dim = self.env.observation_space.shape[0]
        self.ac_dim = self.env.action_space.n if self.discrete else self.env.action_space.shape[0]
        self.batch_size=batch_size
        self.replay_buffer=None
    def sample_trajectory(self,model,max_step):
        obs, acs, rewards = [], [], []
        self.env.seed(1)

        ob = self.env.reset()
        steps = 0
        while True:
            obs.append(ob)
            ac = model.sampleAction((torch.Tensor(ob[None,:])))
            ac = ac[0]
            ac=ac.detach().numpy()
            acs.append(ac)
            ob, rew, done, _ = self.env.step(ac)
            rewards.append(rew)
            steps += 1
            if done or steps > max_step:
                break
        path = {"observation": np.array(obs, dtype=np.float32),
                "reward": np.array(rewards, dtype=np.float32),
                "action": np.array(acs, dtype=np.float32)}
        return path
    def sample_trajectories(self,model,max_step):
        timesteps_this_batch = 0
        paths = []
        while True:
            path = self.sample_trajectory(model,max_step)
            paths.append(path)
            timesteps_this_batch += len(path["reward"])
            if timesteps_this_batch > 20000:
                break
        return paths, timesteps_this_batch


def train_test(env_name,iter_num,gamma,max_step,batch_size):
    #writer=SummaryWriter()
    agnet_train=agent(env_name,gamma,max_step,batch_size)
    model=PolicyNetwork(agnet_train.obv_dim,agnet_train.ac_dim,agnet_train.discrete)
    Vnet=Vnetwork(agnet_train.obv_dim)
    optimizer=optim.Adam(model.parameters(),lr=0.005)
    opt_sigma=optim.Adam([model.sigma],lr=0.01)
    #for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        print(name)
    for iter in range(iter_num):
        optimizer.zero_grad()
        opt_sigma.zero_grad()
        reward_all=[]
        reward_discount_all=[]
        obv_start_all=[]
        obv_next_all=[]
        obv_end_state=[]
        reward_end=[]
        obv_all=[]
        num=0
        return_all=[]
        while(True):
            traj=agnet_train.sample_trajectory(model,150)
            reward=traj["reward"]
            return_all.append(np.sum(reward))
            reward_discount=traj['reward']
            obv=traj['observation']
            for i in range(len(obv)-1):
                obv_start_all.append(obv[i])
                obv_next_all.append(obv[i+1])
                reward_all.append(reward[i])
            obv_end_state.append(obv[-1])
            reward_end.append(0)

            #compute discount reward return
            for i in range(len(reward)):
                temp=[]
                for j in range(len(reward[i:])):
                    temp.append(pow(gamma,j-1))
                reward_discount[i]=np.sum(np.array(temp)*np.array(reward[i:]))


            for i in range(len(reward)):
                obv_all.append(obv[i])
                reward_discount_all.append(reward_discount[i])
                num+=1
            if(num>30000):
                break
        #network baseline
        #obv_start_all,obv_next_all,reward_all
        '''id=len(obv_start_all)
        obv_start_all=torch.Tensor(np.concatenate([np.array(obv_start_all),np.array(obv_end_state)],axis=0))
        obv_next_all=torch.Tensor(np.concatenate([np.array(obv_next_all),np.array(obv_end_state)],axis=0))
        reward_all=torch.Tensor(np.concatenate([np.array(reward_all),np.array(reward_end)],axis=0)).view(-1,1)
        criterion = nn.MSELoss()
        optimizer_baseline = optim.Adam(Vnet.parameters(), lr=0.001)
        for i in range(500):
            with torch.no_grad():
                V=reward_all+gamma*Vnet(obv_next_all)
                V[id:,:]=0
                mean = V.mean(dim=0)
                std = V.std(dim=0)
                V=(V-V.mean(dim=0)/V.std(dim=0))
            output=Vnet(obv_start_all)
            optimizer_baseline.zero_grad()
            loss_v=criterion(V,output)
            loss_v.backward()
            #print(loss_v.item())
            optimizer_baseline.step()

'''

        return_mean = np.mean(return_all)
        return_std=np.std(return_all)
        reward_mean = torch.Tensor([np.mean(reward_discount_all)])
        rewatd_std = torch.Tensor([np.std(reward_discount_all)])
        reward_discount_all=(reward_discount_all-np.mean(reward_discount_all))/(np.std(reward_discount_all)+0.0001)
        temp=[]
        for i in range(len(reward_discount_all)):
            #baseline=Vnet(torch.Tensor(obv_all[i]))*std+mean
            #baseline=(baseline- reward_mean) / rewatd_std
            #temp.append(-model.log_prob_list[i]*(torch.Tensor([reward_discount_all[i]])-baseline))
            temp.append(-model.log_prob_list[i] * (torch.Tensor([reward_discount_all[i]])))
        loss=torch.cat(temp).sum()
        loss.backward()
        optimizer.step()
        opt_sigma.step()
        del model.log_prob_list[:]
        #writer.add_scalar("data/mean",return_mean,iter)

        print("iter:{}".format(iter))
        #print("Vfunction loss:{}".format(loss_v.item()))
        print("sigma:{}".format(torch.exp(model.sigma).item()))
        print("reward_mean:{}".format(return_mean))
        print("reward_std:{}".format(return_std))
        if(iter%50==1):
            torch.save(model.state_dict(),"./P.pkl")
            torch.save(Vnet.state_dict(),"./V.pkl")
