import torch.nn as nn
import numpy as np
import torch
class DQN_lander(nn.Module):
    '''
    pytorch CNN model for Atari games
    '''
    def __init__(self,input_shape,num_actions):
        super(DQN_lander,self).__init__()
        self._linear=nn.Sequential(
            nn.Linear(input_shape,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,num_actions)
        )

        self.num_actions=num_actions

    def forward(self,img_in):
        '''

        :param x:input image:N*C*W*H
        :return:Q-values of actions N*num_actions
        '''
        return self._linear(img_in)

    def _selectAction(self,img_in,eps_threshold):
        '''
        select action according to Q values,
        :param img_in:input images
        :return:action selected
        '''
        sample=np.random.random()

        if(sample>eps_threshold):
            with torch.no_grad():
                q_value = self.forward(img_in)
            return q_value.max(1)[1].item()
        else:
            return np.random.randint(0,self.num_actions)



def main():
    '''
    unitest
    :return:
    '''
    import torch
    import numpy as np
    dqn=DQN_RAM(128,4)
    dqn.eval()
    img=torch.Tensor(np.zeros((1,128)))
    q=dqn.forward(img)
    print(q)
    print(q.max(1))
    print(dqn._selectAction(img,0.01))
    print("finish test")


if __name__=="__main__":
    main()
