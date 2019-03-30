import torch.nn as nn

class DQN(nn.Module):
    '''
    pytorch CNN model for Atari games
    '''
    def __init__(self,img_shape,num_actions):
        super(DQN,self).__init__()
        self._conv=nn.Sequential(
            nn.Conv2d(3,16,kernel_size=5,stride=2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16,32,kernel_size=5,stride=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,64,kernel_size=5,stride=2),
            nn.BatchNorm2d(64)
        )
        convw=img_shape[0]
        convh=img_shape[1]
        for i in range(3):
            convw=self._getConvSize(convw,5,2)
            convh=self._getConvSize(convh,5,2)
        linear_input_size=convh*convw*64
        self._linear=nn.Sequential(
            nn.Linear(linear_input_size,512),
            nn.ReLU(),
            nn.Linear(512,num_actions)
        )


    def _getConvSize(self,size,size_kernal,stride):
        '''
        get the tensor size after Conv operation
        :param size:
        :param size_kernal:
        :param stride:
        :return:
        '''
        return (size-(size_kernal-1)-1)//stride+1

    def forward(self,img_in):
        '''

        :param x:input image:N*C*W*H
        :return:Q-values of actions N*num_actions
        '''
        x=self._conv(img_in)
        x=x.view(x.size(0),-1)
        return self._linear(x)


def main():
    '''
    unitest
    :return:
    '''
    import torch
    import numpy as np
    dqn=DQN((100,100,3),4)
    img=torch.Tensor(np.zeros((1,3,100,100)))
    q=dqn.forward(img)
    print(q)
    print(q.max(1))
    print("finish test")


if __name__=="__main__":
    main()
