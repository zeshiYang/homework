import pybullet as p
import numpy as np


def IsConatct(bodyID,linkID):
    info=p.getContactPoints(bodyA=bodyID,linkIndexA=linkID)
    if(info!=()):
        return True
    else:
        return False
def getComPos(bodyID):
    pos=0
    m=0
    for i in range(p.getNumJoints(bodyID)):
        temp=p.getLinkState(bodyID,i)[0][1]
        #print(temp)
        mass=p.getDynamicsInfo(bodyID,i)[0]
        #print(mass)
        pos+=temp*mass
        m+=mass
    return pos/m


def getComVel(bodyID):
    vel = 0
    m = 0
    for i in range(p.getNumJoints(bodyID)):
        temp = p.getLinkState(bodyID, i,1)[-2][1]
        # print(temp)
        mass = p.getDynamicsInfo(bodyID, i)[0]
        # print(mass)
        vel += temp * mass
        m += mass
    return vel / m

def getState(bodyID):
    pass
class BipedalState(object):
    def __init__(self):
        self.targetAngles=[]
    def setTargetAngles(self,tar):
        self.targetAngles=tar
class FSM(object):
    def __init__(self):
        self.stateID=0
        self.states=[]
        for i in range(4):
            self.states.append(BipedalState())
        pose0 = np.array([-0.1, 0.0, 0.0, 0.0, 0.25, -0.4, -0.0]) * np.pi
        pose1 = np.array([-0.1, 0.0, 0.0, 0.0, -0.2, 0.0, -0.]) * np.pi
        pose2 = np.array([-0.1, 0.25, -0.4, 0.0, 0.0, 0.0, -0.0]) * np.pi
        pose3 = np.array([-0.1, -0.2, 0.0, 0.0, 0.0, 0.0, -0.]) * np.pi
        self.pose=[]
        self.pose.append(pose0)
        self.pose.append(pose1)
        self.pose.append(pose2)
        self.pose.append(pose3)
        for i in range(4):
            self.states[i].setTargetAngles(self.pose[i])

    def transition(self,dt,bodyID,start_flag):
        if(self.stateID==0 or self.stateID==2):
            if(start_flag==False):
                if(IsConatct(bodyID,8)==True and self.stateID==0):
                    self.stateID=2
                    return
                elif (IsConatct(bodyID, 5) == True and self.stateID == 2):
                    self.stateID = 0
                    return


            if(dt>=0.4):
                if(self.stateID==0):
                    self.stateID=1
                    return
                if(self.stateID==2):
                    self.stateID=3
                    return


        else:
            if(self.stateID==1):
                if(IsConatct(bodyID,8)):
                    self.stateID=2
                    return
            if(self.stateID==3):
                if(IsConatct(bodyID,5)):
                    self.stateID=0
                    return
    def getD(self,bodyID,state):
        pos_com=getComPos(bodyID)
        pos_feet=0
        if(state==0 or state==1):
            pos_feet=p.getLinkState(bodyID,5)[0][1]
        else:
            pos_feet=p.getLinkState(bodyID,8)[0][1]
        return pos_com-pos_feet
    def feedback(self,bodyID,state):
        d=self.getD(bodyID,state)
        v=getComVel(bodyID)
        #print(v)
        #print(d)
        pos_torso=p.getJointState(bodyID,2)[0]
        #print("pos_torso:{}".format(pos_torso))
        pose0 = np.array([-0.0, 0.0, -0.05, 0.2, 0.62-pos_torso, -1.10, 0.2])
        pose1 = np.array([-0.0, 0.0, -0.1, 0.2, -0.1-pos_torso, -0.05, 0.2])
        pose2 = np.array([-0.0, 0.62-pos_torso, -1.10, 0.2, 0.0, -0.05, 0.2])
        pose3 = np.array([-0.0, -0.1-pos_torso, -0.05, 0.2, 0.0,-0.1, 0.2])
        pose = []
        pose.append(pose0)
        pose.append(pose1)
        pose.append(pose2)
        pose.append(pose3)
        for i in range(4):
            self.states[i].setTargetAngles(pose[i])


        if(state==1):
            self.states[state].setTargetAngles(pose[state])
            self.states[state].targetAngles[4] += d * 2.5
            #self.states[state].targetAngles[5] += d * 0.01 * np.pi
            #self.states[state].targetAngles[6] += d * 0.1 * np.pi
            #self.states[state].targetAngles[2] += d * 0.01 * np.pi
        if(state==3):
            self.states[state].setTargetAngles(pose[state])
            #self.states[state].targetAngles[0] += d * 0.03 * np.pi
            #self.states[state].targetAngles[4] += d * 0.01 * np.pi
            #self.states[state].targetAngles[5] += d * 0.01 * np.pi
            self.states[state].targetAngles[1] += d * 2.5
            #self.states[state].targetAngles[3] += d * 0.1 * np.pi
        if(state==0):
            self.states[state].setTargetAngles(pose[state])
            #self.states[state].targetAngles[0] += d * 0.03 * np.pi
            # self.states[state].targetAngles[4] += d * 0.01 * np.pi
            # self.states[state].targetAngles[5] += d * 0.01 * np.pi
            self.states[state].targetAngles[4] += v * 0.1
            # self.states[state].targetAngles[2] += d * 0.01 * np.pi'''



        if(state==2):
            self.states[state].setTargetAngles(pose[state])
            #self.states[state].targetAngles[0] += d * 0.03 * np.pi
            # self.states[state].targetAngles[4] += d * 0.01 * np.pi
            # self.states[state].targetAngles[5] += d * 0.01 * np.pi
            self.states[state].targetAngles[1] += v * 0.1
            # self.states[state].targetAngles[2] += d * 0.01 * np.pi





