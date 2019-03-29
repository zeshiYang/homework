import pybullet as p
import numpy as np
import pybullet_data
from pdControllerExplicit import *
from utils import *
import os
import time

GRAVITY = -9.8
dt = 1e-3
iters = 2000

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, GRAVITY)
p.setTimeStep(1/240)
planeId = p.createCollisionShape(p.GEOM_PLANE)
p.createMultiBody(planeId, 0)
cubeStartPos = [0, 0, -0.31]
cubeStartOrientation = p.getQuaternionFromEuler([0., 0, 0])
botId = p.loadURDF("biped/biped2d_pybullet.urdf",
                   cubeStartPos,
                   cubeStartOrientation)
#dsiable unused joints
for i in range(3):
    p.setJointMotorControl2(botId,i,p.POSITION_CONTROL,force=0)
#p.setJointMotorControl2(botId,2,p.POSITION_CONTROL,force=0)
for i in range(9):
    p.enableJointForceTorqueSensor(botId,i,1)
fsm=FSM()
#initilze PDController
PdController=PDControllerExplicit(p)

import time
time_last=time.time()
dt=0
leg_move_ID=0
leg_stand_ID=0
start_flag=False
while (1):
    time_now=time.time()
    dt+=time_now-time_last
    state_last=fsm.stateID
    fsm.transition(dt,botId,start_flag)
    start_flag=False
    state_now=fsm.stateID
    if(state_last==1 and state_now==2):
        dt=0
    if(state_last==3 and state_now==0):
        dt=0
    time_last=time_now
    print(state_now)
    p.stepSimulation()
    fsm.feedback(botId,state_now)
    force_torso=PdController.computePD(botId,2,fsm.states[state_now].targetAngles[0],0,300,30,500)
    print("force_torso:{}".format(force_torso))
    if(state_now==0 or state_now==1):
        leg_move_ID=6
        leg_stand_ID=3
    else:
        leg_move_ID=3
        leg_stand_ID=6
    print(fsm.states[state_now].targetAngles)
    force_leg_move =PdController.computePD(botId,leg_move_ID,fsm.states[state_now].targetAngles[leg_move_ID-2],0,300,30,500)
    print("force_leg_move:{}".format(force_leg_move))
    force_leg_stand=-force_torso-force_leg_move
    print("force_leg_stand:{}".format(force_leg_stand))
    print("joint_pos:{}".format(p.getJointState(botId,3)[0]))
    #print("vel:{}".format(p.getLinkState(botId,3,1)[-2]))
    p.setJointMotorControl2(botId, leg_move_ID, p.TORQUE_CONTROL,force=force_leg_move)
    p.setJointMotorControl2(botId, leg_stand_ID, p.TORQUE_CONTROL,force=force_leg_stand)
    jointID=[4,5,7,8]
    for i in jointID:
        p.setJointMotorControl2(botId, i, p.POSITION_CONTROL,targetPosition=fsm.states[state_now].targetAngles[i-2], force=500.0)
    time.sleep(1/240)

time.sleep(1000)