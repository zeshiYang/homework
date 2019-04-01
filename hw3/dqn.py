import uuid
import time
import pickle
import sys
import gym.spaces
from collections import namedtuple
from dqn_utils import *
from DQN_model import *
import torch.optim as optim
import torch
import IPython
from tensorboardX import SummaryWriter
from DQN_RAM import *
from DQN_lander import *

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

class QLearner(object):

  def __init__(
    self,
    env,
    lr_schedule,
    exploration=LinearSchedule(1000000, 0.1),
    stopping_criterion=None,
    replay_buffer_size=1000000,
    batch_size=32,
    gamma=0.99,
    learning_starts=50000,
    learning_freq=4,
    frame_history_len=4,
    target_update_freq=10000,
    grad_norm_clipping=1,
    rew_file=None,
    double_q=True,
    lander=False):
    """Run Deep Q-learning algorithm.

    You can specify your own convnet using q_func.

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            img_in: tf.Tensor
                tensorflow tensor representing the input image
            num_actions: int
                number of actions
            scope: str
                scope in which all the model related variables
                should be created
            reuse: bool
                whether previously created variables should be reused.
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    session: tf.Session
        tensorflow session to use.
    exploration: rl_algs.deepq.utils.schedules.Schedule
        schedule for probability of chosing random action.
    stopping_criterion: (env, t) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    grad_norm_clipping: float or None
        If not None gradients' norms are clipped to this value.
    double_q: bool
        If True, then use double Q-learning to compute target values. Otherwise, use vanilla DQN.
        https://papers.nips.cc/paper/3964-double-q-learning.pdf
    """
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete

    self.target_update_freq = target_update_freq
    self.batch_size = batch_size
    self.learning_freq = learning_freq
    self.learning_starts = learning_starts
    self.stopping_criterion = stopping_criterion
    self.env = env
    self.exploration = exploration
    self.rew_file = str(uuid.uuid4()) + '.pkl' if rew_file is None else rew_file
    self.gamma=gamma
    self.writer=SummaryWriter()
    self.lr_schedule=lr_schedule

    ###############
    # BUILD MODEL #
    ###############

    if len(self.env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = self.env.observation_space.shape
    else:
        img_h, img_w, img_c = self.env.observation_space.shape
        input_shape = (img_h, img_w, frame_history_len * img_c)
    self.num_actions = self.env.action_space.n
    print(input_shape)
    #Qnet and target Qnet
    self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.policy_net=DQN_lander(input_shape[0],self.num_actions).to(self.device)
    self.target_net=DQN_lander(input_shape[0],self.num_actions).to(self.device)
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()

    self.optimizer = optim.Adam(self.policy_net.parameters(),lr=1e-4)

    # set up placeholders
    # placeholder for current observation (or state)
    # placeholder for end of episode mask
    # this value is 1 if the next state corresponds to the end of an episode,
    # in which case there is no Q-value at the next state; at the end of an
    # episode, only the current state reward contributes to the target, not the
    # next state Q-value (i.e. target is just rew_t_ph, not rew_t_ph + gamma * q_tp1)

    # casting to float on GPU ensures lower data transfer times.

    # Here, you should fill in your own code to compute the Bellman error. This requires
    # evaluating the current and next Q-values and constructing the corresponding error.
    # TensorFlow will differentiate this error for you, you just need to pass it to the
    # optimizer. See assignment text for details.
    # Your code should produce one scalar-valued tensor: total_error
    # This will be passed to the optimizer in the provided code below.
    # Your code should also produce two collections of variables:
    # q_func_vars
    # target_q_func_vars
    # These should hold all of the variables of the Q-function network and target network,
    # respectively. A convenient way to get these is to make use of TF's "scope" feature.
    # For example, you can create your Q-function network with the scope "q_func" like this:
    # <something> = q_func(obs_t_float, num_actions, scope="q_func", reuse=False)
    # And then you can obtain the variables like this:
    # q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
    # Older versions of TensorFlow may require using "VARIABLES" instead of "GLOBAL_VARIABLES"
    # Tip: use huber_loss (from dqn_utils) instead of squared error when defining self.total_error
    ######

    # YOUR CODE HERE

    ######



    # construct the replay buffer
    self.replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len, lander=lander)
    self.replay_buffer_idx = None

    ###############
    # RUN ENV     #
    ###############
    self.model_initialized = True
    self.num_param_updates = 0
    self.mean_episode_reward      = -float('nan')
    self.best_mean_episode_reward = -float('inf')
    self.last_obs = self.env.reset()
    self.log_every_n_steps = 10000

    self.start_time = None
    self.t = 0

  def stopping_criterion_met(self):
    return self.stopping_criterion is not None and self.stopping_criterion(self.env, self.t)

  def step_env(self):
    print("t:{}".format(self.t))
    #print("start sampling")
    ### 2. Step the env and store the transition
    # At this point, "self.last_obs" contains the latest observation that was
    # recorded from the simulator. Here, your code needs to store this
    # observation and its outcome (reward, next observation, etc.) into
    # the replay buffer while stepping the simulator forward one step.
    # At the end of this block of code, the simulator should have been
    # advanced one step, and the replay buffer should contain one more
    # transition.
    # Specifically, self.last_obs must point to the new latest observation.
    # Useful functions you'll need to call:
    # obs, reward, done, info = env.step(action)
    # this steps the environment forward one step
    # obs = env.reset()
    # this resets the environment if you reached an episode boundary.
    # Don't forget to call env.reset() to get a new observation if done
    # is true!!
    # Note that you cannot use "self.last_obs" directly as input
    # into your network, since it needs to be processed to include context
    # from previous frames. You should check out the replay buffer
    # implementation in dqn_utils.py to see what functionality the replay
    # buffer exposes. The replay buffer has a function called
    # encode_recent_observation that will take the latest observation
    # that you pushed into the buffer and compute the corresponding
    # input that should be given to a Q network by appending some
    # previous frames.
    # Don't forget to include epsilon greedy exploration!
    # And remember that the first time you enter this loop, the model
    # may not yet have been initialized (but of course, the first step
    # might as well be random, since you haven't trained your net...)
    idx=self.replay_buffer.store_frame(self.last_obs)
    img_in=self.replay_buffer.encode_recent_observation()
    #img_in=img_in.transpose((2,0,1))
    #normalize the input image
    #IPython.embed()
    #img_in=img_in/255.0
    img_in=torch.Tensor(img_in).to(self.device).unsqueeze(0)
    #print(img_in.size())
    action=self.policy_net._selectAction(img_in,self.exploration.value(self.t))
    obs, reward, done, info = self.env.step(action)
    if(done==True):
      obs=self.env.reset()
    self.last_obs=obs
    self.replay_buffer.store_effect(idx,action,reward,done)

    #####

    # YOUR CODE HERE

  def update_model(self):
    ### 3. Perform experience replay and train the network.
    # note that this is only done if the replay buffer contains enough samples
    # for us to learn something useful -- until then, the model will not be
    # initialized and random actions should be taken
    if (self.t > self.learning_starts and \
        self.t % self.learning_freq == 0 and \
        self.replay_buffer.can_sample(self.batch_size)):
      #print("update model")
      obs_batch, act_batch, rew_batch, next_obs_batch, done_mask_batch=self.replay_buffer.sample(self.batch_size)
      rew_batch=torch.from_numpy(rew_batch).to(self.device).float().unsqueeze(1)
      #obs_batch=obs_batch.transpose((0,3,1,2))/255.0
      obs_batch=torch.from_numpy(obs_batch).to(self.device).float()
      act_batch=torch.from_numpy(act_batch).to(self.device).long().unsqueeze(1)
      #next_obs_batch = next_obs_batch.transpose((0, 3, 1, 2)) / 255.0
      next_obs_batch=torch.from_numpy(next_obs_batch).to(self.device).float()
      done_mask_batch=torch.from_numpy(done_mask_batch).to(self.device).unsqueeze(1)


      predicted_qvalues=self.policy_net(obs_batch).gather(1,act_batch)
      next_state_qvalues=torch.mul(self.target_net(next_obs_batch).max(1)[0].detach().unsqueeze(1), 1-done_mask_batch)
      #IPython.embed()
      true_qvalues=rew_batch+self.gamma*next_state_qvalues

      loss=torch.nn.functional.smooth_l1_loss(predicted_qvalues,true_qvalues)

      adjust_learning_rate(self.optimizer, self.lr_schedule.value(self.t))


      self.optimizer.zero_grad()
      loss.backward()
      self.writer.add_scalar("data/loss",loss.item(),self.t)
      self.writer.add_scalar("data/lr",self.lr_schedule.value(self.t) , self.t)
      self.writer.add_scalar("data/num_in_buffer", self.replay_buffer.num_in_buffer, self.t)
      for para in self.policy_net.parameters():
        para.grad.data.clamp(-10,10)
      self.optimizer.step()

      if(self.t%self.target_update_freq==0):
        self.target_net.load_state_dict(self.policy_net.state_dict())



      # Here, you should perform training. Training consists of four steps:
      # 3.a: use the replay buffer to sample a batch of transitions (see the
      # replay buffer code for function definition, each batch that you sample
      # should consist of current observations, current actions, rewards,
      # next observations, and done indicator).
      # 3.b: initialize the model if it has not been initialized yet; to do
      # that, call
      #    initialize_interdependent_variables(self.session, tf.global_variables(), {
      #        self.obs_t_ph: obs_t_batch,
      #        self.obs_tp1_ph: obs_tp1_batch,
      #    })
      # where obs_t_batch and obs_tp1_batch are the batches of observations at
      # the current and next time step. The boolean variable model_initialized
      # indicates whether or not the model has been initialized.
      # Remember that you have to update the target network too (see 3.d)!
      # 3.c: train the model. To do this, you'll need to use the self.train_fn and
      # self.total_error ops that were created earlier: self.total_error is what you
      # created to compute the total Bellman error in a batch, and self.train_fn
      # will actually perform a gradient step and update the network parameters
      # to reduce total_error. When calling self.session.run on these you'll need to
      # populate the following placeholders:
      # self.obs_t_ph
      # self.act_t_ph
      # self.rew_t_ph
      # self.obs_tp1_ph
      # self.done_mask_ph
      # (this is needed for computing self.total_error)
      # self.learning_rate -- you can get this from self.optimizer_spec.lr_schedule.value(t)
      # (this is needed by the optimizer to choose the learning rate)
      # 3.d: periodically update the target network by calling
      # self.session.run(self.update_target_fn)
      # you should update every target_update_freq steps, and you may find the
      # variable self.num_param_updates useful for this (it was initialized to 0)
      #####

      # YOUR CODE HERE

      self.num_param_updates += 1

    self.t += 1

  def log_progress(self):
    episode_rewards = get_wrapper_by_name(self.env, "Monitor").get_episode_rewards()

    if len(episode_rewards) > 0:
      self.mean_episode_reward = np.mean(episode_rewards[-100:])

    if len(episode_rewards) > 100:
      self.best_mean_episode_reward = max(self.best_mean_episode_reward, self.mean_episode_reward)

    if self.t % self.log_every_n_steps == 0 and self.model_initialized:
      self.writer.add_scalar('data/mean reward (100 episodes)',self.mean_episode_reward,self.t)
      self.writer.add_scalar('data/best mean reward (100 episodes)', self.best_mean_episode_reward, self.t)
      self.writer.add_scalar('data/episodes', len(episode_rewards), self.t)
      self.writer.add_scalar('data/exploration', self.exploration.value(self.t), self.t)
      print("Timestep %d" % (self.t,))
      print("mean reward (100 episodes) %f" % self.mean_episode_reward)
      print("best mean reward %f" % self.best_mean_episode_reward)
      print("episodes %d" % len(episode_rewards))
      print("exploration %f" % self.exploration.value(self.t))
      #print("learning_rate %f" % self.optimizer_spec.lr_schedule.value(self.t))
      if self.start_time is not None:
        print("running time %f" % ((time.time() - self.start_time) / 60.))

      self.start_time = time.time()

      sys.stdout.flush()

      with open(self.rew_file, 'wb') as f:
        pickle.dump(episode_rewards, f, pickle.HIGHEST_PROTOCOL)

def learn(*args, **kwargs):
  alg = QLearner(*args, **kwargs)
  while not alg.stopping_criterion_met():
    alg.step_env()
    # at this point, the environment should have been advanced one step (and
    # reset if done was true), and self.last_obs should point to the new latest
    # observation
    alg.update_model()
    alg.log_progress()

