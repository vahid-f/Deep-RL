# Deep Reinforcement Learning (DRL) 'Collaboration and Competition' Report
Udacity Deep Reinforcement Learning Nanodegree Program<br/>
P3-Collaboration and Competition <br/>


### Description
For this project, I will train a pair of agents to play tennis in a [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment. 

<img src=https://github.com/HadisAB/Collaboration-and-Competition_DRL_Udacity/blob/master/Images/env.example.png />



### Goal
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.<br/>
The task is episodic, and in order to solve the environment, our agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). <br/>

### States and actions
The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. <br/>
Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

### Environment
I have used Unity's rich environments to train and design the algorithms.<br/>
Download the used environment for Windows 64 :
[Tennis Environment](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)<br/>


### Solution
I have solved the project by Deep Deterministic Policy Gradients (DDPG) algorithm. The main used reference to solve this project is the udacity [DDPG exercises](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum). <br/>
You can find the solution by refering to [Training code](https://github.com/HadisAB/Collaboration-and-Competition_DRL_Udacity/tree/master/Training%20Codes) and using below clarification. <br/>

* I decided to start with the DDPG code to solve this project.
* Please note that each agent receives its own, local observation. Thus, we can adapt the code to simultaneously train both agents through self-play. 
* Each agent used the same actor network to select actions, and the experience was added to a shared replay buffer.
* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
* This yields a single score for each episode.
* The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5



**Collaboration-and-Competition.py**: This is the main script which shows the intersections between the environment and agent by using following functions. <br/>

**model** : This is neural model for Actor and Critic network. In these networks the actor does the policy approximation, and the critic does the value estimation. 

**Buffer** : This clarifies the codes related to used Replay Buffer.

**ddpg_agent** : This is a class of agents activites which make agent be trained.

**MADDPG** : This is an extention of ddpg_agent to multiple agents.

**train_mddpg**: This is a class of activites to traine multiple agents and our network.

**agent1_checkpoint_actor.pth**: This stores the trained weights of Actor for first agent. <br/>

**agent1_checkpoint_critic.pth**: This stores the trained weights for Critic for first agent. <br/>

**agent2_checkpoint_actor.pth**: This stores the trained weights of Actor for second agent. <br/>

**agent2_checkpoint_critic.pth**: This stores the trained weights for Critic for second agent. <br/>

### used parameters
BUFFER_SIZE = int(1e5)   &nbsp; &nbsp; replay buffer size<br/>
BATCH_SIZE = 250         &nbsp; &nbsp; minibatch size<br/>
GAMMA = 0.99            &nbsp; &nbsp; discount factor<br/>
TAU = 1e-3              &nbsp; &nbsp; for soft update of target parameters<br/>
LR_ACTOR = 1e-4         &nbsp; &nbsp; learning rate for Actor<br/>
LR_CRITIC = 1e-3        &nbsp; &nbsp; learning rate for Critic<br/>

### Result

<img src="https://github.com/HadisAB/Collaboration-and-Competition_DRL_Udacity/blob/master/Images/scores.png" />


<img src="https://github.com/HadisAB/Collaboration-and-Competition_DRL_Udacity/blob/master/Images/Avg_scores.png" />


<img src="https://github.com/HadisAB/Collaboration-and-Competition_DRL_Udacity/blob/master/Images/trend.png" />


### Ideas for future work
Here are some ideas for future work:

[Prioritized Experience Replay](https://arxiv.org/abs/1511.05952): 
> The idea behind using these technique for sampling from the replay buffer is that not all experiences are equal, some are more important than others in terms of reward, so naturally the agent should at least prioritize between the different experiences.


[Asynchronous Actor Critic](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2): 
> The idea is to have a global network and multiple agents who all interact with the environment separately and send their gradients to the global network for optimization in an asynchronous way.



