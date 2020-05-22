# Collaboration-and-Competition_DRL_Udacity
This is the third project of Nano-degree Deep Reinforcement Learning Udacity .


## Project details
For this project, I will train a pair of agents to play tennis in a [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment. 

<img src=https://github.com/HadisAB/Collaboration-and-Competition_DRL_Udacity/blob/master/Images/env.example.png />


### Goal
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.<br/>
The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). <br/>
Specifically,

* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.<br/>
* This yields a single score for each episode.<br/>

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5


### States and actions
The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. <br/>
Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

## Getting Started

1. Download [Anaconda](https://www.anaconda.com/distribution/) and install it.
2. Run the 'anaconda prompt' and use below commands to creat an environment and install python 3.6
> conda create --name drlnd python=3.6 <br/>
> activate drlnd 
3. Open anaconda navigator, select drland in the anaconda environments, install and launch spyder. I have used spyder to run my codes. You may use another interface. 
4. I have used Unity's rich environments to train and design the algorithms.<br/>
Download the used environment for Windows 64 :
[Tennis Environment](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)<br/>
5. Install [unity ml-agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md), [numpy](https://numpy.org/) and [pytorch](https://pytorch.org/) through shown links.
6. Install matplotlib.pyplot by using below command in 'anaconda prompt':<br/>
> pip install matplotlib.pyplot



## Instructions
I have solved the project by Deep Deterministic Policy Gradients (DDPG) algorithm. The main used reference to solve this project is the udacity [DDPG exercises](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum). <br/>

Follow below steps to run the code:
1. Install all required dependencies of the project based on above links.
2. Download the [Collaboration and Competitionl.py](https://github.com/vahid-f/Deep-RL/blob/master/Collaboration-and-Competition_DRL_Udacity-master/Collaboration-and-Competition_DRL_Udacity-master/Training%20Codes/Collaboration%20and%20Competition.py) files from [Training code](https://github.com/HadisAB/Continous-Control---DRL---Udacity/tree/master/Training%20Code).
3. Put all the downloded .py files in one folder.
4. Open the [Collaboration and Competitionl.py](https://github.com/vahid-f/Deep-RL/blob/master/Collaboration-and-Competition_DRL_Udacity-master/Collaboration-and-Competition_DRL_Udacity-master/Training%20Codes/Collaboration%20and%20Competition.py) in spyder and Run the code.


Also refer to the [report](https://github.com/vahid-f/Deep-RL/blob/master/Collaboration-and-Competition_DRL_Udacity-master/Collaboration-and-Competition_DRL_Udacity-master/Report.md) for more clarification of the method. 


