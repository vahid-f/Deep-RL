# Navigation-DRL-Udacity
P1-Navigation- Udacity Deep Reinforcement Learning Nanodegree Program

## Project details
For this project, I will train an agent to navigate (and collect yellow bananas!) in a large, square world.


<img src="https://github.com/HadisAB/Navigation-DRL-Udacity/blob/master/images/envexample.png" />


### Goal
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of agent is to collect as many yellow bananas as possible while avoiding blue bananas.
The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

### States and actions
The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:<br/>

0 - move forward.<br/>
1 - move backward.<br/>
2 - turn left.<br/>
3 - turn right.

## Getting Started

1. Download [Anaconda](https://www.anaconda.com/distribution/) and install it.
2. Run the 'anaconda prompt' and use below commands to creat an environment and install python 3.6
> conda create --name drlnd python=3.6 <br/>
> activate drlnd 
3. Open anaconda navigator, select drland in the anaconda environments, install and launch spyder. I have used spyder to run my codes. You may use another interface. 
4. I have used Unity's rich environments to train and design the algorithms.<br/>
Download the used environment for Windows 64 :
[Banan Envoronment](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)<br/>
5. Install [unity ml-agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md), [numpy](https://numpy.org/) and [pytorch](https://pytorch.org/) through shown links.
6. Install matplotlib.pyplot by using below command in 'anaconda prompt':<br/>
> pip install matplotlib.pyplot



## Instructions
I have solved the project by Deep Q-Networks and also considering the 'Experience Replay' and 'Fixed Q Targets' to improvement the training. The main used reference to solve this project is the udacity [DQN exercises](https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn). <br/>

Follow below steps to run the code:
1. Install all required dependencies of the project based on above links.
2. Download the [dqn-agent.py](https://github.com/HadisAB/Navigation-DRL-Udacity/blob/master/Training%20code/dqn_agent.py), [model.py](https://github.com/HadisAB/Navigation-DRL-Udacity/blob/master/Training%20code/model.py) and [Navigation.py](https://github.com/HadisAB/Navigation-DRL-Udacity/blob/master/Training%20code/navigation.py) files from [training code](https://github.com/HadisAB/Navigation-DRL-Udacity/tree/master/Training%20code).
3. Put all the downloded .py files in one folder.
4. Open the [Navigation.py](https://github.com/HadisAB/Navigation-DRL-Udacity/blob/master/Training%20code/navigation.py) in spyder and Run the code.


Also refer to the [report](https://github.com/HadisAB/Navigation-DRL-Udacity/blob/master/Report.md) for more clarification of the method. 




