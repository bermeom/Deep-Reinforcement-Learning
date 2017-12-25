# Deep Reinforcement Learning

## Background

   Reinforcement learning is learning what to do to map situations to actions in an environment as to maximize a reward.  
   The learner is not told which actions to take, but instead must discover which actions yield the most reward by trying them (interacting with the environment).  
   A reinforcement learning agent interacts with its environment formulated as a Markov Decision Process (MDP): At each time *t*, the agent receives an observation *o(t)*, which typically includes the reward *r(t)*. It then chooses an action a *a(t)* from the set of available actions available, which is subsequently sent to the environment. The environment moves to a new state *s(t+1)* and the reward *r(t+1)* associated with the transition *s(t) , a(t), s(t+1)* is determined. The goal of a reinforcement learning agent is to collect as much reward as possible.

![Picture](http://web.stanford.edu/class/cs234/images/header2.png)

## Tensorblock

Tensorblock is a Google's TensorFlow API, developed by Vitor Guizilini to facilitate the implementation of Machine Learning Algorithms: [BitBucket/TensorBlock](https://bitbucket.org/vguizilini/tensorblock/overview).  

See the documentation: [TensorBlock](docs/TensorBlock.md)

## Environments:

The environments currently supported are:

- Gym 
- Gym Mujoco
- PyGame
- Unity Machine Learning

See the documentation: [Environments](docs/Environments.md)

## Algorithms:

The Reinforcement Learning Algorithms currently implemented are:  

- [Q-Learning](docs/QLearning.md)
- [REINFORCE](docs/REINFORCE.md)
- [Actor Critic](docs/ActorCritic.md)
- [DDPG](docs/DDPG.md)
- [PPO](docs/PPO.md)

## Results

Some Results:

- [Result Comparison](statistics/comparison.md)

## Versions

- Python 3.5
- Tensorflow 1.3
- Pygame 1.9.3
- OpenCV-Python 3.3.0.9
- NumPy 1.13.1
- Gym 0.9.2
- MatplotLib 2.0.2

## Bibliography

- [Bibliography](docs/Bibliography.md)

## Acknowledgement

TensorBlock is developed by Vitor Guizilini. Many thanks for his guidance and support. Also, I thank Professor Glauco Caurin and all colleagues from the Mechatronics Laboratory of the Engineering School of São Carlos, and the support from CNPq.   
