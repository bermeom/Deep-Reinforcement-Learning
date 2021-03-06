[DEPRECATED]
THIS REPOSITORY IS DEPRECATED, PLEASE CONSIDER USING:  
https://github.com/NiloFreitas/Deep-RL-and-IL


# Deep Reinforcement Learning

## Background

   Reinforcement learning is learning what to do to map situations to actions in an environment as to maximize a reward.  
   The learner is not told which actions to take, but instead must discover which actions yield the most reward by trial and error.  
   A reinforcement learning agent interacts with its environment formulated as a Markov Decision Process (MDP): At each time *t*, the agent receives an observation *o(t)*, and a reward *r(t)*. It then chooses an action a *a(t)*, which is sent to the environment. The environment moves to a new state *s(t+1)*. The goal of a reinforcement learning agent is to collect as much reward as possible.

A video on these implementations is available: [YouTube](https://www.youtube.com/watch?v=cEcCHs1GGpg)

## Algorithms:

The Reinforcement Learning Algorithms currently implemented are:  

- [Deep Q-Learning](docs/QLearning.md)
- [REINFORCE](docs/REINFORCE.md)
- [Actor Critic](docs/ActorCritic.md)
- [Deep Deterministic Policy Gradients (DDPG)](docs/DDPG.md)
- [Proximal Policy Optimization (PPO)](docs/PPO.md)

## Environments:

The environments currently supported are:

- OpenAI's Gym
- OpenAI's Gym MuJoCo
- PyGame
- Unity Machine Learning
- CARLA

See the documentation: [Environments](docs/Environments.md)

## Tensorblock

<div align="center">

<img align="center" width="400" src="docs/images/tBlockDiagram.png">

</div>
<br />

Tensorblock is an API to facilitate the implementation of Machine Learning Algorithms using TensorFlow: [BitBucket/TensorBlock](https://bitbucket.org/vguizilini/tensorblock/).  

See the documentation: [TensorBlock](docs/TensorBlock.md)

## Results

In the [statistics](statistics/) folder are trained models and some graphics for each algorithm.

## Dependencies

- Python 3.5
- Tensorflow 1.4
- Pygame 1.9.3
- OpenCV-Python 3.3.0.9
- NumPy 1.13.1
- Gym 0.10.3
- MatplotLib 2.0.2
- Unity ML 0.3
- MuJoCo 1.50
- CARLA 0.8.3

## Bibliography

- [Bibliography](docs/Bibliography.md)

## Contributions

Are more than welcome!

## Acknowledgement

TensorBlock was developed by Vitor Guizilini. Many thanks for his guidance and support. Also, I thank Professor Glauco Caurin and all colleagues from the Mechatronics Laboratory of the São Carlos School of Engineering / University of São Paulo, and the support from CNPq.   
