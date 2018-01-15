### Proximal Policy Optimization

- PPO  
	- [player_PPO_1](../reinforcement/players/player_PPO_1.py)
	- [player_PPO_1A](../reinforcement/players/player_PPO_1A.py)

See [TensorBlock](./TensorBlock.md) documentation.  

It has been tested on the Unity Machine Learning Environments.
See the results on [PPO Statistics](https://github.com/NiloFreitas/Deep-Reinforcement-Learning/tree/master/statistics/PPO%201A) and the [video](https://youtu.be/0cBAjqQ8nw4).  

To run, please put the Unity 3DBall [build files](https://drive.google.com/drive/folders/13_uD0QtYc8fzWaVAz5auVNzk9WQUOqi4?usp=sharing) on *reinforcement/sources/unity/* and run [script4](../reinforcement/script4.sh) or on Unity with the .bytes file.  

### Overview

The Proximal Policy Optimization Algorithm works by alternating between colecting batches of samples and optimizing the policy by a number of epochs with a “surrogate” objective function using stochastic gradient ascent. It can be used to solve many high-dimensional continuous and discrete problems  

The algorithm has an Actor-Critic architecture and uses the Generalized Advantage Estimation.  

The gradients of the Actor comes from an objective function, clipped by a parameter epsilon, that is the ratio of probabilities of the actions given by the Actor Network and an Old Actor Network multiplied by the advantage. The Old Actor is not updated by the optimizer, but has its weights copied from the Actor Network before each epoch of updates.  

The gradients of the Critic Network, that provides the values for the advantage estimation, comes from the TD errors.  

This implementation does not share parameters between the Actor and the Critic.  
It does not have parallel actors, so the sampled batch comes from a single agent.  
It does not uses an entropy cost.  
The Algorithm works on continuous or discrete action spaces environments, but this implementation works on the continuous case.  

### References

SCHULMAN, John; et al. Proximal Policy Optimization Algorithms. aug. 2017.  
SCHULMAN, John; et al. High-Dimensional Continuous Control Using Generalized Advantage Estimation. sep. 2017  
HEESS, Nicolas; et al. Emergence of Locomotion Behaviours in Rich Environments. jul. 2017  
