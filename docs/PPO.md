### Proximal Policy Optimization

- PPO  
	- [player_PPO_1](../reinforcement/players/player_PPO_1.py)
	- [player_PPO_1A](../reinforcement/players/player_PPO_1A.py)

It has been tested on the Unity Machine Learning Environments.  
See the results on [PPO 1A Statistics](https://github.com/NiloFreitas/Deep-Reinforcement-Learning/tree/master/statistics/PPO%201A).  
Use [script4](../reinforcement/script4.sh) to run a trained model, after putting the Unity 3DBall [build files](https://drive.google.com/drive/folders/13_uD0QtYc8fzWaVAz5auVNzk9WQUOqi4?usp=sharing) on *reinforcement/sources/unity/*  
This made [video](https://youtu.be/0cBAjqQ8nw4) shows it running on Unity with the .bytes file created by this implementation.  

### Overview

The Proximal Policy Optimization is an On-Policy with Importance Sampling Actor Critic algorithm that alternates between colecting batches of samples and optimizing the policy by a number of epochs with a “surrogate” objective function.  
This implementation does not share parameters between the Actor and the Critic. It does not have parallel actors, so the sampled batch comes from a single agent.  
It works on continuous or discrete action space environments.  

### References

SCHULMAN, John; et al. Proximal Policy Optimization Algorithms. aug. 2017.  
SCHULMAN, John; et al. High-Dimensional Continuous Control Using Generalized Advantage Estimation. sep. 2017  
HEESS, Nicolas; et al. Emergence of Locomotion Behaviours in Rich Environments. jul. 2017  
SCHULMAN, John; et al. Trust region policy optimization. ICML. 2015.  
