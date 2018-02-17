### Deep Deterministic Policy Gradients

- DDPG  
	- [player_DDPG_1](../reinforcement/players/player_DDPG_1.py)
	- [player_DDPG_1A](../reinforcement/players/player_DDPG_1A.py)

It has been tested on the Gym Mujoco Environments.  
See the results on [DDPG 1A Statistics](https://github.com/NiloFreitas/Deep-Reinforcement-Learning/tree/master/statistics/DDPG%201A)  
Use [script3](../reinforcement/script3.sh) to run a trained model.  

### Overview

The Deep Deterministic Policy Gradients is an Off-Policy Temporal Difference Actor-Critic algorithm.  
It has an Experience Replay to sample experiences from different policies to calculate the TD Erros.  
It uses Target Networks, updated softly, to stabilize the action-value functions estimations.  
It operates on continuous action spaces and uses the Uhlenbeck and Ornstein process to improve exploration.  

### References

LILLICRAP, Timothy P.; et al. Continuous control with deep reinforcement learning. ICLR, fev. 2016.  
SILVER, David; et al. Deterministic policy gradient algorithms. ICML, 2014.  


