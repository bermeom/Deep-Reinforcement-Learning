### Q-Learning

- Deep Q Learning  

	- [player_dql_bayesian_2](../reinforcement/players/player_dql_bayesian_2.py)
	- [player_dql_bayesian_2A](../reinforcement/players/player_dql_bayesian_2A.py)  

This implementation features Convolutional Neural Networks.  
It has been tested on the PyGame Catch Environment.  
See the results on [DQL 2A Statistics](https://github.com/NiloFreitas/Deep-Reinforcement-Learning/tree/master/statistics/dql%20bayesian)  
Use [script0](../reinforcement/script0.sh) to run a trained model.  

### Overview

Q-learning is an Off-Policy Temporal Difference Value-Based algorithm that learns an action-value function to find the optimal policy.  
It has an Experience Replay to sample experiences from different policies to calculate the TD Erros.  
The action-value function is updated using MSE of the TD Errors.  
The most common implicity policy is the e-greedy but this implementation has an Bayesian approach using Dropout.

### References

WATKINS, Christopher. Learning from Delayed Rewards. Cambridge, UK: King’s College, 1989.  
MNIH, Volodymyr; et al.Playing Atari with Deep Reinforcement Learning. 2013  
MNIH, Volodymyr; et al. Human-level control through deep reinforcement learning. Nature, Vol 518, fev. 2015.  
