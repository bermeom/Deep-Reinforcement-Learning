### REINFORCE

- REINFORCE  

	- [player_reinforce_1](../reinforcement/players/player_reinforce_1.py) and [player_reinforce_1A](../reinforcement/players/player_reinforce_1A.py)  

	- [player_reinforce_2](../reinforcement/players/player_reinforce_2.py) and [player_reinforce_2A](../reinforcement/players/player_reinforce_2A.py)  

	- [player_reinforce_rnn_1](../reinforcement/players/player_reinforce_rnn_1.py) and [player_reinforce_rnn_1A](../reinforcement/players/player_reinforce_rnn_1A.py)  

	- [player_reinforce_rnn](../reinforcement/players/player_reinforce_rnn_2.py) and [player_reinforce_rnn_2A](../reinforcement/players/player_reinforce_rnn_2A.py)  

*1A* features only MLP;  *2A* features Conv. Nets.;  *rnn* features LSTM Rec. Nets.  
It has been tested on the PyGame Catch and Gym Cartpole Environments.  
See the results on [REINFORCE 2A Statistics](https://github.com/NiloFreitas/Deep-Reinforcement-Learning/tree/master/statistics/reinforce%202A), [REINFORCE RNN 1A Statistics](https://github.com/NiloFreitas/Deep-Reinforcement-Learning/tree/master/statistics/reinforce%20rnn%201A) and [REINFORCE RNN 2A Statistics](https://github.com/NiloFreitas/Deep-Reinforcement-Learning/tree/master/statistics/reinforce%20rnn%202A).  
Use [script1](../reinforcement/script1.sh) to run a trained model.  


### Overview

REINFORCE is an On-Policy Monte Carlo Policy-Based algorithm that learns an stochastic policy using Policy Gradients.  
It uses the discounted Monte Carlo returns and the log likelihood ratio to optimize the policy directly.  

### References

WILLIAMS, Ronald. Simple statistical gradient-following algorithms for connectionist reinforcement learning. 1992.  
