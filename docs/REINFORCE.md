### REINFORCE

The REINFORCE algorithm is an monte carlo stochastic policy gradients algorithm that uses a likelihood ratio to estimate the policy gradient.  

WILLIAMS, Ronald. Simple statistical gradient-following algorithms for connectionist reinforcement learning. 1992.  

- REINFORCE Algorithm  
	- Only Fully Connected Layers - Untested
		- [player_reinforce_1](../reinforcement/players/player_reinforce_1.py)
		- [player_reinforce_1A](../reinforcement/players/player_reinforce_1A.py)
	- Using Convolutional Layers - Tested on Pygame Catch
		- [player_reinforce_2](../reinforcement/players/player_reinforce_2.py)
		- [player_reinforce_2A](../reinforcement/players/player_reinforce_2A.py)
	- Using Recurrent and Fully Connected Layers - Tested on Gym Cartpole
		- [player_reinforce_rnn_1](../reinforcement/players/player_reinforce_rnn_1.py)
		- [player_reinforce_rnn_1A](../reinforcement/players/player_reinforce_rnn_1A.py)
	- Using Recurrent and Convolutional Layers - Tested on Pygame Catch
		- [player_reinforce_rnn](../reinforcement/players/player_reinforce_rnn_2.py)
		- [player_reinforce_rnn_2A](../reinforcement/players/player_reinforce_rnn_2A.py)

