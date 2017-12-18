### Proximal Policy Optimization 

The Proximal Policy Optimization works by optimizing a “surrogate” objective function using stochastic gradient ascent with multiple minibatch updates. It works on continuous or discrete action spaces, but this implementation is about the continuous case.  

SCHULMAN John; et al. Proximal Policy Optimization Algorithms. aug. 2017.  

- PPO  
	- Only Fully Connected Layers - Tested on Unity Continuous Catcher
		- [player_PPO_1](../reinforcement/players/player_PPO_1.py)
		- [player_PPO_1A](../reinforcement/players/player_PPO_1A.py)
