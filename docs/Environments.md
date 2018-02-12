## Enviroments

The suppported Environments are in the [sources](../reinforcement/sources) folder.  
The [source.py](../reinforcement/sources/source.py) contains the parent class.  
Currently, there are interfaces with [OpenAI's Gym](https://github.com/NiloFreitas/Deep-Reinforcement-Learning/blob/master/reinforcement/sources/source_gym.py), [PyGame](https://github.com/NiloFreitas/Deep-Reinforcement-Learning/blob/master/reinforcement/sources/source_pygame.py) and [Unity ML](https://github.com/NiloFreitas/Deep-Reinforcement-Learning/blob/master/reinforcement/sources/source_unity.py).  

The Environments in the folder are:

Raw Inputs (1 dimension):  

	- Pygame
		- Chase
	- Gym
		- CartPole  
		- Continuous Mountain Car
		- Mujoco
			- Pendulum
			- HalfCheetah
			- Hopper
			- Reacher
	- Unity Machine Learning
		- 3D Ball
		- Continuous Catcher 


Raw Image Inputs (2 dimensions):  

	- Pygame
		- Catch
	- Gym
		- Breakout
		- Pong  

### Notes

1) It is easy to add new environments.

2) To use a Unity Env the build files of the environment must be on folder reinforcement/sources/unity/  
You can build it yourself following [Unity ML GitHub](https://github.com/Unity-Technologies/ml-agents) or download some in my [Google Drive](https://drive.google.com/drive/folders/13_uD0QtYc8fzWaVAz5auVNzk9WQUOqi4?usp=sharing).

3) To export a Unity Env .bytes file to run the trained model on Unity:  
Execute this function on the algorithm when a saved trained model of the env is in _trained_models_ folder:     
	<sub>from sources.source_unity_exporter import *  
	export_ugraph (self.brain, "./trained_models/" + trainedmodel, envname, nnoutput)  
	raise SystemExit(0)  
	#Example with PPO and 3DBall: trainedmodel = "unity_3dball_ppo/",
	                              envname = "3dball",
				      nnoutput =  "Actor/Mu/MatMul" </sub>  
