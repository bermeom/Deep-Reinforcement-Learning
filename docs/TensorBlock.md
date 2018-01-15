# TensorBlock

Tensorblock is a Google's TensorFlow API, developed by Vitor Guizilini to facilitate the implementation of Machine Learning Algorithms: [BitBucket/TensorBlock](https://bitbucket.org/vguizilini/tensorblock/overview).  

## Overview

In the *reinforcement* folder, *players* contain the the Reinforcement Learning algorithms/agents, that consiste in two files, one for the core algorithm, describing the *agent* operations and training, and one for the *neural network* model used.  
Also in the *reinforcement* folder, *sources* contain the files that are the interface to the environments, like the OpenAi's Gym, Pygame, Mujoco or Unity.  
The *tensorblock* cointains all the functions and classes that facilitates the implementation of the algorithms on the *reinforcement* folder.  
The *execute* file run it all.

## Usage

The algorithms are trained using the command: python execute.py source_*your_source* ... player_*your_player*  
There are some flags that can be used:  
--save *your_saved_model_name*  
--load *your_saved_model_name*  
--run (just run the saved model and do not train)  
And some scripts on the *reinforcement* folder, as well as saved trained models.
