from players.player import player
from auxiliar.aux_plot import *

import random
from collections import deque

import sys
sys.path.append('..')
import tensorblock as tb
import numpy as np

# PLAYER PPO

class player_PPO_1(player):

    # __INIT__
    def __init__(self):

        player.__init__(self)

        self.experiences = deque()
        self.epsilon = 0.2
        self.gamma = 0.9

    # CHOOSE NEXT ACTION
    def act(self, state):

        return self.calculate(state)

    # CALCULATE NETWORK
    def calculate(self, state):

        action = self.brain.run( 'Actor/Output', [ [ 'Actor/Observation', [state] ] ] )
        return self.create_action( np.reshape( action[0], [self.num_actions] ) )

    # PREPARE NETWORK

    def operations(self):

        # Action Placeholders

        self.brain.addInput( shape = [ None, self.num_actions ], name = 'Actions'  )
        self.brain.addInput( shape = [ None, 1 ] ,               name = 'Advantage')
        self.brain.addInput( shape = [ None ],                   name = 'Epsilon'  )

        # Operations

            # Critic

        self.brain.addOperation( function = tb.ops.mean_squared_errorHL,
                                 input    = [ 'Critic/Value','Advantage' ],
                                 name     = 'CriticCost' )

        self.brain.addOperation( function      = tb.optims.adam,
                                 input         = 'CriticCost',
                                 learning_rate = self.LEARNING_RATE,
                                 name          = 'CriticOptimizer' )
            # Actor

        self.brain.addOperation( function = tb.ops.klcost,
                                 input    = ['Actor/mu',
                                             'Actor/sigma',
                                             'Old/mu',
                                             'Old/sigma',
                                             'Actions',
                                             'Advantage',
                                             'Epsilon'],
                                 name     = 'ActorCost' )

        self.brain.addOperation( function      = tb.optims.adam,
                                 input         = 'ActorCost',
                                 learning_rate = self.LEARNING_RATE,
                                 name          = 'ActorOptimizer' )

            # Assign

        self.brain.addOperation( function = tb.ops.assignold,
                                 input    = [],
                                 name     = 'AssignOld' )

    # TRAIN NETWORK

    def train( self, prev_state, curr_state, actn, rewd, done, episode ):

        # Store New Experience Until Done

        self.experiences.append( (prev_state, curr_state, actn, rewd, done) )

        # Check for Train
        if done:

            # Select Batch

            batch = self.experiences

            # Separate Batch Data

            prev_states = [d[0] for d in batch]
            curr_states = [d[1] for d in batch]
            actions =     [d[2] for d in batch]
            rewards =     [d[3] for d in batch]
            dones =       [d[4] for d in batch]

            # Update Old Pi

            _ = self.brain.run( ['AssignOld'], [] )

            # States Values

            prev_values = np.squeeze( self.brain.run( 'Critic/Value' , [ [ 'Critic/Observation', prev_states  ] ] ) )

            curr_values = np.squeeze( self.brain.run( 'Critic/Value' , [ [ 'Critic/Observation', curr_states  ] ] ) )

            # Calculate Discounted Reward

            running_add = 0
            rewards = rewards + (self.gamma * curr_values) - prev_values
            advantage= np.zeros_like(rewards)
            for t in reversed ( range( 0, len( rewards ) ) ):
                running_add  = running_add * self.gamma + rewards[t]
                advantage[t] = running_add

            advantage = np.expand_dims(advantage,1)

            # Update Actor

            [ self.brain.run( [ 'ActorOptimizer' ], [ [ 'Actor/Observation',  prev_states    ],
                                                      [ 'Old/Observation',    prev_states    ],
                                                      [ 'Actions',            actions        ],
                                                      [ 'Advantage',          advantage      ],
                                                      [ 'Epsilon',            [self.epsilon] ] ] ) for _ in range (self.UPDATE_SIZE) ]

            # Update Critic

            [ self.brain.run( [ 'CriticOptimizer' ], [ ['Critic/Observation', prev_states ],
                                                       ['Advantage',          advantage   ] ] ) for _ in range (self.UPDATE_SIZE) ]

            # Reset

            self.experiences = deque()
