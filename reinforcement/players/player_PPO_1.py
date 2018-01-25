from players.player import player
from auxiliar.aux_plot import *

import random
from collections import deque

import sys
sys.path.append('..')
import tensorblock as tb
import numpy as np

from sources.source_unity_exporter import *


# PLAYER PPO
class player_PPO_1(player):

    # __INIT__
    def __init__(self):

        player.__init__(self)

        self.experiences = deque()

    # CHOOSE NEXT ACTION
    def act(self, state):

        return self.calculate(state)

    # CALCULATE NETWORK
    def calculate(self, state):

        action = self.brain.run( 'Actor/Output', [ [ 'Actor/Observation', [state] ] ] )
        return self.create_action( np.reshape ( action,  self.num_actions ) )

    # PREPARE NETWORK
    def operations(self):

        # Placeholders

        self.brain.addInput( shape = [ None, self.num_actions ], name = 'Actions'  )
        self.brain.addInput( shape = [ None, self.num_actions ], name = 'O_mu'  )
        self.brain.addInput( shape = [ None, self.num_actions ], name = 'O_sigma'  )
        self.brain.addInput( shape = [ None, 1 ] ,               name = 'Advantage')
        self.brain.addInput( shape = [ None ],                   name = 'Epsilon'  )

        # Operations

            # Critic

        self.brain.addOperation( function = tb.ops.hlmean_squared_error,
                                 input    = [ 'Critic/Value','Advantage' ],
                                 name     = 'CriticCost' )

        self.brain.addOperation( function      = tb.optims.adam,
                                 input         = 'CriticCost',
                                 learning_rate = self.LEARNING_RATE,
                                 name          = 'CriticOptimizer' )
            # Actor

        self.brain.addOperation( function = tb.ops.ppocost,
                                 input    = ['Actor/mu',
                                             'Actor/sigma',
                                             'O_mu',
                                             'O_sigma',
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

        # To export Unity 3DBall .bytes file execute this when saved trained model is in trained_models folder:
        #export_ugraph (self.brain, "./trained_models/unity_3dball_ppo/", "3dball", "Actor/mu/MatMul")
        #raise SystemExit(0)

        # Store New Experience Until Done

        self.experiences.append( (prev_state, curr_state, actn, rewd, done) )

        # Check for Train

        if ( len(self.experiences) >= self.BATCH_SIZE ):

            batch = self.experiences

            # Separate Batch Data

            prev_states = [d[0] for d in batch]
            curr_states = [d[1] for d in batch]
            actions     = [d[2] for d in batch]
            rewards     = [d[3] for d in batch]
            dones       = [d[4] for d in batch]

            # States Values

            prev_values = np.squeeze( self.brain.run( 'Critic/Value' , [ [ 'Critic/Observation', prev_states  ] ] ) )
            curr_values = np.squeeze( self.brain.run( 'Critic/Value' , [ [ 'Critic/Observation', curr_states  ] ] ) )

            # Calculate Generalized Advantage Estimation

            running_add_y = 0
            running_add_a = 0
            y             = np.zeros_like(rewards)
            advantage     = rewards + (self.GAMMA * curr_values) - prev_values
            for t in reversed ( range( 0, len( advantage ) ) ):
                if dones[t]:
                    curr_values[t] = 0
                    running_add_a  = 0
                running_add_y  = curr_values[t] * self.GAMMA            + rewards   [t]
                running_add_a  = running_add_a  * self.GAMMA * self.LAM + advantage [t]
                y         [t]  = running_add_y
                advantage [t]  = running_add_a
            y         = np.expand_dims( y,         1 )
            advantage = np.expand_dims( advantage, 1 )

            # Update Old Pi

            assign = self.brain.run( ['AssignOld'], [] )

            # Get Old Mu and Sigma

            o_mu, o_sigma = self.brain.run( [ 'Old/mu', 'Old/sigma' ], [ [ 'Old/Observation', prev_states ] ] )

            # Update

            for _ in range (self.UPDATE_SIZE):

                self.brain.run( [ 'ActorOptimizer' ], [ [ 'Actor/Observation',  prev_states    ],
                                                        [ 'O_mu',               o_mu           ],
                                                        [ 'O_sigma',            o_sigma        ],
                                                        [ 'Actions',            actions        ],
                                                        [ 'Advantage',          advantage      ],
                                                        [ 'Epsilon',            [self.EPSILON] ] ] )

                self.brain.run( [ 'CriticOptimizer' ], [ [ 'Critic/Observation', prev_states ],
                                                         [ 'Advantage',          y           ] ] )

            # Reset

            self.experiences = deque()
