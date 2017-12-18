from players.player import player
from auxiliar.aux_plot import *
import tensorflow as tf

import random
from collections import deque

import sys
sys.path.append('..')
import tensorblock as tb
import numpy as np

# PLAYER DDPG

class player_DDPG_1(player):

    # __INIT__
    def __init__(self):

        player.__init__(self)

        self.experiences = deque()
        self.num_stored_obsv = self.NUM_FRAMES
        self.noise_state = 0
        self.dt = 0.01

        # Copy initial weights (need to find better way)

        vars = tf.trainable_variables()

        normal_critic_vars = [var for var in vars if 'NormalCritic' in var.name]
        target_critic_vars = [var for var in vars if 'TargetCritic' in var.name]
        normal_actor_vars = [var for var in vars if 'NormalActor' in var.name]
        target_actor_vars = [var for var in vars if 'TargetActor' in var.name]

        update_critic = [target_critic_vars[i].assign(normal_critic_vars[i])
                            for i in range(len(target_critic_vars))]

        update_actor = [target_actor_vars[i].assign(normal_actor_vars[i])
                            for i in range(len(target_actor_vars))]

    ## ORNSTEIN-UHLENBECK PROCESS
    def OU( self, mu, theta, sigma):

        x = self.noise_state
        dx =  self.dt * theta * (mu - self.noise_state) + sigma * np.random.randn(self.num_actions) *  np.sqrt(self.dt)

        self.noise_state = x + dx

        return self.noise_state

    ### CHOOSE NEXT ACTION
    def act( self , state):

        return self.calculate( state )

    # CALCULATE NETWORK
    def calculate(self, state):

        action = self.brain.run( 'NormalActor/Output', [ [ 'NormalActor/Observation', [state] ] ] )
        noise =  self.OU( mu = 0, theta = 0.15 , sigma = 0.2 )

        action = action[0] + noise

        return self.create_action( np.reshape( action, [self.num_actions] ) )

    # PREPARE NETWORK
    def operations(self):

        # Placeholders

        self.brain.addInput( shape = [ None,1 ], name = 'Advantage', dtype = tf.float32 )

        self.brain.addInput( shape = [ None,self.num_actions ], name = 'ActionGrads', dtype = tf.float32 )

        # Operations

        self.brain.addOperation( function = tb.ops.assign, input = [],
                                 name = 'Assign' )
            # Critic

        self.brain.addOperation( function = tb.ops.get_grads, input=['NormalCritic/Value','NormalCritic/Actions'],
                                 summary = 'Summary',
                                 writer = 'Writer',
                                 name = 'GetGrads' )

        self.brain.addOperation( function = tb.ops.mean_squared_errorHL, # this mse gets the L2 reg on fullyhl layers
                                 input = [ 'NormalCritic/Value','Advantage' ],
                                 name = 'CriticCost' )

        self.brain.addOperation( function = tb.optims.adam, input = 'CriticCost',
                                 learning_rate = self.LEARNING_RATE*10,
                                 name = 'CriticOptimizer' )
            # Actor

        self.brain.addOperation( function=tb.ops.combine_grads, input = [ 'NormalActor/Output','ActionGrads' ],
                                 name = 'CombineGrads' )

        self.brain.addOperation( function=tb.optims.adam_apply, input = [ 'CombineGrads' ],
                                 learning_rate = self.LEARNING_RATE,
                                 name = 'ActorOptimizer' )

        # TensorBoard

        self.brain.addSummaryScalar( input = 'CriticCost' )
        self.brain.addSummaryHistogram( input = 'GetGrads' )
        self.brain.addSummaryHistogram( input = 'NormalActor/Output' )
        self.brain.addWriter( name = 'Writer' , dir = './' )
        self.brain.addSummary( name = 'Summary' )
        self.brain.initialize()

    # TRAIN NETWORK
    def train(self, prev_state, curr_state, actn, rewd, done, episode):

        # Store New Experience

        self.experiences.append( ( prev_state , curr_state , actn , rewd , done ) )
        if len( self.experiences ) > self.EXPERIENCES_LEN: self.experiences.popleft()

        # Check for Train

        if len( self.experiences ) > self.STEPS_BEFORE_TRAIN and self.BATCH_SIZE > 0:

            # Select Random Batch

            batch = random.sample( self.experiences , self.BATCH_SIZE )

            # Separate Batch Data

            prev_states = [d[0] for d in batch]
            curr_states = [d[1] for d in batch]
            actions =     [d[2] for d in batch]
            rewards =     [d[3] for d in batch]
            dones =       [d[4] for d in batch]

            # States Values

            target_actns = self.brain.run( 'TargetActor/Output', [ [ 'TargetActor/Observation', curr_states ] ] )

            next_values = self.brain.run( 'TargetCritic/Value' , [ [ 'TargetCritic/Observation', curr_states  ],
                                                                   [ 'TargetCritic/Actions',     target_actns ] ] )

            # Calculate Expected Reward

            expected_rewards = []
            for i in range( self.BATCH_SIZE ):
                if dones[i]:
                     expected_rewards.append( rewards[i] )
                else:
                    expected_rewards.append( rewards[i] + self.REWARD_DISCOUNT * next_values[i] )

            expected_rewards = np.reshape( expected_rewards, [ self.BATCH_SIZE, 1 ] )

            # Optimize Critic

            _ = self.brain.run( ['CriticOptimizer'], [ [ 'NormalCritic/Observation', prev_states      ],
                                                       [ 'NormalCritic/Actions',     actions          ],
                                                       [ 'Advantage',                expected_rewards ] ] )
            # Get New Actions

            new_a = self.brain.run( 'NormalActor/Output', [ ['NormalActor/Observation', prev_states] ] )

            # Get Critic Grads w.r.t. New Actions

            grads = self.brain.run( ['GetGrads'], [ [ 'NormalCritic/Observation', prev_states],
                                                    [ 'NormalCritic/Actions',     new_a      ] ] )

            # Run Summary (need to find better way)

            _, s = self.brain.run( ['GetGrads','Summary'], [ [ 'NormalCritic/Observation', prev_states      ],
                                                             [ 'NormalCritic/Actions',     new_a            ],
                                                             [ 'Advantage',                expected_rewards ],
                                                             [ 'NormalActor/Observation',  prev_states      ] ] )
            # Optimize Actor

            _ = self.brain.run( ['ActorOptimizer'], [ [ 'NormalActor/Observation', prev_states ],
                                                      [ 'ActionGrads',             grads[0]    ] ] )

            # Copy weights to Target Networks

            _,_ = self.brain.run( 'Assign' , [] )

            # TensorBoard

            self.brain.write( summary = s, iter = episode )
