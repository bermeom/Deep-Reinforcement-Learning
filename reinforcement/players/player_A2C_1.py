from players.player import player
from auxiliar.aux_plot import *
import tensorflow as tf

import random
from collections import deque

import sys
sys.path.append('..')
import tensorblock as tb
import numpy as np

# PLAYER A2C

class player_A2C_1(player):

    # __INIT__
    def __init__(self):

        player.__init__(self)

        self.experiences = deque()
        self.num_stored_obsv = self.NUM_FRAMES

    # CHOOSE NEXT ACTION
    def act(self, state):

        return self.calculate( state )

    # CALCULATE NETWORK
    def calculate(self, state):

        output = np.squeeze(self.brain.run('Output', [['Observation', [state]]]))
        action = np.random.choice(np.arange(len(output)), p=output)
        return self.create_action(action)

    # PREPARE NETWORK

    def operations(self):

        # Action Placeholders

        self.brain.addInput(shape=[None, self.num_actions], name='Actions')
        self.brain.addInput(shape=[None], name='Advantage')

        # Operations

            # Critic

        self.brain.addOperation(function=tb.ops.mean_squared_error,
                                input=['Value','Advantage'], name='CriticLoss')

        self.brain.addOperation(function=tb.optims.adam, input='CriticLoss',
                                learning_rate=self.LEARNING_RATE*10, name='CriticOptimizer')

            # Actor

        self.brain.addOperation(function=tb.ops.log_sum_mul,
                                input=['Output', 'Actions'], name='Readout')

        self.brain.addOperation(function=tb.ops.adv_mul,
                                input=['Readout', 'Advantage'], name='PolicyLoss')

        self.brain.addOperation(function=tb.optims.adam, input='PolicyLoss',
                                learning_rate=self.LEARNING_RATE, name='ActorOptimizer', summary = 'Summary' , writer = 'Writer')

        # TensorBoard
        self.brain.addSummaryScalar( input = 'PolicyLoss' )
        self.brain.addWriter(name = 'Writer' , dir = './' )
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
            actions = [d[2] for d in batch]
            rewards = [d[3] for d in batch]
            dones = [d[4] for d in batch]

            # States Value
            prev_values = np.squeeze(self.brain.run( 'Value' , [ [ 'Observation' , prev_states ] ] ) )
            next_values = np.squeeze(self.brain.run( 'Value' , [ [ 'Observation' , curr_states ] ] ) )

            # Calculate Expected Reward and TD Error

            expected_rewards = []
            td_errors = []
            for i in range( len(rewards) ):
                if dones[i]:
                    expected_rewards.append( rewards[i] )
                    td_errors.append( expected_rewards[i] - prev_values[i] )
                else:
                    expected_rewards.append( rewards[i] + self.REWARD_DISCOUNT * next_values[i] )
                    td_errors.append( expected_rewards[i] - prev_values[i] )

            # Optimize Neural Network

            _, = self.brain.run(['CriticOptimizer'], [ ['Observation', prev_states     ],
                                                       ['Advantage',   expected_rewards] ] )

            _,c,summary = self.brain.run(['ActorOptimizer','PolicyLoss','Summary'], [ ['Observation', prev_states   ],
                                                                                      ['Actions',     actions       ],
                                                                                      ['Advantage',   td_errors     ] ] )

            # TensorBoard
            self.brain.write( summary = summary, iter = episode )
