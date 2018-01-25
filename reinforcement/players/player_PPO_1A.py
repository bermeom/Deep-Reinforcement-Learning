from players.player_PPO_1 import *
import tensorflow as tf


# PLAYER PPO
class player_PPO_1A( player_PPO_1 ):

    LEARNING_RATE = 3e-4
    UPDATE_SIZE   = 5
    BATCH_SIZE    = 2048
    EPSILON       = 0.2
    GAMMA         = 0.99
    LAM           = 0.95

    ### __INIT__
    def __init__( self ):

        player_PPO_1.__init__( self )

    ### PREPARE NETWORK
    def network( self ):

        # Critic

        Critic = self.brain.addBlock( 'Critic' )

        Critic.addInput( shape = [ None, self.obsv_shape[0] ], name='Observation' )

        Critic.setLayerDefaults( type       = tb.layers.fully,
                                 activation = tb.activs.tanh )

        Critic.addLayer( out_channels = 64, input = 'Observation' )
        Critic.addLayer( out_channels = 64 )
        Critic.addLayer( out_channels = 1, name = 'Value', activation = None )

        # Actor

        Actor = self.brain.addBlock( 'Actor' )

        Actor.addInput( shape = [ None, self.obsv_shape[0] ], name = 'Observation' )

        Actor.setLayerDefaults( type       = tb.layers.fully,
                                activation = tb.activs.tanh )

        Actor.addLayer( out_channels = 64 , input = 'Observation' )
        Actor.addLayer( out_channels = 64,  name = 'Hidden' )

        Actor.addLayer( out_channels = self.num_actions , input = 'Hidden', name = 'mu', activation = None)
        Actor.addLayer( out_channels = self.num_actions , input = 'Hidden', activation = tb.activs.softplus, name = 'sigma' )

        a_mu     = Actor.tensor( 'mu' )
        a_sigma  = Actor.tensor( 'sigma' )
        a_dist   = tf.distributions.Normal( a_mu, a_sigma )
        a_action = a_dist.sample( 1 )

        Actor.addInput( tensor = a_action, name = 'Output')

        # OldActor

        Old = self.brain.addBlock( 'Old' )

        Old.addInput( shape = [ None, self.obsv_shape[0] ], name = 'Observation' )

        Old.setLayerDefaults( type       = tb.layers.fully,
                              activation = tb.activs.tanh )

        Old.addLayer( out_channels = 64 , input = 'Observation' )
        Old.addLayer( out_channels = 64,  name = 'Hidden' )

        Old.addLayer( out_channels = self.num_actions , input = 'Hidden', name = 'mu', activation = None )
        Old.addLayer( out_channels = self.num_actions , input = 'Hidden', activation = tb.activs.softplus, name = 'sigma' )
