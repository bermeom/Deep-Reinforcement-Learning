from players.player_PPO_1 import *
import tensorflow as tf

##### PLAYER PPO
class player_PPO_1A( player_PPO_1 ):

    LEARNING_RATE = 1e-4
    UPDATE_SIZE   = 10

    ### __INIT__
    def __init__( self ):

        player_PPO_1.__init__( self )

    ### PREPARE NETWORK
    def network( self ):

        # Critic

        Critic = self.brain.addBlock( 'Critic' )

        Critic.addInput( shape = [ None, self.obsv_shape[0] ], name='Observation' )

        Critic.setLayerDefaults( type = tb.layers.fully,
                                 activation = tb.activs.relu )

        Critic.addLayer( out_channels = 64, input = 'Observation' )
        Critic.addLayer( out_channels = 1,  name = 'Value' )

        # Actor

        Actor = self.brain.addBlock( 'Actor' )

        Actor.addInput( shape = [ None, self.obsv_shape[0] ], name = 'Observation' )

        Actor.setLayerDefaults( type = tb.layers.fully,
                                activation = tb.activs.tanh )

        Actor.addLayer( out_channels = 64 , input = 'Observation' )
        Actor.addLayer( out_channels = 64,  name = 'Hidden' )

        Actor.addLayer( out_channels = self.num_actions , input = 'Hidden', name = 'mu' )
        Actor.addLayer( out_channels = self.num_actions , input = 'Hidden', activation = tb.activs.softplus, name = 'sigma' )

        a_mu     = tf.multiply( Actor.tensor( 'mu' ), self.range_actions )
        a_sigma  = Actor.tensor( 'sigma' )
        a_dist   = tf.distributions.Normal( a_mu, a_sigma )
        a_action = tf.squeeze( a_dist.sample( self.num_actions ), [0] )

        Actor.addInput( tensor = a_action, name = 'Output')

        # OldActor

        Old = self.brain.addBlock( 'Old' )

        Old.addInput( shape = [ None, self.obsv_shape[0] ], name = 'Observation' )

        Old.setLayerDefaults( type = tb.layers.fully,
                              activation = tb.activs.tanh )

        Old.addLayer( out_channels = 64 , input = 'Observation' )
        Old.addLayer( out_channels = 64,  name = 'Hidden' )

        Old.addLayer( out_channels = self.num_actions , input = 'Hidden', name = 'mu' )
        Old.addLayer( out_channels = self.num_actions , input = 'Hidden', activation = tb.activs.softplus, name = 'sigma' )

        o_mu     = tf.multiply( Old.tensor( 'mu' ), self.range_actions )
        o_sigma  = Old.tensor( 'sigma' )
        o_dist   = tf.distributions.Normal( o_mu, o_sigma )
        o_action = tf.squeeze( o_dist.sample( self.num_actions ), [0] )

        Old.addInput( tensor = o_action, name = 'Output' )
