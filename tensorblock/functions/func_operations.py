import tensorflow as tf
import tensorblock as tb

### Copy
def copy( tensors , extras , pars ):

    list = []
    for i in range( len( tensors[0] ) ):
        list.append( tensors[1][i].assign( tensors[0][i] ) )
    return list

### Assign
def assign( tensors , extras , pars ):

    vars = tf.trainable_variables()
    variables0 = [ var for var in vars if str(tensors[0]) in var.name ]
    variables1 = [ var for var in vars if str(tensors[1]) in var.name ]

    return [ v0.assign(v1) for v0, v1 in zip(variables0, variables1) ]

### Assign Soft
def assign_soft( tensors , extras , pars ):

    vars = tf.trainable_variables()
    variables0 = [ var for var in vars if str(tensors[0]) in var.name ]
    variables1 = [ var for var in vars if str(tensors[1]) in var.name ]

    return [ v0.assign( v1 * tensors[2] + v0 * (1. - tensors[2] )) for v0, v1 in zip(variables0, variables1) ]

### Mean SoftMax Cross Entropy Logit
def mean_soft_cross_logit( tensors , extras , pars ):

    return tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(
                logits = tensors[0] , labels = tensors[1] ) )

### Weighted Mean SoftMax Cross Entropy Logit
def weighted_mean_soft_cross_logit( tensors , extras , pars ):

    return tf.reduce_mean( tf.multiply(
            tf.nn.softmax_cross_entropy_with_logits( tensors[0] , tensors[1] ) , tensors[2] ) )

### Mean Squared Error
def mean_squared_error( tensors , extras , pars ):

    return tf.reduce_mean( tf.square( tensors[0] - tensors[1] ) )

### Mean Squared Error
def hlmean_squared_error( tensors , extras , pars ):

    return tf.losses.mean_squared_error ( tensors[0] , tensors[1]  )

### Masked Mean Squared Error
def masked_mean_squared_error( tensors , extras , pars ):

    shape = tensors[0].get_shape().as_list()
    label , max_seqlen = tensors[0] , shape[1]

    if len( tensors ) == 3:
        output , seqlen = tensors[1] , tensors[2]
    else:
        output , seqlen = extras[0] , tensors[1]

    mask = tf.sequence_mask( seqlen , max_seqlen , dtype = tf.float32 )

    cost = tf.square( label - output )

    if len( shape ) == 3:
        cost = tf.reduce_sum( cost , reduction_indices = 2 )

    cost = tf.reduce_sum( cost * mask , reduction_indices = 1 )
    cost /= tf.reduce_sum( mask , reduction_indices = 1 )

    return tf.reduce_mean( cost )

### Mean Equal Argmax
def mean_equal_argmax( tensors , extras , pars ):

    correct = tf.equal( tf.argmax( tensors[0] , 1 , name = 'ArgMax_1' ) ,
                        tf.argmax( tensors[1] , 1 , name = 'ArgMax_2' ) )

    return tf.reduce_mean( tf.cast( correct , tf.float32 ) )

### Mean Cast
def mean_cast( tensors , extras , pars ):

    return tf.reduce_mean( tf.cast( tensors[0] , tf.float32 ) )


### Sum Mul
def sum_mul( tensors , extras , pars ):

    axis = len( tb.aux.tf_shape( tensors[0] ) ) - 1

    return tf.reduce_sum( tf.multiply(
                tensors[0] , tensors[1] ) , axis = axis )

### Mean Variational
def mean_variational( tensors , extras , pars ):

    z_mu , z_sig = extras
    z_mu2 , z_sig2 = tf.square( z_mu ) , tf.square( z_sig )

    rec_loss = tf.reduce_sum( tf.square( tensors[0] - tensors[1] ) )
    kl_div = - 0.5 * tf.reduce_sum( 1.0 + tf.log( z_sig2 + 1e-10 ) - z_mu2 - z_sig2 , 1 )

    return tf.reduce_mean( rec_loss + kl_div )


### Policy Gradients Cost
def pgcost(tensors, extras, pars):

    axis = len(tb.aux.tf_shape(tensors[0])) - 1
    loglik = tf.reduce_sum( tensors[1] * tensors[0], axis=axis)

    return tf.reduce_mean( tf.multiply( -tf.log(loglik + 1e-8), tensors[2] ) )

### Get Gradients
def get_grads(tensors, extras, pars):

    return tf.gradients(tensors[0], tensors[1])

### Combine Gradients (DDPG)
def combine_grads(tensors, extras, pars):

    vars = tf.trainable_variables()
    normal_actor_vars = [var for var in vars if 'NormalActor' in var.name]

    return tf.gradients(tensors[0], normal_actor_vars, -tensors[1])

### Policy Gradients Cost (PPO)
def ppocost(tensors, extras, pars):

    a_mu      = tensors[0]
    a_sigma   = tensors[1]
    o_mu      = tensors[2]
    o_sigma   = tensors[3]
    actions   = tensors[4]
    advantage = tensors[5]
    epsilon   = tensors[6]

    pi    = tf.distributions.Normal( a_mu, a_sigma )
    oldpi = tf.distributions.Normal( o_mu, o_sigma )

    ratio = pi.prob( actions ) / ( oldpi.prob( actions ) + 1e-6 )
    surr  = ratio * advantage
    cost  = -tf.reduce_mean( tf.minimum( surr, tf.clip_by_value( ratio, 1.- epsilon, 1. + epsilon) * advantage ) )

    entropy = tf.reduce_mean (pi.entropy())

    return cost - 0.001 * entropy
