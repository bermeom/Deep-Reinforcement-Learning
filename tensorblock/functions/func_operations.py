import tensorflow as tf
import tensorblock as tb

### Copy
def copy( tensors , extras , pars ):

    list = []
    for i in range( len( tensors[0] ) ):
        list.append( tensors[1][i].assign( tensors[0][i] ) )
    return list

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

### Mean Squared ErrorHL
def mean_squared_errorHL( tensors , extras , pars ):

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
def log_sum_mul(tensors, extras, pars):

    axis = len(tb.aux.tf_shape(tensors[0])) - 1

    loglik =  - tensors[1] * tf.log(tensors[0] + 1e-5)

    return tf.reduce_sum(loglik, axis=axis)

### Calculate Cost
def adv_mul(tensors, extras, pars):

    loglikadv = tensors[0] * tensors[1]

    return tf.reduce_mean(loglikadv)

### Get Gradients
def get_grads(tensors, extras, pars):

    return tf.gradients(tensors[0], tensors[1])

### Combine Gradients
def combine_grads(tensors, extras, pars):

    vars = tf.trainable_variables()
    normal_actor_vars = [var for var in vars if 'NormalActor' in var.name]

    return tf.gradients(tensors[0], normal_actor_vars, -tensors[1])

### Assign Gradients on Layers
def assign(tensors, extras, pars):

    TAU = 0.001

    vars = tf.trainable_variables()

    normal_critic_vars = [var for var in vars if 'NormalCritic' in var.name]
    target_critic_vars = [var for var in vars if 'TargetCritic' in var.name]

    normal_actor_vars = [var for var in vars if 'NormalActor' in var.name]
    target_actor_vars = [var for var in vars if 'TargetActor' in var.name]

    update_critic = [target_critic_vars[i].assign(tf.multiply(normal_critic_vars[i], TAU) +
             tf.multiply(target_critic_vars[i], 1. - TAU))
             for i in range(len(target_critic_vars))]

    update_actor = [target_actor_vars[i].assign(tf.multiply(normal_actor_vars[i], TAU) +
             tf.multiply(target_actor_vars[i], 1. - TAU))
             for i in range(len(target_actor_vars))]

    return update_critic, update_actor

### Surrogate Cost
def klcost(tensors, extras, pars):

    a_mu      = tensors[0]
    a_sigma   = tensors[1]
    o_mu      = tensors[2]
    o_sigma   = tensors[3]
    actions   = tensors[4]
    advantage = tensors[5]
    epsilon   = tensors[6][0]

    pi = tf.distributions.Normal( a_mu, a_sigma )
    oldpi = tf.distributions.Normal( o_mu, o_sigma )

    ratio =  pi.prob( actions ) / ( oldpi.prob( actions ) + 1e-6 )
    surr = ratio * advantage
    cost = -tf.reduce_mean( tf.minimum( surr, tf.clip_by_value( ratio, 1.- epsilon, 1. + epsilon ) * advantage ) )

    return cost

### Assign Pi and OldPi
def assignold(tensors, extras, pars):

    vars = tf.trainable_variables()

    pi_vars = [var for var in vars if 'Actor' in var.name]
    oldpi_vars = [var for var in vars if 'Old' in var.name]

    update = [oldpi.assign(pi) for pi, oldpi in zip(pi_vars, oldpi_vars)]

    return update
