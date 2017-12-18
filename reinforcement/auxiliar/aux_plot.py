import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

### PLOT STATE
def plot_state( state ):

    n = state.shape[2]

    fig = plt.figure()
    for i in range( n ):
        plt.subplot( 2 , n , i + 1 )
        plt.imshow( state[:,:,i] , cmap = 'gray' )
    plt.show()

### PLOT STATES
def plot_states( prev_state , curr_state ):

    n = prev_state.shape[2]

    fig = plt.figure()
    for i in range( n ):
        plt.subplot( 2 , n , i + 1 )
        plt.imshow( curr_state[:,:,i] , cmap = 'gray' )
        plt.subplot( 2 , n , i + n + 1 )
        plt.imshow( prev_state[:,:,i] , cmap = 'gray' )
    plt.show()

### SAVE STATISTICS
def plot_episode_stats(episode_lengths,episode_rewards):

    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.xlim(0, len(episode_lengths))
    plt.title("Episode Length over Time")
    plt.savefig('./auxiliar/LenghtsPlot.png')
    np.savetxt('./auxiliar/EpisodeLengths.txt', episode_lengths, fmt='%.3f', newline='\n')
    plt.close()

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.xlim(0, len(episode_rewards))
    plt.title("Episode Rewards")
    plt.savefig('./auxiliar/RewardsPlot.png')
    np.savetxt('./auxiliar/EpisodeRewards.txt', episode_rewards, fmt='%.3f', newline='\n')
    plt.close()

    return None
