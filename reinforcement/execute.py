import argparse
import importlib

parser = argparse.ArgumentParser( description = 'Input Arguments' )
parser.add_argument( 'inputs' , nargs = 2 )
parser.add_argument( '--load' , dest = 'load' , default = None )
parser.add_argument( '--save' , dest = 'save' , default = [ None ] , nargs = '*' )
parser.add_argument( '--epis' , dest = 'epis' , default = 1e9 )
parser.add_argument( '--run'  , dest = 'run'  , action = 'store_true' )
args = parser.parse_args()

source_string , player_string = args.inputs
source_module = importlib.import_module( 'sources.' + source_string )
player_module = importlib.import_module( 'players.' + player_string )

source = getattr( source_module , source_string )()
player = getattr( player_module , player_string )()
player.parse_args( args )

obsv = source.start()
state = player.start( source , obsv )

episode , n_episodes = 0 , int( args.epis )
while episode < n_episodes:

    actn = player.act( state )                                 # Choose Next Action
    obsv , rewd , done = source.move( actn )                   # Run Next Action On Source
    state = player.learn( state , obsv , actn , rewd , done, episode )  # Learn From This Action

    source.verbose( episode , rewd , done ) # Source Text Output
    player.verbose( episode , rewd , done ) # Player Text Output

    if done: # If Game Is Over
        obsv = source.start()
        state = player.restart( obsv )
        episode += 1
