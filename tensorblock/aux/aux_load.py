
import pickle
import numpy as np

### Load Matrix
def load_mat( file ):

    lines = [ line.rstrip('\n') for line in open( file ) ]

    nd = lines[0].split( ' ' )
    mat = np.zeros( ( int(nd[0]) , int(nd[1]) ) )

    for i in range( 2 , len( lines ) ):
        list = lines[i].split( ' ' )
        for j in range( 0 , len( list ) - 1 ):
            mat[i-2,j] = float( list[j] )

    return mat

### Save List
def save_list( file , list ):

    pickle.dump( list , open( file + '.lst' , 'wb' ) )

### Load List
def load_list( file ):

    return pickle.load( open( file + '.lst' , 'rb' ) )

### Load Numpy
def load_numpy( file ):

    return np.load( file + '.npy' )
