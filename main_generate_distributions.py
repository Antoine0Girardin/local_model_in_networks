# -*- coding: utf-8 -*-
"""
Code to generate distribution in network. See the main part to set the parameters
"""

import numpy as np
import os
            
from module.quantum_network_module.quantum_network import quantum_network_fct

from utils.measurements import get_measurement, TCfamily

from utils.states import get_state
#######################################################

def save_to_file(d, path):
    f= open(path, 'w')
    np.savetxt(f, np.reshape(d, [1,np.size(d)]))
    f.close()

    
if __name__=="__main__":
    
    ## choose location to save the distribution
    folder="target_distributions/test/"
    
    ## create the folder 
    model_dir = os.path.dirname(folder)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    #example for the triangle with 3 sources and 3 parties
    n_sources=3
    n_parties=3
    
    ## choose the state distributed by the sources (see utils.states.py for some default states). 
    ## states should be given as density matrices.
    ## v is the visibility of the state mixed with white noise
    ## the format is a Tuple of the density matrix and a list of the dimensions of each subsystem: (density_matrix, [dimension_subsystem1, dimension_subsystem2, ...])
    sources = [get_state(0, v=1) for i in range(n_sources)]
    
    ## choose the measurement performed by the parties (see utils.measurements.py for some default measurements)
    measurements = [TCfamily(np.sqrt(0.785)) for i in range(n_parties)]
    # measurements = [get_measurement(0) for i in range(n_parties)]
    
    ## choose the permutation of the measurements and state (see documentation for examples)
    order_sources=[3,4,5,0,1,2]
    
    ## compute the target distribution
    target_distribution = quantum_network_fct(sources, measurements, order_hs_sources=order_sources)

    ## save the target distribution in a file
    # name_distribution = "rgb4_umax.txt"
    # path = folder+name_distribution
    # save_to_file(target_distribution, path)

