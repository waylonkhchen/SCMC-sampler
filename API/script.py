#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 12:03:47 2019

@author: waylonchen
"""

import sys
from SCMC_module import SCMC
import os


def main(argv):
    input_path, output_path = argv[0], argv[1]
    n_results = int(argv[2])
    
    output_dir = "/".join(output_path.split("/")[:-1])
    try:
        os.makedirs(output_dir, exist_ok=True)
    except:
        pass
    result = SCMC(input_path, n_results,track_correctness=True )
    
    result.write_output(output_path)
    output_path = os.path.abspath(output_path)
    print('The output file is saved at: \n {}'.format(output_path))
    

    
if __name__ == "__main__":
    main(sys.argv[1:])