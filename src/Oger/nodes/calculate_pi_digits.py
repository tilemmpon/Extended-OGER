# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 17:30:35 2017

@author: Tilemachos Bontzorlos
"""

import numpy as np

PI_DIGITS_FILENAME = 'pi_10_mil_digits.txt'

def calculate_n_digits_pi(n):
    """
    Calculates and returns a list of integers of the n first digits of pi.
    
    Parameters
    ----------
    
    n : int
        number of digits to return
    """
    # algorithm is a partially adapted version of one in:
    # http://www.codecodex.com/wiki/Calculate_digits_of_pi
    
    # check if n is integer
    if not isinstance(n, (int, long)):
        print "n is not integer. Function requires integer n."
        return None
    pi_digits_string = ''
    scale = 10000
    maxarr = max(2800, n*4)  # my magic number - TO CHECK
    arrinit = 2000  
    carry = 0  
    arr = [arrinit] * (maxarr + 1)  
    for i in xrange(maxarr, 1, -14):  
        total = 0  
        for j in xrange(i, 0, -1):  
            total = (total * j) + (scale * arr[j])  
            arr[j] = total % ((j * 2) - 1)  
            total = total / ((j * 2) - 1)  
        pi_digits_string += ("%04d" % (carry + (total / scale)))
        carry = total % scale
    return [int(i) for i in pi_digits_string[0:n]]
    
def load_n_digits_of_pi(n, filename=PI_DIGITS_FILENAME):
    """
    Load and returns a list of integers of the n first digits of pi from a pi
    value stored in a file.
    
    Parameters
    ----------
    
    n : int
        number of digits to return
    
    filename : string
        filename and path of the saved pi value
    """
    pi_digits_string = ''
    with open(filename) as f:
        for line in f:
            pi_digits_string += line.strip().replace('\n', '')
            if len(pi_digits_string) >= n:
                break
    
    return [int(i) for i in pi_digits_string[0:n]]
    
    
def get_pi_digits_in_array(shape, pi_file=PI_DIGITS_FILENAME, use_file=True):
    """
    Returns the pi digits in a 2d-numpy array of the specified shape.
    
    Parameters
    ----------
    
    shape : list of two ints
        shape of the array to return
        
    pi_file: string
        the location of the pi digits file

    use_file : boolean
        True if user want to use precalculated pi from file. False to calculate
        pi. (for many digits it takes too long to calculate.)
    """
    if len(shape) != 2:
        print "Shape must be 2 dimensional."
        return None
    if shape[0] <= 0 or shape[1] <= 0 or \
                not isinstance(shape[0], (int, long)) or \
                not isinstance(shape[1], (int, long)):
        print "shape digits must be positive integers"
        return None
        
    if use_file:
        digits = np.asarray(load_n_digits_of_pi(shape[0]*shape[1], pi_file))
    else:
        digits = np.asarray(calculate_n_digits_pi(shape[0]*shape[1]))
    return np.reshape(digits, shape)
    
#test = np.where(get_pi_digits_in_array([10000, 1000]) < 5, -1, 1)