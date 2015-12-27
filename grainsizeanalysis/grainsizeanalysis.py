# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 00:03:08 2015

@author: Lukas Gartmair
"""

import numpy as np
import matplotlib.pyplot as pl
import path
from matplotlib import gridspec

def load_txt(filepath):
    # skip header 
    l = np.loadtxt(filepath,skiprows = 1,unpack = True)
    return l

def summarize(a,b):
    summary = a,b
    return summary

def save_txt(summary):
    filename = 'summary.txt'
    np.savetxt(filename, np.transpose([summary]), header= 'cumulative frequency / %,classes / unit of length', delimiter=',', comments = '')

def get_sum_freq(lens):
    classes, inverse_lens = np.unique(lens, return_inverse=True)
    freq = np.bincount(inverse_lens)
    # number of values
    nov = lens.size
    rel_freq = freq/ nov *100    
    sum_freq = np.cumsum(rel_freq)
    sum_freq_rounded = np.round(sum_freq,decimals =2)
    return sum_freq_rounded, classes
    
def plot(sum_freq,classes):
    #import pdb; pdb.set_trace()
    gs = gridspec.GridSpec(2,1)
    fig = pl.figure()
    ax1 = fig.add_subplot(gs[0,:])
    ax1.scatter(classes,sum_freq,color = 'red')
    ax1.grid()
    ax1.set_xlabel('classes / unit of length')
    ax1.set_ylabel('cumulative frequency / %')
    ax2 =fig.add_subplot(gs[1:])
    ax2.hist(classes)
    ax2.grid() 
    ax2.set_xlabel('classes / unit of length')
    ax2.set_ylabel('counts / a.u.')
    gs.update(wspace=0.5, hspace=0.5)
 
def grain_size_analysis():
    filepath = path.get_path()
    lens = load_txt(filepath)
    sum_freq, classes = get_sum_freq(lens)
    
    plot(sum_freq, classes)
    summary = summarize(sum_freq, classes)
    save_txt(summary)
    
import unittest

class GrainSizeAnalysisTest(unittest.TestCase):
    
    def setUp(self):
        self.test_array1 = np.array([0,1,2,3,2,1,0])
        self.test_array1_res = np.array([28.57,57.14,85.71,100])
        self.test_array2 = np.array([ 30.81,  21.93,  19.84,  17.23,  90.86,  35.51,  34.99,  19.84,
         9.92,  29.77,  50.65,  59.53,  25.59,  30.81,  36.55,  43.86,
        36.03,  24.54,  23.5 ,  26.11,  51.7 ,  36.55,   7.83,  30.29,
        36.55,  56.92,  42.82,  42.82,  50.13,  29.77,  40.21,  65.8 ,
        36.55,  37.08,  29.77,  14.62,  25.59,   8.36,  16.19,  56.92,
        33.42,  27.15,  16.71,  39.69])
        self.test_array2_res = np.array([   2.27,    4.55,    6.82,    9.09,   11.36,   13.64,   15.91,
         20.45,   22.73,   25.  ,   27.27,   31.82,   34.09,   36.36,
         43.18,   45.45,   50.  ,   52.27,   54.55,   56.82,   59.09,
         68.18,   70.45,   72.73,   75.  ,   79.55,   81.82,   84.09,
         86.36,   88.64,   93.18,   95.45,   97.73,  100.  ])
            
    def test_grain_size_analysis(self):
        np.testing.assert_array_equal(get_sum_freq(self.test_array1)[0],self.test_array1_res) 
        np.testing.assert_array_almost_equal(get_sum_freq(self.test_array2)[0],self.test_array2_res, decimal = 2) 
    
def main():
    grain_size_analysis()
    unittest.main()
    
if __name__ == '__main__':
    main()

    
