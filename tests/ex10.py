# vector fitting example

import os.path
import sys

# include parent directory in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.path.pardir))

import numpy
import scipy
import vectfit
import matplotlib.pyplot as plt


fname ='./SDR0703-4R7ML.txt'

data = numpy.loadtxt(fname, skiprows=1, delimiter=',')

freq = data[:,0]
s = 2j*scipy.pi*freq
func = data[:,1]+1j*data[:,2]

# order of approximation
N = 30 
# initial poles
poles = -2*numpy.pi*numpy.logspace(numpy.log10(freq[0]), numpy.log10(freq[-1]), N)

for i in range(15):
    poles = vectfit.vectfit_step(func, s, poles)
    residues, d, h = vectfit.calculate_residues(func, s, poles)
    func_fit = vectfit.model(s, poles, residues, d, h)
    print ('rmserror(%i) = '% i), numpy.linalg.norm(func-func_fit)/numpy.sqrt(func_fit.size)
        

omega = scipy.imag(s)

plt.figure(1)
plt.semilogy(omega, numpy.absolute(func), 'r', label='input data')
plt.semilogy(omega, numpy.absolute(func_fit), 'g', label='data fit')
plt.semilogy(omega, numpy.absolute(func-func_fit), 'b', label='difference')
plt.xlabel('omega, rad/s')
plt.ylabel('impedance, ohm')
plt.grid(True)
plt.legend()
plt.show()
