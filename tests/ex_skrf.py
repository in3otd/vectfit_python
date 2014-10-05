# vector fitting example using scikit-rf to read a S-parameters file

import os.path
import sys

# include parent directory in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.path.pardir))

import numpy
import scipy
import skrf as rf
import vectfit
import matplotlib.pyplot as plt


fname = './BN43-2402_4t_26AWG.s1p'

dut = rf.Network(fname)
freq = dut.f.T
s = 2j*numpy.pi*freq
func = dut.y.squeeze()

# order of approximation
N = 4 
# initial (real) poles
poles = -2*numpy.pi*numpy.logspace(numpy.log10(freq[0]), numpy.log10(freq[-1]), N)

for i in range(5):
    poles = vectfit.vectfit_step(func, s, poles)
    residues, d, h = vectfit.calculate_residues(func, s, poles)
    func_fit = vectfit.model(s, poles, residues, d, h)
    print ('rmserror(%i) = '% i), numpy.linalg.norm(func-func_fit)/numpy.sqrt(func_fit.size)
        
omega = scipy.imag(s)

plt.figure(1)
plt.loglog(omega, numpy.absolute(func), 'r', label='input data')
plt.loglog(omega, numpy.absolute(func_fit), 'g', label='data fit')
plt.loglog(omega, numpy.absolute(func-func_fit), 'b', label='difference')
plt.xlabel('omega, rad/s')
plt.ylabel('admittance, S')
plt.grid(True)
plt.legend()
plt.show()
