import numpy
import scipy
import skrf as rf
import vectfit
import matplotlib.pyplot as plt

j = complex(0.0, 1.0)

#fname = '../gpib/cuwire49t_2.s1p'
#fname = '../gpib/TV_choke.s2p'
fname = '../gpib/BN43-2402_4t_26AWG.s1p'

dut = rf.Network(fname)
test_s = 2*j*numpy.pi*dut.f
test_f = dut.y.squeeze()

N = 4 #order of approximation
poles = -2*numpy.pi*numpy.logspace(numpy.log10(dut.f.T[0]), numpy.log10(dut.f.T[-1]), N)
#poles = [-6.1788e+03+0.0000e+00j, -4.7030e+07+0.0000e+00j, -3.1118e+06+6.5321e+06j, -3.1118e+06-6.5321e+06j]

#poles = [-3.5874e+07, -3.1760e+06, -2.0305e+06, -1.2169e+05]
#residues = [1.9911e+04, 2.6723e+03, -7.4412e+03, 4.3436e+04]
#d = 7.3219e-04
#h = 1.2230e-12

for i in range(5):
    poles = vectfit.vectfit_step(test_f, test_s, poles)
    residues, d, h = vectfit.calculate_residues(test_f, test_s, poles)
    #print poles
    #print residues
    #print d
    #print h
    ffit = sum(c/(test_s - a) for c, a in zip(residues, poles)) + d + test_s*h
    print 'rmserror = ', numpy.linalg.norm(dut.y[:,0,0]-ffit)/numpy.sqrt(ffit.size)
    #vectfit.make_plot(test_s, test_f, poles, residues, d, h)
        

omega = scipy.imag(test_s)

plt.figure(1)
plt.loglog(omega, numpy.absolute(dut.y[:,0,0]), 'g')
plt.loglog(omega, numpy.absolute(ffit), 'r')
plt.loglog(omega, numpy.absolute(dut.y[:,0,0]-ffit), 'b')
plt.grid(True)
plt.show()
