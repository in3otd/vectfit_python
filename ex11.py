import numpy
import scipy
import vectfit
import matplotlib.pyplot as plt

j = complex(0.0, 1.0)

fname ='/home/otd/python/vfit/SDR0703-4R7ML.txt'

Y_data = numpy.loadtxt(fname, skiprows=1, delimiter=',')

test_freq = Y_data[:,0]
test_s = 2*j*scipy.pi*test_freq
test_f = Y_data[:,1]+j*Y_data[:,2]

N = 30 #order of approximation

poles = -2*numpy.pi*numpy.logspace(numpy.log10(test_freq[0]), numpy.log10(test_freq[-1]), N)

#b = 2*numpy.pi*numpy.logspace(numpy.log10(test_freq[0]), numpy.log10(test_freq[-1]), N/2)
#poles = numpy.empty(N, dtype=complex)
#poles[0:N:2] = -b/100 + j*b
#poles[1:N:2] = -b/100 - j*b

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
    print 'rmserror = ', numpy.linalg.norm(test_f-ffit)/numpy.sqrt(ffit.size)
    #vectfit.make_plot(test_s, test_f, poles, residues, d, h)
        

omega = scipy.imag(test_s)

plt.figure(1)
plt.semilogy(omega, numpy.absolute(test_f), 'g')
plt.semilogy(omega, numpy.absolute(ffit), 'r')
plt.semilogy(omega, numpy.absolute(test_f-ffit), 'b')
plt.grid(True)
plt.show()
