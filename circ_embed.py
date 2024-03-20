import numpy as np
import matplotlib.pyplot as plt
import numpy.fft
from numpy.fft import fft, ifft

def circ_cov_sample(c):
    
    N=len(c)
    d=np.fft.ifft(c)*N
    xi=np.random.normal(0,1,2*N).reshape((N,2))@np.array([1,1j])

    Z=fft((d**0.5)*xi)/np.sqrt(N)
    X=Z.real
    Y=Z.imag
    
    return X, Y, d

def circ_embed(c):
    
    N = len(c)
    c_new = np.zeros(2*(N-1))
    c_new[0:N] = c
    
    for i in range(N-2):
        c_new[N+i]=c[N-1-i]
        
    return c_new

# Plotting the sample paths

T=1
l=1
dt=0.001
n=int(T/dt)

c = np.zeros(n+1)

for i in range(n+1):
    c[i] = np.exp(-i*dt/l)

c_new = circ_embed(c)

X,Y,d = circ_cov_sample(c_new)

sample1 = X[0:(n+1)]
sample2 = Y[0:(n+1)]

plt.plot(sample1)


# Plotting the eigenvalue decay for various l

T=1
l_list=[0.01,0.1,1,10]
dt=0.001
n=int(T/dt)
print(n)

for l in l_list:
    
    c = np.zeros(n+1)
    
    for i in range(n+1):
        c[i] = np.exp(-i*dt/l)
    
    c_new = circ_embed(c)
    
    X,Y,d = circ_cov_sample(c_new)
    
    sample1 = X[0:(n+1)]
    sample2 = Y[0:(n+1)]
    
    e = sorted(d,reverse=True)
    
    plt.loglog(e)