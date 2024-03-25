#!/usr/bin/env python
# coding: utf-8

# In[166]:


from scipy.optimize import fsolve
import numpy as np

# Define the function to solve tan(w) = 2λw / (λ^2 w^2 - 1)
def equation_to_solve(w, lam):
    return np.tan(w) - 2 * lam * w / (lam**2 * w**2 - 1)

# A reasonable starting guess for w_n solutions could be around the zeros of the tan function, which are at (n+0.5)*pi
# We will use lambda = 1 (for example) and find the first few solutions
lambda_value = 1
solutions = []
number_of_solutions = 51

# We'll start the initial guesses just above the zero points of the tangent function to avoid division by zero
initial_guesses = [(n + 0.5) * np.pi for n in range(number_of_solutions)]

# Using fsolve to find the solutions
for guess in initial_guesses:
    wn_solution = fsolve(equation_to_solve, guess, args=(lambda_value))
    # We only append the solution if it is not already in the list (to avoid duplicates due to numerical precision)
    if not any(np.isclose(wn_solution, sol) for sol in solutions):
        solutions.append(wn_solution[0])

number_of_solutions = len(solutions)

print(number_of_solutions)

print


# In[167]:


# Given the solutions w, we now want to calculate the corresponding theta_n and b_n(x)
# Let's define the functions to calculate theta_n and b_n(x) for lambda_value = 1
# Based on the paper's formulas:
# theta_n^1D = 2 * lambda / (lambda^2 * w_n^2 + 1)
# b_n^1D(x) = A_n * sin(w_n * x) + A_n * lambda * cos(w_n * x)

# Function to calculate theta_n
def theta_n_1D(w_n, lam):
    return np.sqrt(2 * lam / ((lam**2)*(w_n**2) + 1)) 

# Function to calculate b_n(x)
def b_n_1D(w_n, lam, x): 
    A_n =np.sqrt((np.sin(w_n * x))**2 + (lam*w_n*np.cos(w_n * x))**2)
    return  (np.sin(w_n * x) + lam * w_n * np.cos(w_n * x)) / A_n

# Calculate theta_n and b_n for a range of x values
# Choose an arbitrary range for x, for example, 0 to 2*pi with 100 points

x_values = np.linspace(0, 1, number_of_solutions)

# Calculate theta_n for each solution w
theta_values_1D = np.array([theta_n_1D(w, lambda_value) for w in solutions])

# Calculate b_n for each combination of w_n and x_values
b_values_1D = np.array([[b_n_1D(w, lambda_value, x) for x in x_values] for w in solutions])

# The result will be a list of theta values and a 2D-array of b_n values across the x range
#theta_values, b_values.shape
theta_values_1D.shape



# In[168]:


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import numpy.fft
from numpy.fft import fft, ifft
from scipy.fft import fft, ifft

def circ_cov_sample(c):
    
    N=len(c)
    d=sp.fft.ifft(c).real*N

    xi=np.random.normal(0,1,N)

    Z=sp.fft.fft((d**0.5)*xi)/np.sqrt(N)
    
    X=Z.real
    Y=Z.imag
    
    return X, Y, d, xi

def circ_embed(c):
    
    N = len(c)
    c_new = np.zeros(2*(N-1))
    c_new[0:N] = c
    
    for i in range(N-2):
        c_new[N+i]=c[N-2-i]
        
    return c_new


# In[169]:


# Plotting the sample paths

T=1
l=1

n=number_of_solutions-1
dt=T/(number_of_solutions)

c = np.zeros(n+1)

for i in range(n+1):
    c[i] = np.exp(-i*dt/l)

c_new = circ_embed(c)

X,Y,d,xi = circ_cov_sample(c_new)

sample1 = X[0:(n+1)]
sample2 = Y[0:(n+1)]

plt.plot(x_values,sample1)


#1D
import matplotlib.pyplot as plt
#assume the expectation of Z is 0
mean = 0

xi_n = xi[0:number_of_solutions]
n_list = np.arange(0,number_of_solutions)
Z = mean + np.dot((xi_n * theta_values_1D).T,b_values_1D)

plt.plot(x_values,Z)


# In[171]:


M=1000
error = 0

T=1
l=1

n=number_of_solutions-1
dt=T/(number_of_solutions)

c = np.zeros(n+1)

for i in range(n+1):
    c[i] = np.exp(-i*dt/l)

c_new = circ_embed(c)

for m in range(M):

    X,Y,d,xi = circ_cov_sample(c_new)

    sample1 = X[0:(n+1)]
    sample2 = Y[0:(n+1)]

    #1D
    import matplotlib.pyplot as plt
    #assume the expectation of Z is 0
    mean = 0

    xi_n = xi[0:number_of_solutions]
    n_list = np.arange(0,number_of_solutions)
    Z = mean + np.dot((xi_n * theta_values_1D).T,b_values_1D)
    
    error += dt*sum(abs(sample1-Z))/M

print(error)


# In[ ]:




