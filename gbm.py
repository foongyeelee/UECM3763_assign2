import pylab as p
import numpy as np

# Setup parameters
mu = 0.1; sigma = 0.26; S0 = 39;
n_path = 1000; #number of simulations
n = 1000;#number of partitions

# Create Brownian paths with 1000 paths for 0 < t < 3
t = p.linspace(0,3,n+1)
dB = p.randn(n_path, n+1)/p.sqrt(n/3)
dB[:,0] = 0
B = dB.cumsum(axis = 1)
 
# Calculate stock prices
nu = mu - sigma*sigma/2.0
S = p.zeros_like(B);
S[:,0] = S0
S[:,1:] = S0*p.exp(nu*t[1:]+sigma*B[:,1:])

# Plot 5 realizations of GBM
S_plot = S[0:5]
p.plot(t,S_plot.transpose())
label = 'Time, $t$';p.xlabel(label)
label = 'Stock prices, $RM$';p.ylabel(label)
p.title('5 realizations of Geometric Brownian Motion with $\mu$ =' + str(mu) + 'and $\sigma$ =' +str(sigma))
p.show()

#Calculations
S3 = p.array(S[:,-1])
E_S3 = np.mean(S3);Var_S3 = np.var(S3)
print('E(S3) = ' + str(E_S3) , '\nVar(S3) = ' + str(Var_S3))
mask = S3 > 39
P_S3_more_than_39 = sum(mask)/len(mask)
S3_39 = S3 * mask
E_S3_more_than_39 = sum(S3_39)/sum(mask)
print('P(S3 > 39) = ' + str(P_S3_more_than_39) , '\nE(S3 | S3 >39) = ' + str(E_S3_more_than_39))
