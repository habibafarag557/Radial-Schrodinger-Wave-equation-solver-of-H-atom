import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import genlaguerre , factorial
from scipy.integrate import trapezoid

#Radial equation for U(r) is d**2U/dr**2-2m/h_bar**2 (h_bar**2 (l(l+1))/2mr**2 -e**2/r-E)U(r)=0
#we're going to plot the U(r) vs r graphs for 3 methods to solve the equation to get the wavefunction U(r) where U(r)=rR(r)


#we fisrt start with the exact solution, fix n=2, l=1 for the 4 curves se the atomic units to be 1 for a0, reduced mass and other 
# parameters and for numerical solution we're going to use the energy for n=2


def enter_quantum_numbers():
    print("\/" * 15)
    print("Hydrogen atom Radial Solver")
    print("\/" * 15)

    while True:
        try:
            n = int(input("Principal quantum number  n:"))
            if n < 1:
                raise ValueError
        except ValueError:
            print("n must be a positive integer")
            continue

        try:
            l = int(input(f"orbital quantum number  l:"))
            if not (0 <= l <= n - 1):
                raise ValueError
        except ValueError:
            print(f"l must satisfy 0 ≤ l ≤ {n-1}.")
            continue


        return n, l
    
n,l= enter_quantum_numbers()    
a0=1
E=-1/(2*n**2)
lamda= 1/(n*a0)


#linespace and grid (r should be from zero to infinity but we need to avoid singularities also we need to make maximum radius adjustable with entered n according
#to the expectation value of r and hence make the grid and linspace adjustable to the maximum radius 
r_min=1e-7
r_max = max(20, int(2.5 * n**2 * a0) + 10)
N = max(5000, r_max * 500)  
N = int(N)
r=np.linspace(r_min,r_max,N)
dr=r[1]-r[0]

def R_exact(r) :
    rho = 2.0 * r / (n * a0)   
    norm = np.sqrt(
        (2.0 / (n * a0)) ** 3
        * factorial(n - l - 1)
        / (2.0 * n * factorial(n + l))
    )
    L = genlaguerre(n - l - 1, 2 * l + 1)
    return  norm * np.exp(-rho / 2.0) * rho ** l * L(rho)
R_ex=R_exact(r)

#now we are going to define the asymptotic solution for the boundaries only which is U(r)=r**l+1 * e**-lam*r
#in this solution we don't include the intermediate region correlation term to show how the asymmptotic can be different from the exact in the middle regions for n,l

def U_bound (r):
     return r**(l+1) * np.exp(-lamda*r)


U_b=U_bound(r)
normalization = np.sqrt(trapezoid(U_b**2, r))
U_b/=normalization
R_b=U_b/r


#now for the numerical solution and since we don't have first derivative in our equation we can use numerov method, 
# for numerov method  d**2U/dr**2=2m/h_bar**2 (h_bar**2 (l(l+1))/2mr**2 -e**2/r-E)U(r)
#y′′(x)=f(x)y(x), f(x) in our case = 2m/h_bar**2 (h_bar**2 (l(l+1))/2mr**2 -e**2/r-E)

def f(r):
    return 2*(E + 1/r) - l*(l+1)/r**2


U_numerov = np.zeros(N)

# Initial conditions
U_numerov[0] = r[0]**(l+1)
U_numerov[1] = r[1]**(l+1)

for i in range(1, N-1):
    f0 = f(r[i-1])
    f1 = f(r[i])
    f2 = f(r[i+1])

    U_numerov[i+1] = (
        (2*(1 - 5*dr**2*f1/12)*U_numerov[i]
        - (1 + dr**2*f0/12)*U_numerov[i-1])
        / (1 + dr**2*f2/12)
    )
   
#we can print out a data sheet for all the intirvals or R and the corresponding U(r) for it to the numerical method 
datasheet=pd.DataFrame({"r":r, "U(r)":U_numerov})   
datasheet.to_excel(r"D:\Python projects\numerical data for radial equation of H atom.xlsx",index=False) 
     
normalization_num = np.sqrt(trapezoid(U_numerov**2, r))
U_numerov /= normalization_num
R_numerov=U_numerov/r

#we can also print Bohr energy value corresponding to the entered n

print(f"Bohr Energy Level:E_{n} ={E} Hartee")

fig, axes = plt.subplots(2, 2, figsize=(13, 11))
fig.suptitle(f"Hydrogen Radial Wavefunction  R_{{n={n}, l={l}}}(r)", fontsize=14)

axes[0, 0].plot(r, R_b, ':', color='red', linewidth=2)
axes[0, 0].set_title(r"Asymptotic solution: $r^{\,l}\,e^{-\lambda r}$")
axes[0, 0].set_xlim(0, r_max)

axes[0, 1].plot(r, R_numerov, '--', color='green', linewidth=2)
axes[0, 1].set_title(f"Numerov numerical solution")
axes[0, 1].set_xlim(0, r_max)

axes[1, 0].plot(r, R_ex, color='blue', linewidth=2)
axes[1, 0].set_title("Exact Solution")
axes[1, 0].set_xlim(0, r_max)

axes[1, 1].plot(r, R_b, ':', color='red', label="Asymptotic", linewidth=2)
axes[1, 1].plot(r, R_numerov, '--', color='green', label="Numerov", linewidth=2)
axes[1, 1].plot(r, R_ex, color='blue', label="Exact", linewidth=2)
axes[1, 1].set_title("Comparison between solutions")


plt.tight_layout()





plt.show()