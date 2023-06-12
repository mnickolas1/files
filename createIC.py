import sys    # system specific calls
import numpy as np    # scientific computing package
import h5py    # hdf5 format
import matplotlib.pyplot as plt
from scipy.integrate import  quad
from scipy.optimize import root

def rho(x):
    M = 10**10
    a = 7670
    fg = 0.16
    
    return 4*np.pi*x**2 * fg*M/(2*np.pi)*a/x*1/(x+a)**3

def cdf(x):
    integral, _ = quad(rho, 0, x)
    normalization, _ = quad(rho, 0, 4)
    return integral / normalization

def sample_from_pdf(num_samples):
    samples = []
    for _ in range(num_samples):
        u = np.random.uniform(0, 1)
        # Find the x value that corresponds to the given u
        x = root(lambda x: cdf(x) - u, x0=0.1, method='hybr').x[0]
        samples.append(x)
        print(len(samples))
    return samples

def spherical_to_cartesian(r,phi,theta):
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z

def density(x):
    M = 10**10
    a = 7670
    fg = 0.16
    
    return fg*M/(2*np.pi)*a/x*1/(x+a)**3

def dm(x):
    
    return 4*np.pi*x**2 * density(x)

"""
def dphi_dr(x):
    G = 4.301871e-03
    M = 10**10
    a = 7670
    Mbh = 10**6

    if(x > 0):
      return G*Mbh/(x)**2
    else:
      print("r is zero")
      return 0
  
def hydro_eqn(x, density, dphi_dr):
 
    return -density(x) * dphi_dr(x)
"""   
   
simulation_directory = str(sys.argv[1])
print("BHs: creating ICs in directory " + simulation_directory)

""" initial condition parameters """
FilePath = simulation_directory + '/IC.hdf5'

# double precision: np.float64, for single use np.float32
FloatType = np.float64  
IntType = np.int32

#blackhole ics
PosBh = np.zeros(3, dtype=FloatType)
PosBh[0] = 4
PosBh[1] = 4
PosBh[2] = 4

MassBh = 2.24*10**6
hsml   = 0.5
IsBh   = 0

#generate random samples from the distribution (i.e. HQ rho)
N=10**5
R = sample_from_pdf(N)
print("Sample done")
#convert the radial distances to Cartesian coordinates
x_values = []
y_values = []
z_values = []

for r in R:
    #generate random values for theta and phi
    phi = np.arccos(1-2*np.random.uniform(0, 1)) # Cosine of Angle between -1 and 1 (polar angle)
    theta = 2*np.pi * np.random.uniform(0, 1)    # Angle between 0 and 2*pi (azimuthal angle)

    #convert from spherical to Cartesian coordinates
    x, y, z = spherical_to_cartesian(r,phi,theta)

    #append the coordinates to the respective lists
    x_values.append(x)
    y_values.append(y)
    z_values.append(z)

R        = np.array(R, dtype=FloatType)
x_values = np.array(x_values, dtype=FloatType)
y_values = np.array(y_values, dtype=FloatType)
z_values = np.array(z_values, dtype=FloatType)
print("Conversion done")

#sort arrays
indices = np.argsort(R)        
r = R[indices]
x = x_values[indices]
y = y_values[indices]
z = z_values[indices]

Boxsize = FloatType(8)
Pos = np.zeros([N,3],dtype=FloatType)

Pos[:,0] = x
Pos[:,1] = y
Pos[:,2] = z

Pos[:,0] += PosBh[0]
Pos[:,1] += PosBh[1]
Pos[:,2] += PosBh[2]

#same mass for sph particles
M, _ = quad(dm, 0, 4)
m = M/N
Mass = np.full(N, m, dtype=FloatType)
print(m)

Pressure = np.zeros(N, dtype=FloatType)

#pressure for ideal gas
for i in range(N):
    #Pressure[i], _ = quad(hydro_eqn, 0.1, r[i], args=(density,dphi_dr))
    Pressure[i] = 10**3* density(r[i]) #(T~10**5K) 

gamma = FloatType(5/3) 
gamma_minus1 = gamma - FloatType(1.0)

#utherm for ideal gas
Density=np.zeros(N,dtype=FloatType)
for i in range(N):
    Density[i] = density(r[i])

Uthermal = Pressure / gamma_minus1 / Density

#checks
bins = 100
count, bin_edges = np.histogram(r, bins=bins)
bin_edges = np.delete(bin_edges, -1)
bin_width = bin_edges[1]-bin_edges[0]
count = np.array(count, dtype=FloatType)
count/= (N * bin_width)

rho_analytic = np.zeros(bins)
for i in range(bins):
    rho_analytic[i] = rho(bin_edges[i])
norm, _ = quad(rho, 0, 4)
rho_analytic /= norm

plt.scatter(bin_edges, count)
plt.plot(bin_edges,rho_analytic)
plt.show()

from mpl_toolkits.mplot3d import Axes3D

# Plot the particles on a sphere
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Pos[:,0], Pos[:,1], Pos[:,2])

# Set plot limits and labels
#ax.set_xlim(-1.5, 1.5)
#ax.set_ylim(-1.5, 1.5)
#ax.set_zlim(-1.5, 1.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set aspect ratio to be equal for all axes
ax.set_box_aspect([1, 1, 1])

# Show the plot
plt.show()


""" write *.hdf5 file; minimum number of fields required by Arepo """
IC = h5py.File(FilePath, 'w')    # open/create file

## create hdf5 groups
header = IC.create_group("Header")    # create header group
part0 = IC.create_group("PartType0")    # create particle group for gas cells
part5 = IC.create_group("PartType5")  

## write header entries
NumPart = np.array([N,0,0,0,0,1], dtype=IntType)
header.attrs.create("NumPart_ThisFile", NumPart)
header.attrs.create("NumPart_Total", NumPart)
header.attrs.create("NumPart_Total_HighWord", np.zeros(6, dtype=IntType) )
header.attrs.create("MassTable", np.zeros(6, dtype=IntType) )
header.attrs.create("Time", 0.0)
header.attrs.create("Redshift", 0.0)
header.attrs.create("BoxSize", Boxsize)
header.attrs.create("NumFilesPerSnapshot", 1)
header.attrs.create("Omega0", 0.0)
header.attrs.create("OmegaB", 0.0)
header.attrs.create("OmegaLambda", 0.0)
header.attrs.create("HubbleParam", 1.0)
header.attrs.create("Flag_Sfr", 0)
header.attrs.create("Flag_Cooling", 0)
header.attrs.create("Flag_StellarAge", 0)
header.attrs.create("Flag_Metals", 0)
header.attrs.create("Flag_Feedback", 0)
if Pos.dtype == np.float64:
    header.attrs.create("Flag_DoublePrecision", 1)
else:
    header.attrs.create("Flag_DoublePrecision", 0)

## write cell data
part0.create_dataset("ParticleIDs", data=np.arange(1, N+1))
part0.create_dataset("Coordinates", data=Pos)
part0.create_dataset("Masses", data=Mass)
#part0.create_dataset("Velocities", data=Velocity)
part0.create_dataset("InternalEnergy", data=Uthermal)
part5.create_dataset("ParticleIDs" , data = np.arange(N+1,N+2))
part5.create_dataset("Coordinates", data=PosBh)
part5.create_dataset("Masses" , data = MassBh)
#part5.create_dataset("Velocities", data=VelBh[0:1])
part5.create_dataset("BlackholeHsml" , data = hsml)
part5.create_dataset("IsBh", data = int(IsBh))


## close file
IC.close()

""" normal exit """
sys.exit(0) 
