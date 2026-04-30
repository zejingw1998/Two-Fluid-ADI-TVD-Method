#Two-Fluid Hydrodynamics Project
# This script studies a simple 1D two-fluid Euler model.
# The main final method is a TVD Rusanov solver with an
# New ADI method in multi-fluids.



#Imports and global settings
import torch as torch 
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time 
from pathlib import Path


# Use GPU if it is available. Otherwise use CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
torch.set_default_dtype(dtype)

# Folder for saving figures
fig_dir = Path(__file__).resolve().parent / "AST5110_TVD_figures"
fig_dir.mkdir(exist_ok=True)

def save_figure(fig, filename):
    path = fig_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {path}")

#We asume that, in this problem we have $U =(\rho,m,e)$ $e$ is the energy. So define the function first.
#Single fluid Euler.
#U_t +F(U)_x = 0
# For the internal-energy formulation used here:
# F(U) = (rho*u, m*u + p, e*u)

#Pro,otove to conservative.

#Parameter

gamma = 5.0 /3.0  
eps = 1e-12 # Do not want have 0 division or blows up.

#For one fluid alpha, we use U_alpha = (rho_alpha, m_alpha, e_alpha)
#where m_alpha = rho_alpha * u_alpha
#and e_alpha is the internal energy density

#This is the internal energy formulation. 

#And e = p/(gamma-1)

#This means we imput the primitive variables rhom u , p we can get out put state vector U=(rho,rhou,e)

#Define the function switch the primitive variables to state vector。

def prim_to_state(rho, u, p, gamma=gamma):
    m = rho * u
    e = p / (gamma - 1.0)
    U = torch.stack([rho, m, e], dim=0)
    return U

#Since we know that m= rho * u, thus we have u = m/prho
#The total energy is E = p/(gamma-1) + 1/2 *rho *u^2
#And  p = (gamma-1)(E-1/2 rho *u^2)
#From conservative variables to primitive variables
#In this part we get the U=(rho,rhou,e) to (rho,rhou,e)

#Define the function switch the state vector to primitive。
def cons_to_prim(U, gamma=gamma):
    rho = torch.clamp(U[0], min=eps) #Will no have negative or zero density 
    m   = U[1]
    e   = torch.clamp(U[2], min=eps)

    u = m / rho
    p = (gamma - 1.0) * e
    return rho, u, p

# Debug
#Just want to verify the primitive to state and backward is true, Just check this

rho = torch.tensor([1.0, 0.125], device=device, dtype=dtype)
u   = torch.tensor([0.0, 0.0], device=device, dtype=dtype)
p   = torch.tensor([1.0, 0.1], device=device, dtype=dtype)
U = prim_to_state(rho, u, p, gamma=gamma)
print("U =", U)

rho2, u2, p2 = cons_to_prim(U, gamma=gamma)
print("rho2 =", rho2)
print("u2 =", u2)
print("p2 =", p2)

#Conclusion
#rho2 closed rho, u2 closed u, and p2 closed to p, this is ok.
#
def make_two_fluid_state(rho_i, u_i, p_i, rho_n, u_n, p_n, gamma=gamma):
    U_i = prim_to_state(rho_i, u_i, p_i, gamma=gamma)
    U_n = prim_to_state(rho_n, u_n, p_n, gamma=gamma)

    U = torch.cat([U_i, U_n], dim=0)
    return U
#

def split_two_fluid_state(U):
    U_i = U[0:3]
    U_n = U[3:6]
    return U_i, U_n

#
def state_to_prim_two_fluid(U, gamma=gamma):
    U_i, U_n = split_two_fluid_state(U)
    rho_i, u_i, p_i = cons_to_prim(U_i, gamma=gamma)
    rho_n, u_n, p_n = cons_to_prim(U_n, gamma=gamma)
    return rho_i, u_i, p_i, rho_n, u_n, p_n


#Simplest two-fluid baseline：

#The simple two fluid 1D equation, without heating term

#Continuity (rho_a)t + (rho_a u_a)_x = 0 

#Momentum (m_a)_t +(m_au_a +p_a)_x = rho_abv__ab(u_a-u_b) #The right hand side is the source

#Energy (e_a)_t + (e_au_a)= -p_a(u_a)_x 

#For one fluid alpha, we use U_alpha = (rho_alpha, m_alpha, e_alpha)
#where m_alpha = rho_alpha * u_alpha
#Consider the fluid for continuity, momentum and energy.
#And we have 
#F_rho =m_a
#F_m = m_au_a +p_a
#F_e = e_au_a



#We consider a simplified version of the prim to state function.
#And I will use this moreover.

def stat_one_fluid(U,gamma=gamma): 
    rho= torch.clamp(U[0],min = eps)
    m   = U[1]
    e   = torch.clamp(U[2],min=eps)

    u = m/rho 
    p = (gamma - 1.0) * e
    return rho, u, p


#Define the one fluid system flux.
#We consider the Euler flux equation
#U_t +F(U)_x = 0
#Where F(U)=(rho*u, rho *u^2 +p, u(E+p)), 

def flux_one_fluid(U, gamma=gamma):
    rho, u, p = stat_one_fluid(U, gamma=gamma)
    m = U[1]
    e = U[2]

    F_rho = m  #  The flux associated with the conservation law for density is m
    F_m   = m * u + p # rho * u *u = rho*u^2
    F_e   = e * u

    F = torch.stack([F_rho, F_m, F_e], dim=0)
    return F

#Define the derivatives, the difference method.

def ddx1(f, dx):
    dfdx = torch.zeros_like(f)
    dfdx[1:-1] = (f[2:] - f[:-2]) / (2.0 * dx) #centered difference
    dfdx[0]    = (f[1] - f[0]) / dx #  forward difference，
    dfdx[-1]   = (f[-1] - f[-2]) / dx # backward difference
    return dfdx

#First we assume 

#Continuity (rho_a)t + (rho_a u_a)_x = 0 

#Momentum (m_a)_t +(m_au_a +p_a)_x = 0

#Energy (e_a)_t + (e_au_a)= -p_a(u_a)_x 


#We consider the right hand, U_t = RHS(U)
def rhs_one_fluid(U, dx, gamma=gamma):
    rho, u, p = stat_one_fluid(U, gamma=gamma)
    m = U[1]
    e = U[2]

    F = flux_one_fluid(U, gamma=gamma)

    rhs_rho = -ddx1(F[0], dx)
    rhs_m   = -ddx1(F[1], dx)
    rhs_e   = -ddx1(F[2], dx) - p * ddx1(u, dx) #pressure-work term

    RHS = torch.stack([rhs_rho, rhs_m, rhs_e], dim=0) 
    return RHS

#Apply the source
#S_a =  rho_av__ab(u_a-u_b)
#S_b =  rho_bv__ba(u_a-u_b)


#momentum exchange source
def momentum_source_pair(U_i, U_n, nu_in, gamma=gamma):
    rho_i, u_i, p_i = stat_one_fluid(U_i, gamma=gamma)
    rho_n, u_n, p_n = stat_one_fluid(U_n, gamma=gamma)

    # alpha = rho_i * nu_in = rho_n * nu_ni  (pointwise)
    alpha = rho_i * nu_in

    S_i = alpha * (u_n - u_i)   # ion momentum source
    S_n = -S_i                  # exact opposite, total momentum conserved

    return S_i, S_n
##The total RHS for two-fluid system
def rhs_two_fluid(U_i, U_n, dx, nu_in, gamma=gamma):
    RHS_i = rhs_one_fluid(U_i, dx, gamma=gamma)
    RHS_n = rhs_one_fluid(U_n, dx, gamma=gamma)

    S_i, S_n = momentum_source_pair(U_i, U_n, nu_in, gamma=gamma)

    RHS_i[1] = RHS_i[1] + S_i
    RHS_n[1] = RHS_n[1] + S_n

    return RHS_i, RHS_n


#Explicit Euler

def euler_step_two_fluid(U_i, U_n, dx, dt, nu_in, gamma=gamma):
    RHS_i, RHS_n = rhs_two_fluid(U_i, U_n, dx, nu_in, gamma=gamma)

    U_i_new = U_i + dt * RHS_i
    U_n_new = U_n + dt * RHS_n

    U_i_new[0] = torch.clamp(U_i_new[0], min=eps)
    U_i_new[2] = torch.clamp(U_i_new[2], min=eps)

    U_n_new[0] = torch.clamp(U_n_new[0], min=eps)
    U_n_new[2] = torch.clamp(U_n_new[2], min=eps)

    return U_i_new, U_n_new

def compute_dt_one_fluid(U, dx, gamma=gamma, cfl=0.2): #With CFl
    rho, u, p = stat_one_fluid(U, gamma=gamma)
    c = torch.sqrt(gamma * p / rho)
    max_speed = torch.max(torch.abs(u) + c)
    dt = cfl * dx / max_speed #CFL 
    return dt #The CFL will be different per fluid

def compute_dt_two_fluid(U_i, U_n, dx, gamma=gamma, cfl=0.2): #two-fluid global time step
    dt_i = compute_dt_one_fluid(U_i, dx, gamma=gamma, cfl=cfl)
    dt_n = compute_dt_one_fluid(U_n, dx, gamma=gamma, cfl=cfl)
    return torch.minimum(dt_i, dt_n)
# ============================================================
# Two-fluid RHS and explicit Euler baseline
# ============================================================
# This is the first simple two-fluid time-stepping method.
#
# Each fluid has state
#     U = (rho, m, e),
# where m = rho*u and e = p/(gamma-1).
#
# The hydrodynamic part is computed by rhs_one_fluid.
# Then I add the ion-neutral momentum exchange source.
#
# This baseline is useful for testing the collision source, but it is
# not the final shock-capturing method. Later I use conservative
# variables, Rusanov flux, TVD reconstruction, and ADI-like splitting.
# ============================================================




#Drift relaxation test
#Try to do simplest ion neutral drift relaxation test
N = 200
x = torch.linspace(-1.0, 1.0, N)
dx = x[1] - x[0]

#Initial data for ion and neutral
rho_i = 0.2 * torch.ones_like(x)
u_i   = 1.0 * torch.ones_like(x)
p_i   = 0.1 * torch.ones_like(x)

rho_n = 0.8 * torch.ones_like(x)
u_n   = 0.0 * torch.ones_like(x)
p_n   = 0.1 * torch.ones_like(x)

#Switch primitive variables to state variables
U_i = torch.stack([rho_i, rho_i * u_i, p_i / (gamma - 1.0)], dim=0)
U_n = torch.stack([rho_n, rho_n * u_n, p_n / (gamma - 1.0)], dim=0)


#collision frequency
nu_in = 1.0
nu_ni = (rho_i[0] * nu_in / rho_n[0]).item()
print("nu_ni =", nu_ni)


#Time
t = 0.0
t_end = 0.2

t_hist = []
drift_hist = []
ui_hist = []
un_hist = []

while t < t_end:
    rho_i_now, u_i_now, p_i_now = stat_one_fluid(U_i)#
    rho_n_now, u_n_now, p_n_now = stat_one_fluid(U_n)#

    t_hist.append(t)
    drift_hist.append(torch.mean(torch.abs(u_i_now - u_n_now)).item())
    ui_hist.append(torch.mean(u_i_now).item())
    un_hist.append(torch.mean(u_n_now).item())
    #global time step
    dt = compute_dt_two_fluid(U_i, U_n, dx, gamma=gamma, cfl=0.2).item()

    if t + dt > t_end:
        dt = t_end - t
    if dt <= 0:
        print("dt <= 0")
        break
        #One Explicit Euler
    U_i, U_n = euler_step_two_fluid(U_i, U_n, dx, dt, nu_in, gamma=gamma)
    t += dt #The Time

rho_i, u_i, p_i = stat_one_fluid(U_i)
rho_n, u_n, p_n = stat_one_fluid(U_n)

print("mean u_i =", torch.mean(u_i).item())
print("mean u_n =", torch.mean(u_n).item())
print("mean drift =", torch.mean(torch.abs(u_i - u_n)).item())



# Two fluid shock tube
# ============================================================
# Two-fluid Sod shock tube initial condition
# ============================================================
# Here I build a two-fluid version of the Sod shock tube.
#
# First I define the total density and total pressure on the left
# and right side. Then I split them into ion and neutral parts.
#
# The total left/right states are:
#   left:  rho_tot = 0.125, p_tot = 0.125 / gamma
#   right: rho_tot = 1.0,   p_tot = 1.0 / gamma
#
# ion_frac_L and ion_frac_R control how much of the total density
# belongs to the ion fluid on the left and right side.
#
# press_frac_i controls how much of the total pressure belongs
# to the ion fluid.
#
# I also allow a neutral velocity drift on the left side. This is used
# to test how the ion-neutral collision source reduces velocity drift.
# ============================================================
#We assume the initial value left and right
def initial_condition_two_fluid_sod(x, x0=0.0, gamma=gamma,
                                    ion_frac_L=0.5, ion_frac_R=0.5,
                                    press_frac_i=0.5,
                                    drift_n_L=0.2, drift_n_R=0.0):
    left_mask = x < x0 ## Points with x < x0 are on the left side of the discontinuity.
    # Total density of the shock tube.
    # Left side has low density, right side has high density.
    rho_tot = torch.where(left_mask,
                          torch.full_like(x, 0.125),
                          torch.full_like(x, 1.0))
    # Total pressure of the shock tube.
    # This is a reversed Sod setup: pressure is larger on the right side.
    p_tot = torch.where(left_mask,
                        torch.full_like(x, 0.125 / gamma),
                        torch.full_like(x, 1.0 / gamma))
    # Split the total density into ion and neutral density.
    # ion_frac can be different on the left and right side.
    ion_frac = torch.where(left_mask,
                           torch.full_like(x, ion_frac_L),
                           torch.full_like(x, ion_frac_R))

    rho_i = ion_frac * rho_tot
    rho_n = (1.0 - ion_frac) * rho_tot

    p_i = press_frac_i * p_tot
    p_n = (1.0 - press_frac_i) * p_tot

    # ion velocity
    u_i0 = torch.zeros_like(x) #    # Split total pressure into ion pressure and neutral pressure.

    # Initial neutral velocity.
    # A drift can be added on the left side.
    u_n0 = torch.where(left_mask,
                       torch.full_like(x, drift_n_L),
                       torch.full_like(x, drift_n_R))
    # Convert primitive variables to state variables.
    # Here U = (rho, m, e), where m = rho*u and e = p/(gamma-1).
    U_i = prim_to_state(rho_i, u_i0, p_i, gamma=gamma)
    U_n = prim_to_state(rho_n, u_n0, p_n, gamma=gamma)

    return U_i, U_n



x_test = torch.linspace(-1.0, 1.0, 11)
# ============================================================
# Debug check for the initial condition
# ============================================================
# I use a small grid with only 11 points to check whether the
# left/right states and the ion-neutral splitting are correct.
# ============================================================
U_i_test, U_n_test = initial_condition_two_fluid_sod(x_test, x0=0.0, gamma=gamma)

rho_i_test, u_i_test, p_i_test = stat_one_fluid(U_i_test, gamma=gamma)
rho_n_test, u_n_test, p_n_test = stat_one_fluid(U_n_test, gamma=gamma)

rho_tot_test = rho_i_test + rho_n_test
p_tot_test   = p_i_test + p_n_test

print("x_test =", x_test)
print("rho_i_test =", rho_i_test)
print("rho_n_test =", rho_n_test)
print("rho_tot_test =", rho_tot_test)
print("p_tot_test =", p_tot_test)




#Fixed BC

#We use fixed boundary conditions equal to the initial left and right states. Since the computational domain is sufficiently large and the final simulation time is short, the main wave structures do not significantly interact with the boundaries.
def apply_bc_two_fluid(U_i, U_n, U_i_L, U_i_R, U_n_L, U_n_R):
    U_i[:, 0]  = U_i_L
    U_i[:, -1] = U_i_R

    U_n[:, 0]  = U_n_L
    U_n[:, -1] = U_n_R

    return U_i, U_n

#Debug

x_test = torch.linspace(-1.0, 1.0, 11)

U_i_test, U_n_test = initial_condition_two_fluid_sod(x_test, x0=0.0, gamma=gamma)

U_i_L = U_i_test[:, 0].clone()
U_i_R = U_i_test[:, -1].clone()
U_n_L = U_n_test[:, 0].clone()
U_n_R = U_n_test[:, -1].clone()

# Break the BC
U_i_test[:, 0] = 999.0
U_i_test[:, -1] = -999.0
U_n_test[:, 0] = 888.0
U_n_test[:, -1] = -888.0

# Apply the BC and try again
U_i_test, U_n_test = apply_bc_two_fluid(U_i_test, U_n_test, U_i_L, U_i_R, U_n_L, U_n_R)

print("U_i left recovered =", U_i_test[:, 0])
print("U_i right recovered =", U_i_test[:, -1])
print("U_n left recovered =", U_n_test[:, 0])
print("U_n right recovered =", U_n_test[:, -1])



#The full two fluid shock tube

# Two-fluid Sod shock tube


# Grid
x_min = -5.0
x_max = 5.0
N = 1000
x = torch.linspace(x_min, x_max, N)
dx = x[1] - x[0]

# Initial discontinuity location
x0 = 0.0

# Initial condition
U_i, U_n = initial_condition_two_fluid_sod(x,x0=x0,gamma=gamma,ion_frac_L=0.5,ion_frac_R=0.5,press_frac_i=0.5)

# Save left/right boundary states
U_i_L = U_i[:, 0].clone()
U_i_R = U_i[:, -1].clone()
U_n_L = U_n[:, 0].clone()
U_n_R = U_n[:, -1].clone()

# Collision frequencies
nu_in = 0.5

# Simplest baseline: choose nu_ni from initial left state
rho_i_L, _, _ = stat_one_fluid(U_i[:, 0:1], gamma=gamma)
rho_n_L, _, _ = stat_one_fluid(U_n[:, 0:1], gamma=gamma)
nu_ni = (rho_i_L[0] * nu_in / rho_n_L[0]).item()

print("nu_in =", nu_in)
print("nu_ni =", nu_ni)

# Time
t = 0.0
t_end = 0.08

# History for checking drift
t_hist = []
drift_hist = []

while t < t_end:
    # 1. enforce boundary values before taking derivatives
    U_i, U_n = apply_bc_two_fluid(U_i, U_n, U_i_L, U_i_R, U_n_L, U_n_R)

    # 2. record current state
    rho_i_now, u_i_now, p_i_now = stat_one_fluid(U_i, gamma=gamma)
    rho_n_now, u_n_now, p_n_now = stat_one_fluid(U_n, gamma=gamma)

    t_hist.append(t)
    drift_hist.append(torch.mean(torch.abs(u_i_now - u_n_now)).item())

    # 3. CFL time step
    dt = compute_dt_two_fluid(U_i, U_n, dx, gamma=gamma, cfl=0.02).item()

    if t + dt > t_end:
        dt = t_end - t

    if dt <= 0:
        print("dt <= 0")
        break

    # 4. take one explicit Euler step
    U_i, U_n = euler_step_two_fluid(U_i, U_n, dx, dt, nu_in, gamma=gamma)

    # 5. re-apply boundary conditions after the update
    U_i, U_n = apply_bc_two_fluid(U_i, U_n, U_i_L, U_i_R, U_n_L, U_n_R)

    t += dt

print("final time =", t)
rho_i, u_i, p_i = stat_one_fluid(U_i, gamma=gamma)
rho_n, u_n, p_n = stat_one_fluid(U_n, gamma=gamma)

rho_tot = rho_i + rho_n
m_tot   = U_i[1] + U_n[1]
u_tot   = m_tot / torch.clamp(rho_tot, min=eps)
p_tot   = p_i + p_n

print("min rho_i =", torch.min(rho_i).item())
print("min rho_n =", torch.min(rho_n).item())
print("min p_i   =", torch.min(p_i).item())
print("min p_n   =", torch.min(p_n).item())
print("final mean drift =", torch.mean(torch.abs(u_i - u_n)).item())



rho_i, u_i, p_i = stat_one_fluid(U_i, gamma=gamma)
rho_n, u_n, p_n = stat_one_fluid(U_n, gamma=gamma)

rho_tot = rho_i + rho_n
m_tot   = U_i[1] + U_n[1]
u_tot   = m_tot / torch.clamp(rho_tot, min=eps)
p_tot   = p_i + p_n


#Debug
print("any NaN in rho_tot?", torch.isnan(rho_tot).any().item())
print("any NaN in u_tot?",   torch.isnan(u_tot).any().item())
print("any NaN in p_tot?",   torch.isnan(p_tot).any().item())

print("any inf in rho_tot?", torch.isinf(rho_tot).any().item())
print("any inf in u_tot?",   torch.isinf(u_tot).any().item())
print("any inf in p_tot?",   torch.isinf(p_tot).any().item())


#Change to numpy and plot


x_np = x.detach().cpu().numpy()
rho_i_np   = rho_i.detach().cpu().numpy()
rho_n_np   = rho_n.detach().cpu().numpy()
rho_tot_np = rho_tot.detach().cpu().numpy()
u_i_np   = u_i.detach().cpu().numpy()
u_n_np   = u_n.detach().cpu().numpy()
u_tot_np = u_tot.detach().cpu().numpy()
p_i_np   = p_i.detach().cpu().numpy()
p_n_np   = p_n.detach().cpu().numpy()
p_tot_np = p_tot.detach().cpu().numpy()


#Plot

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(t_hist, drift_hist, label='mean |u_i - u_n|')
ax.set_xlabel('t')
ax.set_ylabel('drift')
ax.set_title('Drift during two-fluid shock-tube run')
ax.legend()
ax.grid()

save_figure(fig, "01_drift_two_fluid_shock_tube.png")
plt.show()
plt.close(fig)

#single-fluid Sod exact solution

#Give some Starpressure p, and compute the Riemann problem, the left or right nonlnear function.
def pressure_function(p, rho_k, u_k, p_k, gamma):
    a_k = math.sqrt(gamma * p_k / rho_k) # The speed of the sound.
    A_k = 2.0 / ((gamma + 1.0) * rho_k)
    B_k = (gamma - 1.0) / (gamma + 1.0) * p_k

    if p > p_k:   # shock
        f = (p - p_k) * math.sqrt(A_k / (p + B_k))
        fd = math.sqrt(A_k / (p + B_k)) * (1.0 - 0.5 * (p - p_k) / (p + B_k))
    else:         # rarefaction
        pr = p / p_k
        f = (2.0 * a_k / (gamma - 1.0)) * (pr**((gamma - 1.0) / (2.0 * gamma)) - 1.0)
        fd = (1.0 / (rho_k * a_k)) * pr**(-(gamma + 1.0) / (2.0 * gamma))

    return f, fd


def star_pressure_velocity(rho_L, u_L, p_L, rho_R, u_R, p_R, gamma,
                           tol=1e-10, max_iter=100):
    a_L = math.sqrt(gamma * p_L / rho_L)
    a_R = math.sqrt(gamma * p_R / rho_R)

    p_guess = 0.5 * (p_L + p_R) - 0.125 * (u_R - u_L) * (rho_L + rho_R) * (a_L + a_R)
    p = max(1e-8, p_guess)

    for _ in range(max_iter):
        fL, fdL = pressure_function(p, rho_L, u_L, p_L, gamma)
        fR, fdR = pressure_function(p, rho_R, u_R, p_R, gamma)

        p_new = p - (fL + fR + u_R - u_L) / (fdL + fdR)
        p_new = max(1e-8, p_new)

        if abs(p_new - p) / (0.5 * (p_new + p)) < tol:
            p = p_new
            break

        p = p_new

    fL, _ = pressure_function(p, rho_L, u_L, p_L, gamma)
    fR, _ = pressure_function(p, rho_R, u_R, p_R, gamma)

    u_star = 0.5 * (u_L + u_R + fR - fL)

    return p, u_star


def exact_solution_sod(x, t, x0, rho_L, u_L, p_L, rho_R, u_R, p_R, gamma, p_star, u_star):
    rho_ex = torch.zeros_like(x)
    u_ex   = torch.zeros_like(x)
    p_ex   = torch.zeros_like(x)

    a_L = math.sqrt(gamma * p_L / rho_L)
    a_R = math.sqrt(gamma * p_R / rho_R)

    # left star density
    rho_star_L = rho_L * ((p_star / p_L + (gamma - 1.0)/(gamma + 1.0)) /
                          (((gamma - 1.0)/(gamma + 1.0)) * (p_star / p_L) + 1.0))

    # right star density
    rho_star_R = rho_R * (p_star / p_R)**(1.0 / gamma)

    xi = (x - x0) / t

    # left shock speed
    S_L = u_L - a_L * math.sqrt((gamma + 1.0)/(2.0*gamma) * (p_star/p_L)
                                + (gamma - 1.0)/(2.0*gamma))

    S_contact = u_star

    a_star_R = a_R * (p_star / p_R)**((gamma - 1.0)/(2.0*gamma))
    S_tail_R = u_star + a_star_R
    S_head_R = u_R + a_R

    # 1. left constant state
    mask1 = xi <= S_L
    rho_ex[mask1] = rho_L
    u_ex[mask1]   = u_L
    p_ex[mask1]   = p_L

    # 2. left star state
    mask2 = (xi > S_L) & (xi <= S_contact)
    rho_ex[mask2] = rho_star_L
    u_ex[mask2]   = u_star
    p_ex[mask2]   = p_star

    # 3. right star state
    mask3 = (xi > S_contact) & (xi <= S_tail_R)
    rho_ex[mask3] = rho_star_R
    u_ex[mask3]   = u_star
    p_ex[mask3]   = p_star

    # 4. right rarefaction fan
    mask4 = (xi > S_tail_R) & (xi < S_head_R)
    xi_fan = xi[mask4]

    u_fan = 2.0 / (gamma + 1.0) * (-a_R + 0.5*(gamma - 1.0)*u_R + xi_fan)
    a_fan = 2.0 / (gamma + 1.0) * (a_R - 0.5*(gamma - 1.0)*(u_R - xi_fan))

    rho_fan = rho_R * (a_fan / a_R)**(2.0 / (gamma - 1.0))
    p_fan   = p_R   * (a_fan / a_R)**(2.0 * gamma / (gamma - 1.0))

    rho_ex[mask4] = rho_fan
    u_ex[mask4]   = u_fan
    p_ex[mask4]   = p_fan

    # 5. right constant state
    mask5 = xi >= S_head_R
    rho_ex[mask5] = rho_R
    u_ex[mask5]   = u_R
    p_ex[mask5]   = p_R

    return rho_ex, u_ex, p_ex, S_L, S_contact, S_tail_R, S_head_R


rho_L = 0.125
u_L   = 0.0
p_L   = 0.125 / gamma

rho_R = 1.0
u_R   = 0.0
p_R   = 1.0 / gamma


p_star, u_star = star_pressure_velocity(rho_L, u_L, p_L, rho_R, u_R, p_R, gamma)

rho_ex, u_ex, p_ex, S_L, S_contact, S_tail_R, S_head_R = exact_solution_sod(
    x, t_end, x0, rho_L, u_L, p_L, rho_R, u_R, p_R, gamma, p_star, u_star
)

print("p_star =", p_star)
print("u_star =", u_star)

x_np      = x.detach().cpu().numpy()
rho_ex_np = rho_ex.detach().cpu().numpy()
u_ex_np   = u_ex.detach().cpu().numpy()
p_ex_np   = p_ex.detach().cpu().numpy()

#Error 

L1_rho   = torch.mean(torch.abs(rho_tot - rho_ex)).item()
Linf_rho = torch.max(torch.abs(rho_tot - rho_ex)).item()

L1_u     = torch.mean(torch.abs(u_tot - u_ex)).item()
Linf_u   = torch.max(torch.abs(u_tot - u_ex)).item()

L1_p     = torch.mean(torch.abs(p_tot - p_ex)).item()
Linf_p   = torch.max(torch.abs(p_tot - p_ex)).item()

print("L1 error in total density   =", L1_rho)
print("Linf error in total density =", Linf_rho)

print("L1 error in total velocity   =", L1_u)
print("Linf error in total velocity =", Linf_u)

print("L1 error in total pressure   =", L1_p)
print("Linf error in total pressure =", Linf_p)


#Define the initial two-fluid shock tube

def initial_condition_two_fluid_sod(x, x0=0.0, gamma=gamma,
                                    ion_frac_L=0.5, ion_frac_R=0.5,
                                    press_frac_i=0.5,
                                    drift_n_L=0.2, drift_n_R=0.0):
    left_mask = x < x0

    rho_tot = torch.where(left_mask,
                          torch.full_like(x, 0.125),
                          torch.full_like(x, 1.0))

    p_tot = torch.where(left_mask,
                        torch.full_like(x, 0.125 / gamma),
                        torch.full_like(x, 1.0 / gamma))

    ion_frac = torch.where(left_mask,
                           torch.full_like(x, ion_frac_L),
                           torch.full_like(x, ion_frac_R))

    rho_i = ion_frac * rho_tot
    rho_n = (1.0 - ion_frac) * rho_tot

    p_i = press_frac_i * p_tot
    p_n = (1.0 - press_frac_i) * p_tot

    u_i0 = torch.zeros_like(x)
    u_n0 = torch.where(left_mask,
                       torch.full_like(x, drift_n_L),
                       torch.full_like(x, drift_n_R))

    U_i = prim_to_state(rho_i, u_i0, p_i, gamma=gamma)
    U_n = prim_to_state(rho_n, u_n0, p_n, gamma=gamma)

    return U_i, U_n


#Fixed BC

def apply_bc_two_fluid(U_i, U_n, U_i_L, U_i_R, U_n_L, U_n_R):
    U_i[:, 0]  = U_i_L
    U_i[:, -1] = U_i_R

    U_n[:, 0]  = U_n_L
    U_n[:, -1] = U_n_R

    return U_i, U_n

#single-fluid Sod exact


def pressure_function(p, rho_k, u_k, p_k, gamma):
    a_k = math.sqrt(gamma * p_k / rho_k)
    A_k = 2.0 / ((gamma + 1.0) * rho_k)
    B_k = (gamma - 1.0) / (gamma + 1.0) * p_k

    if p > p_k:   # shock
        f = (p - p_k) * math.sqrt(A_k / (p + B_k))
        fd = math.sqrt(A_k / (p + B_k)) * (1.0 - 0.5 * (p - p_k) / (p + B_k))
    else:         # rarefaction
        pr = p / p_k
        f = (2.0 * a_k / (gamma - 1.0)) * (pr**((gamma - 1.0) / (2.0 * gamma)) - 1.0)
        fd = (1.0 / (rho_k * a_k)) * pr**(-(gamma + 1.0) / (2.0 * gamma))

    return f, fd


def star_pressure_velocity(rho_L, u_L, p_L, rho_R, u_R, p_R, gamma,
                           tol=1e-10, max_iter=100):
    a_L = math.sqrt(gamma * p_L / rho_L)
    a_R = math.sqrt(gamma * p_R / rho_R)

    p_guess = 0.5 * (p_L + p_R) - 0.125 * (u_R - u_L) * (rho_L + rho_R) * (a_L + a_R)
    p = max(1e-8, p_guess)

    for _ in range(max_iter):
        fL, fdL = pressure_function(p, rho_L, u_L, p_L, gamma)
        fR, fdR = pressure_function(p, rho_R, u_R, p_R, gamma)

        p_new = p - (fL + fR + u_R - u_L) / (fdL + fdR)
        p_new = max(1e-8, p_new)

        if abs(p_new - p) / (0.5 * (p_new + p)) < tol:
            p = p_new
            break

        p = p_new

    fL, _ = pressure_function(p, rho_L, u_L, p_L, gamma)
    fR, _ = pressure_function(p, rho_R, u_R, p_R, gamma)

    u_star = 0.5 * (u_L + u_R + fR - fL)
    return p, u_star


def exact_solution_sod(x, t, x0, rho_L, u_L, p_L, rho_R, u_R, p_R, gamma, p_star, u_star):
    rho_ex = torch.zeros_like(x)
    u_ex   = torch.zeros_like(x)
    p_ex   = torch.zeros_like(x)
    a_L = math.sqrt(gamma * p_L / rho_L)
    a_R = math.sqrt(gamma * p_R / rho_R)
    rho_star_L = rho_L * ((p_star / p_L + (gamma - 1.0)/(gamma + 1.0)) /
                          (((gamma - 1.0)/(gamma + 1.0)) * (p_star / p_L) + 1.0))
    rho_star_R = rho_R * (p_star / p_R)**(1.0 / gamma)
    xi = (x - x0) / t
    S_L = u_L - a_L * math.sqrt((gamma + 1.0)/(2.0*gamma) * (p_star/p_L)
                                + (gamma - 1.0)/(2.0*gamma))
    S_contact = u_star

    a_star_R = a_R * (p_star / p_R)**((gamma - 1.0)/(2.0*gamma))
    S_tail_R = u_star + a_star_R
    S_head_R = u_R + a_R

    mask1 = xi <= S_L
    rho_ex[mask1] = rho_L
    u_ex[mask1]   = u_L
    p_ex[mask1]   = p_L

    mask2 = (xi > S_L) & (xi <= S_contact)
    rho_ex[mask2] = rho_star_L
    u_ex[mask2]   = u_star
    p_ex[mask2]   = p_star

    mask3 = (xi > S_contact) & (xi <= S_tail_R)
    rho_ex[mask3] = rho_star_R
    u_ex[mask3]   = u_star
    p_ex[mask3]   = p_star

    mask4 = (xi > S_tail_R) & (xi < S_head_R)
    xi_fan = xi[mask4]

    u_fan = 2.0 / (gamma + 1.0) * (-a_R + 0.5*(gamma - 1.0)*u_R + xi_fan)
    a_fan = 2.0 / (gamma + 1.0) * (a_R - 0.5*(gamma - 1.0)*(u_R - xi_fan))

    rho_fan = rho_R * (a_fan / a_R)**(2.0 / (gamma - 1.0))
    p_fan   = p_R   * (a_fan / a_R)**(2.0 * gamma / (gamma - 1.0))

    rho_ex[mask4] = rho_fan
    u_ex[mask4]   = u_fan
    p_ex[mask4]   = p_fan

    mask5 = xi >= S_head_R
    rho_ex[mask5] = rho_R
    u_ex[mask5]   = u_R
    p_ex[mask5]   = p_R

    return rho_ex, u_ex, p_ex


#Run always function

def run_two_fluid_sod_case(nu_in,
                           x_min=-5.0, x_max=5.0, N=1000,
                           x0=0.0, t_end=0.08,
                           ion_frac_L=0.5, ion_frac_R=0.5,
                           press_frac_i=0.5,
                           drift_n_L=0.2, drift_n_R=0.0,
                           cfl=0.02,
                           gamma=gamma):
    x = torch.linspace(x_min, x_max, N)
    dx = x[1] - x[0]

    U_i, U_n = initial_condition_two_fluid_sod(
        x, x0=x0, gamma=gamma,
        ion_frac_L=ion_frac_L, ion_frac_R=ion_frac_R,
        press_frac_i=press_frac_i,
        drift_n_L=drift_n_L, drift_n_R=drift_n_R
    )

    U_i_L = U_i[:, 0].clone()
    U_i_R = U_i[:, -1].clone()
    U_n_L = U_n[:, 0].clone()
    U_n_R = U_n[:, -1].clone()

    rho_i_L, _, _ = stat_one_fluid(U_i[:, 0:1], gamma=gamma)
    rho_n_L, _, _ = stat_one_fluid(U_n[:, 0:1], gamma=gamma)
    nu_ni_local = (rho_i_L[0] * nu_in / torch.clamp(rho_n_L[0], min=eps)).item()

    t = 0.0
    t_hist = []
    drift_hist = []

    while t < t_end:
        U_i, U_n = apply_bc_two_fluid(U_i, U_n, U_i_L, U_i_R, U_n_L, U_n_R)

        rho_i_now, u_i_now, p_i_now = stat_one_fluid(U_i, gamma=gamma)
        rho_n_now, u_n_now, p_n_now = stat_one_fluid(U_n, gamma=gamma)

        t_hist.append(t)
        drift_hist.append(torch.mean(torch.abs(u_i_now - u_n_now)).item())

        dt = compute_dt_two_fluid(U_i, U_n, dx, gamma=gamma, cfl=cfl).item()

        if t + dt > t_end:
            dt = t_end - t
        if dt <= 0:
            print("dt <= 0")
            break

        U_i, U_n = euler_step_two_fluid(U_i, U_n, dx, dt, nu_in, gamma=gamma)
        U_i, U_n = apply_bc_two_fluid(U_i, U_n, U_i_L, U_i_R, U_n_L, U_n_R)

        t += dt

    rho_i, u_i, p_i = stat_one_fluid(U_i, gamma=gamma)
    rho_n, u_n, p_n = stat_one_fluid(U_n, gamma=gamma)

    rho_tot = rho_i + rho_n
    m_tot   = U_i[1] + U_n[1]
    u_tot   = m_tot / torch.clamp(rho_tot, min=eps)
    p_tot   = p_i + p_n

    rho_L = 0.125
    u_L   = 0.0
    p_L   = 0.125 / gamma

    rho_R = 1.0
    u_R   = 0.0
    p_R   = 1.0 / gamma

    p_star, u_star = star_pressure_velocity(rho_L, u_L, p_L, rho_R, u_R, p_R, gamma)
    rho_ex, u_ex, p_ex = exact_solution_sod(
        x, t_end, x0, rho_L, u_L, p_L, rho_R, u_R, p_R, gamma, p_star, u_star
    )

    L1_rho   = torch.mean(torch.abs(rho_tot - rho_ex)).item()
    Linf_rho = torch.max(torch.abs(rho_tot - rho_ex)).item()

    L1_u     = torch.mean(torch.abs(u_tot - u_ex)).item()
    Linf_u   = torch.max(torch.abs(u_tot - u_ex)).item()

    L1_p     = torch.mean(torch.abs(p_tot - p_ex)).item()
    Linf_p   = torch.max(torch.abs(p_tot - p_ex)).item()

    mean_drift_final = torch.mean(torch.abs(u_i - u_n)).item()
    max_drift_final  = torch.max(torch.abs(u_i - u_n)).item()

    result = {
        "x": x,
        "rho_i": rho_i, "u_i": u_i, "p_i": p_i,
        "rho_n": rho_n, "u_n": u_n, "p_n": p_n,
        "rho_tot": rho_tot, "u_tot": u_tot, "p_tot": p_tot,
        "rho_ex": rho_ex, "u_ex": u_ex, "p_ex": p_ex,
        "t_hist": t_hist, "drift_hist": drift_hist,
        "errors": {
            "L1_rho": L1_rho, "Linf_rho": Linf_rho,
            "L1_u": L1_u, "Linf_u": Linf_u,
            "L1_p": L1_p, "Linf_p": Linf_p,
        },
        "diagnostics": {
            "mean_drift_final": mean_drift_final,
            "max_drift_final": max_drift_final,
        },
        "nu_in": nu_in,
        "final_time": t,
    }

    return result


#plot

def plot_two_fluid_result(result):
    x_np = result["x"].detach().cpu().numpy()

    rho_i_np   = result["rho_i"].detach().cpu().numpy()
    rho_n_np   = result["rho_n"].detach().cpu().numpy()
    rho_tot_np = result["rho_tot"].detach().cpu().numpy()
    rho_ex_np  = result["rho_ex"].detach().cpu().numpy()

    u_i_np   = result["u_i"].detach().cpu().numpy()
    u_n_np   = result["u_n"].detach().cpu().numpy()
    u_tot_np = result["u_tot"].detach().cpu().numpy()
    u_ex_np  = result["u_ex"].detach().cpu().numpy()

    p_i_np   = result["p_i"].detach().cpu().numpy()
    p_n_np   = result["p_n"].detach().cpu().numpy()
    p_tot_np = result["p_tot"].detach().cpu().numpy()
    p_ex_np  = result["p_ex"].detach().cpu().numpy()

    nu_in = result["nu_in"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].plot(x_np, rho_i_np, label='rho_i')
    axes[0, 0].plot(x_np, rho_n_np, label='rho_n')
    axes[0, 0].plot(x_np, rho_tot_np, label='rho_total', linewidth=2)
    axes[0, 0].plot(x_np, rho_ex_np, '--', label='single-fluid exact')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('density')
    axes[0, 0].set_title('Density')
    axes[0, 0].legend()
    axes[0, 0].grid()

    axes[0, 1].plot(x_np, u_i_np, label='u_i')
    axes[0, 1].plot(x_np, u_n_np, label='u_n')
    axes[0, 1].plot(x_np, u_tot_np, label='u_total', linewidth=2)
    axes[0, 1].plot(x_np, u_ex_np, '--', label='single-fluid exact')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('velocity')
    axes[0, 1].set_title('Velocity')
    axes[0, 1].legend()
    axes[0, 1].grid()

    axes[1, 0].plot(x_np, p_i_np, label='p_i')
    axes[1, 0].plot(x_np, p_n_np, label='p_n')
    axes[1, 0].plot(x_np, p_tot_np, label='p_total', linewidth=2)
    axes[1, 0].plot(x_np, p_ex_np, '--', label='single-fluid exact')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('pressure')
    axes[1, 0].set_title('Pressure')
    axes[1, 0].legend()
    axes[1, 0].grid()

    axes[1, 1].plot(result["t_hist"], result["drift_hist"])
    axes[1, 1].set_xlabel('t')
    axes[1, 1].set_ylabel('mean |u_i - u_n|')
    axes[1, 1].set_title('Drift history')
    axes[1, 1].grid()

    fig.suptitle(f'Two-fluid result, nu_in = {nu_in}', fontsize=14)

    save_figure(fig, f"02_two_fluid_result_nu_{nu_in}.png")
    plt.show()
    plt.close(fig)

nu_list = [0.0, 0.1, 0.5, 1.0]
results = []

for nu in nu_list:
    print(f"\nRunning case nu_in = {nu}")
    result = run_two_fluid_sod_case(
        nu_in=nu,
        ion_frac_L=0.8,
        ion_frac_R=0.2,
        drift_n_L=0.2,
        drift_n_R=0.0
    )
    results.append(result)

    err = result["errors"]
    diag = result["diagnostics"]

    print("L1_rho   =", err["L1_rho"])
    print("Linf_rho =", err["Linf_rho"])
    print("L1_u     =", err["L1_u"])
    print("Linf_u   =", err["Linf_u"])
    print("L1_p     =", err["L1_p"])
    print("Linf_p   =", err["Linf_p"])
    print("mean_drift_final =", diag["mean_drift_final"])
    print("max_drift_final  =", diag["max_drift_final"])

plot_two_fluid_result(results[2])


nu_vals = [r["nu_in"] for r in results]
L1_rho_vals = [r["errors"]["L1_rho"] for r in results]
L1_u_vals   = [r["errors"]["L1_u"] for r in results]
L1_p_vals   = [r["errors"]["L1_p"] for r in results]




# Test 1：drift relaxation
#No heating, constant collision frequency: compute the slope of the velocity drift

#w(t) = w_i(t)-u_n(t) , w'(t)= -(nu_in+nu_ni)w

#Solve this ODE and the solution for this ODE is w(t)=w(0)*exp(-(nu_in+nu_ni))

def run_drift_relaxation_case(nu_in,
                              rho_i0=0.2, rho_n0=0.8,
                              u_i0=1.0, u_n0=0.0,
                              p_i0=0.1, p_n0=0.1,
                              x_min=-1.0, x_max=1.0, N=200,
                              t_end=0.2, cfl=0.2,
                              gamma=gamma):
    x = torch.linspace(x_min, x_max, N)
    dx = x[1] - x[0]
    rho_i = rho_i0 * torch.ones_like(x)
    rho_n = rho_n0 * torch.ones_like(x)
    u_i = u_i0 * torch.ones_like(x)
    u_n = u_n0 * torch.ones_like(x)
    p_i = p_i0 * torch.ones_like(x)
    p_n = p_n0 * torch.ones_like(x)

    U_i = torch.stack([rho_i, rho_i * u_i, p_i / (gamma - 1.0)], dim=0)
    U_n = torch.stack([rho_n, rho_n * u_n, p_n / (gamma - 1.0)], dim=0)

    nu_ni = rho_i0 * nu_in / rho_n0
    lam = nu_in + nu_ni

    t = 0.0
    t_hist = [t]
    drift_hist = [torch.mean(torch.abs(u_i - u_n)).item()]
    exact_hist = [abs(u_i0 - u_n0) * math.exp(-lam * t)]

    while t < t_end:
        dt = compute_dt_two_fluid(U_i, U_n, dx, gamma=gamma, cfl=cfl).item()

        if t + dt > t_end:
            dt = t_end - t
        if dt <= 0:
            print("dt <= 0")
            break

        U_i, U_n = euler_step_two_fluid(U_i, U_n, dx, dt, nu_in, gamma=gamma)
        t += dt

        rho_i_now, u_i_now, p_i_now = stat_one_fluid(U_i, gamma=gamma)
        rho_n_now, u_n_now, p_n_now = stat_one_fluid(U_n, gamma=gamma)

        t_hist.append(t)
        drift_hist.append(torch.mean(torch.abs(u_i_now - u_n_now)).item())
        exact_hist.append(abs(u_i0 - u_n0) * math.exp(-lam * t))

    return {
        "nu_in": nu_in,
        "nu_ni": nu_ni,
        "lambda_exact": lam,
        "t_hist": np.array(t_hist),
        "drift_hist": np.array(drift_hist),
        "exact_hist": np.array(exact_hist),
        "final_drift": drift_hist[-1],
    }

def fit_semilogy_slope(t_hist, drift_hist, fit_start=0.0):
    mask = (t_hist >= fit_start) & (drift_hist > 0)

    t_fit = t_hist[mask]
    y_fit = np.log(drift_hist[mask])

    coef = np.polyfit(t_fit, y_fit, 1)
    slope = coef[0]
    intercept = coef[1]

    return slope, intercept


nu_list_drift = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
drift_results = []

for nu in nu_list_drift:
    res = run_drift_relaxation_case(nu_in=nu, t_end=0.2)
    slope_num, intercept_num = fit_semilogy_slope(res["t_hist"], res["drift_hist"], fit_start=0.0)

    print(f"\nnu_in = {nu}")
    print("nu_ni          =", res["nu_ni"])
    print("lambda_exact   =", res["lambda_exact"])
    print("slope_num      =", slope_num)
    print("final_drift    =", res["final_drift"])

    drift_results.append({
        "nu_in": nu,
        "nu_ni": res["nu_ni"],
        "lambda_exact": res["lambda_exact"],
        "slope_num": slope_num,
        "final_drift": res["final_drift"],
        "full_result": res
    })



plt.figure(figsize=(8,5))

for item in drift_results:
    res = item["full_result"]
    plt.semilogy(res["t_hist"], res["drift_hist"], label=f'nu_in={item["nu_in"]}')

plt.xlabel('t')
plt.ylabel('mean |u_i - u_n|')
plt.title('Drift relaxation (semi-log)')
plt.grid()
plt.legend()


res = drift_results[3]["full_result"]   # Corresponding nu_in = 0.5, compare with exact exponential

plt.figure(figsize=(8,5))
plt.semilogy(res["t_hist"], res["drift_hist"], label='numerical drift')
plt.semilogy(res["t_hist"], res["exact_hist"], '--', label='exact exponential')
plt.xlabel('t')
plt.ylabel('mean |u_i - u_n|')
plt.title(f'Drift vs exact, nu_in={res["nu_in"]}')
plt.grid()
plt.legend()
plt.tight_layout()
plt.close()

rows = []
for item in drift_results:
    lam_exact = item["lambda_exact"]
    slope_num = item["slope_num"]

    rows.append({
        "nu_in": item["nu_in"],
        "nu_ni": item["nu_ni"],
        "lambda_exact": lam_exact,
        "slope_num": slope_num,
        "abs_error": abs(slope_num + lam_exact),
        "rel_error": abs(slope_num + lam_exact) / lam_exact
    })

df_drift = pd.DataFrame(rows)
print(df_drift)


plt.figure(figsize=(8,5))
plt.plot(df_drift["nu_in"], -df_drift["slope_num"], marker='o', label='- numerical slope')
plt.plot(df_drift["nu_in"], df_drift["lambda_exact"], marker='s', label='exact lambda')
plt.xlabel('nu_in')
plt.ylabel('decay rate')
plt.title('Decay rate: numerical vs exact')
plt.grid()
plt.legend()
plt.tight_layout()
plt.close()


m_tot = U_i[1] + U_n[1]
print("mean total momentum =", torch.mean(m_tot).item())

# Test 2: two-fluid Sod shock tube


def initial_condition_two_fluid_sod(x, x0=0.0, gamma=gamma,
                                    ion_frac_L=0.5, ion_frac_R=0.5,
                                    press_frac_i=0.5,
                                    drift_n_L=0.0, drift_n_R=0.0):
    left_mask = x < x0

    rho_tot = torch.where(left_mask,
                          torch.full_like(x, 0.125),
                          torch.full_like(x, 1.0))

    p_tot = torch.where(left_mask,
                        torch.full_like(x, 0.125 / gamma),
                        torch.full_like(x, 1.0 / gamma))

    ion_frac = torch.where(left_mask,
                           torch.full_like(x, ion_frac_L),
                           torch.full_like(x, ion_frac_R))

    rho_i = ion_frac * rho_tot
    rho_n = (1.0 - ion_frac) * rho_tot

    p_i = press_frac_i * p_tot
    p_n = (1.0 - press_frac_i) * p_tot

    # ion velocity
    u_i0 = torch.zeros_like(x)

    # neutral velocity
    u_n0 = torch.where(left_mask,
                       torch.full_like(x, drift_n_L),
                       torch.full_like(x, drift_n_R))

    U_i = prim_to_state(rho_i, u_i0, p_i, gamma=gamma)
    U_n = prim_to_state(rho_n, u_n0, p_n, gamma=gamma)

    return U_i, U_n


def apply_bc_two_fluid(U_i, U_n, U_i_L, U_i_R, U_n_L, U_n_R):
    U_i[:, 0]  = U_i_L
    U_i[:, -1] = U_i_R

    U_n[:, 0]  = U_n_L
    U_n[:, -1] = U_n_R

    return U_i, U_n


def pressure_function(p, rho_k, u_k, p_k, gamma):
    a_k = math.sqrt(gamma * p_k / rho_k)
    A_k = 2.0 / ((gamma + 1.0) * rho_k)
    B_k = (gamma - 1.0) / (gamma + 1.0) * p_k

    if p > p_k:   # shock
        f = (p - p_k) * math.sqrt(A_k / (p + B_k))
        fd = math.sqrt(A_k / (p + B_k)) * (1.0 - 0.5 * (p - p_k) / (p + B_k))
    else:         # rarefaction
        pr = p / p_k
        f = (2.0 * a_k / (gamma - 1.0)) * (pr**((gamma - 1.0) / (2.0 * gamma)) - 1.0)
        fd = (1.0 / (rho_k * a_k)) * pr**(-(gamma + 1.0) / (2.0 * gamma))

    return f, fd


def star_pressure_velocity(rho_L, u_L, p_L, rho_R, u_R, p_R, gamma,
                           tol=1e-10, max_iter=100):
    a_L = math.sqrt(gamma * p_L / rho_L)
    a_R = math.sqrt(gamma * p_R / rho_R)

    p_guess = 0.5 * (p_L + p_R) - 0.125 * (u_R - u_L) * (rho_L + rho_R) * (a_L + a_R)
    p = max(1e-8, p_guess)

    for _ in range(max_iter):
        fL, fdL = pressure_function(p, rho_L, u_L, p_L, gamma)
        fR, fdR = pressure_function(p, rho_R, u_R, p_R, gamma)

        p_new = p - (fL + fR + u_R - u_L) / (fdL + fdR)
        p_new = max(1e-8, p_new)

        if abs(p_new - p) / (0.5 * (p_new + p)) < tol:
            p = p_new
            break

        p = p_new

    fL, _ = pressure_function(p, rho_L, u_L, p_L, gamma)
    fR, _ = pressure_function(p, rho_R, u_R, p_R, gamma)

    u_star = 0.5 * (u_L + u_R + fR - fL)
    return p, u_star


def exact_solution_sod(x, t, x0, rho_L, u_L, p_L, rho_R, u_R, p_R, gamma, p_star, u_star):
    rho_ex = torch.zeros_like(x)
    u_ex   = torch.zeros_like(x)
    p_ex   = torch.zeros_like(x)

    a_L = math.sqrt(gamma * p_L / rho_L)
    a_R = math.sqrt(gamma * p_R / rho_R)

    rho_star_L = rho_L * ((p_star / p_L + (gamma - 1.0)/(gamma + 1.0)) /
                          (((gamma - 1.0)/(gamma + 1.0)) * (p_star / p_L) + 1.0))

    rho_star_R = rho_R * (p_star / p_R)**(1.0 / gamma)

    xi = (x - x0) / t

    # left shock
    S_L = u_L - a_L * math.sqrt((gamma + 1.0)/(2.0*gamma) * (p_star/p_L)
                                + (gamma - 1.0)/(2.0*gamma))

    S_contact = u_star

    # right rarefaction
    a_star_R = a_R * (p_star / p_R)**((gamma - 1.0)/(2.0*gamma))
    S_tail_R = u_star + a_star_R
    S_head_R = u_R + a_R

    mask1 = xi <= S_L
    rho_ex[mask1] = rho_L
    u_ex[mask1]   = u_L
    p_ex[mask1]   = p_L

    mask2 = (xi > S_L) & (xi <= S_contact)
    rho_ex[mask2] = rho_star_L
    u_ex[mask2]   = u_star
    p_ex[mask2]   = p_star

    mask3 = (xi > S_contact) & (xi <= S_tail_R)
    rho_ex[mask3] = rho_star_R
    u_ex[mask3]   = u_star
    p_ex[mask3]   = p_star

    mask4 = (xi > S_tail_R) & (xi < S_head_R)
    xi_fan = xi[mask4]

    u_fan = 2.0 / (gamma + 1.0) * (-a_R + 0.5*(gamma - 1.0)*u_R + xi_fan)
    a_fan = 2.0 / (gamma + 1.0) * (a_R - 0.5*(gamma - 1.0)*(u_R - xi_fan))

    rho_fan = rho_R * (a_fan / a_R)**(2.0 / (gamma - 1.0))
    p_fan   = p_R   * (a_fan / a_R)**(2.0 * gamma / (gamma - 1.0))

    rho_ex[mask4] = rho_fan
    u_ex[mask4]   = u_fan
    p_ex[mask4]   = p_fan

    mask5 = xi >= S_head_R
    rho_ex[mask5] = rho_R
    u_ex[mask5]   = u_R
    p_ex[mask5]   = p_R

    return rho_ex, u_ex, p_ex


def run_two_fluid_sod_case(nu_in,
                           x_min=-5.0, x_max=5.0, N=1000,
                           x0=0.0, t_end=0.08,
                           ion_frac_L=0.8, ion_frac_R=0.2,
                           press_frac_i=0.5,
                           drift_n_L=0.0, drift_n_R=0.0,
                           cfl=0.02,
                           gamma=gamma):

    x = torch.linspace(x_min, x_max, N)
    dx = x[1] - x[0]

    U_i, U_n = initial_condition_two_fluid_sod(
        x, x0=x0, gamma=gamma,
        ion_frac_L=ion_frac_L, ion_frac_R=ion_frac_R,
        press_frac_i=press_frac_i,
        drift_n_L=drift_n_L, drift_n_R=drift_n_R
    )

    U_i_L = U_i[:, 0].clone()
    U_i_R = U_i[:, -1].clone()
    U_n_L = U_n[:, 0].clone()
    U_n_R = U_n[:, -1].clone()

    t = 0.0
    t_hist = []
    drift_hist = []

    while t < t_end:
        U_i, U_n = apply_bc_two_fluid(U_i, U_n, U_i_L, U_i_R, U_n_L, U_n_R)

        rho_i_now, u_i_now, p_i_now = stat_one_fluid(U_i, gamma=gamma)
        rho_n_now, u_n_now, p_n_now = stat_one_fluid(U_n, gamma=gamma)

        t_hist.append(t)
        drift_hist.append(torch.mean(torch.abs(u_i_now - u_n_now)).item())

        dt = compute_dt_two_fluid(U_i, U_n, dx, gamma=gamma, cfl=cfl).item()

        if t + dt > t_end:
            dt = t_end - t
        if dt <= 0:
            print("dt <= 0")
            break

        U_i, U_n = euler_step_two_fluid(U_i, U_n, dx, dt, nu_in, gamma=gamma)
        U_i, U_n = apply_bc_two_fluid(U_i, U_n, U_i_L, U_i_R, U_n_L, U_n_R)

        t += dt

    rho_i, u_i, p_i = stat_one_fluid(U_i, gamma=gamma)
    rho_n, u_n, p_n = stat_one_fluid(U_n, gamma=gamma)

    rho_tot = rho_i + rho_n
    m_tot   = U_i[1] + U_n[1]
    u_tot   = m_tot / torch.clamp(rho_tot, min=eps)
    p_tot   = p_i + p_n

    # exact single-fluid Sod for total variables
    rho_L = 0.125
    u_L   = 0.0
    p_L   = 0.125 / gamma

    rho_R = 1.0
    u_R   = 0.0
    p_R   = 1.0 / gamma

    p_star, u_star = star_pressure_velocity(rho_L, u_L, p_L, rho_R, u_R, p_R, gamma)
    rho_ex, u_ex, p_ex = exact_solution_sod(
        x, t_end, x0, rho_L, u_L, p_L, rho_R, u_R, p_R, gamma, p_star, u_star
    )

    L1_rho   = torch.mean(torch.abs(rho_tot - rho_ex)).item()
    Linf_rho = torch.max(torch.abs(rho_tot - rho_ex)).item()

    L1_u     = torch.mean(torch.abs(u_tot - u_ex)).item()
    Linf_u   = torch.max(torch.abs(u_tot - u_ex)).item()

    L1_p     = torch.mean(torch.abs(p_tot - p_ex)).item()
    Linf_p   = torch.max(torch.abs(p_tot - p_ex)).item()

    mean_drift_final = torch.mean(torch.abs(u_i - u_n)).item()
    max_drift_final  = torch.max(torch.abs(u_i - u_n)).item()

    result = {
        "x": x,
        "rho_i": rho_i, "u_i": u_i, "p_i": p_i,
        "rho_n": rho_n, "u_n": u_n, "p_n": p_n,
        "rho_tot": rho_tot, "u_tot": u_tot, "p_tot": p_tot,
        "rho_ex": rho_ex, "u_ex": u_ex, "p_ex": p_ex,
        "t_hist": t_hist, "drift_hist": drift_hist,
        "errors": {
            "L1_rho": L1_rho, "Linf_rho": Linf_rho,
            "L1_u": L1_u, "Linf_u": Linf_u,
            "L1_p": L1_p, "Linf_p": Linf_p,
        },
        "diagnostics": {
            "mean_drift_final": mean_drift_final,
            "max_drift_final": max_drift_final,
        },
        "nu_in": nu_in,
        "final_time": t,
    }

    return result


# Run Test 2


nu_list = [0.0, 0.01, 0.1, 0.5, 1.0, 5.0]
results = []

for nu in nu_list:
    print(f"\nRunning case nu_in = {nu}")

    result = run_two_fluid_sod_case(
        nu_in=nu,
        ion_frac_L=0.8,
        ion_frac_R=0.2,
        drift_n_L=0.0,
        drift_n_R=0.0
    )

    results.append(result)

    err = result["errors"]
    diag = result["diagnostics"]

    print("L1_rho   =", err["L1_rho"])
    print("Linf_rho =", err["Linf_rho"])
    print("L1_u     =", err["L1_u"])
    print("Linf_u   =", err["Linf_u"])
    print("L1_p     =", err["L1_p"])
    print("Linf_p   =", err["Linf_p"])
    print("mean_drift_final =", diag["mean_drift_final"])
    print("max_drift_final  =", diag["max_drift_final"])


rows = []
for r in results:
    rows.append({
        "nu_in": r["nu_in"],
        "L1_rho": r["errors"]["L1_rho"],
        "Linf_rho": r["errors"]["Linf_rho"],
        "L1_u": r["errors"]["L1_u"],
        "Linf_u": r["errors"]["Linf_u"],
        "L1_p": r["errors"]["L1_p"],
        "Linf_p": r["errors"]["Linf_p"],
        "mean_drift_final": r["diagnostics"]["mean_drift_final"],
        "max_drift_final": r["diagnostics"]["max_drift_final"]
    })

df_test2 = pd.DataFrame(rows)
print("\nTest 2 summary table:")
print(df_test2)



# Plots


# representative cases
selected_nu = [0.0, 0.1, 5.0]
selected_results = [r for r in results if r["nu_in"] in selected_nu]

x_exact = results[0]["x"].detach().cpu().numpy()
rho_exact = results[0]["rho_ex"].detach().cpu().numpy()
u_exact = results[0]["u_ex"].detach().cpu().numpy()
p_exact = results[0]["p_ex"].detach().cpu().numpy()

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# density
for r in selected_results:
    x_np = r["x"].detach().cpu().numpy()
    rho_tot_np = r["rho_tot"].detach().cpu().numpy()
    axes[0, 0].plot(x_np, rho_tot_np, label=f'nu={r["nu_in"]}')
axes[0, 0].plot(x_exact, rho_exact, 'k--', linewidth=2, label='exact')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('rho_total')
axes[0, 0].set_title('Total density')
axes[0, 0].legend()
axes[0, 0].grid()

# velocity
for r in selected_results:
    x_np = r["x"].detach().cpu().numpy()
    u_tot_np = r["u_tot"].detach().cpu().numpy()
    axes[0, 1].plot(x_np, u_tot_np, label=f'nu={r["nu_in"]}')
axes[0, 1].plot(x_exact, u_exact, 'k--', linewidth=2, label='exact')
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('u_total')
axes[0, 1].set_title('Total velocity')
axes[0, 1].legend()
axes[0, 1].grid()

# pressure
for r in selected_results:
    x_np = r["x"].detach().cpu().numpy()
    p_tot_np = r["p_tot"].detach().cpu().numpy()
    axes[1, 0].plot(x_np, p_tot_np, label=f'nu={r["nu_in"]}')
axes[1, 0].plot(x_exact, p_exact, 'k--', linewidth=2, label='exact')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('p_total')
axes[1, 0].set_title('Total pressure')
axes[1, 0].legend()
axes[1, 0].grid()

# drift history
for r in selected_results:
    axes[1, 1].plot(r["t_hist"], r["drift_hist"], label=f'nu={r["nu_in"]}')
axes[1, 1].set_xlabel('t')
axes[1, 1].set_ylabel('mean |u_i - u_n|')
axes[1, 1].set_title('Drift history')
axes[1, 1].legend()
axes[1, 1].grid()

fig.suptitle('Representative two-fluid cases', fontsize=14)

save_figure(fig, "03_representative_cases_test2.png")
plt.show()
plt.close(fig)


# log plots: remove nu=0
df_test2_pos = df_test2[df_test2["nu_in"] > 0].copy()

df_test2_pos = df_test2[df_test2["nu_in"] > 0].copy()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(df_test2_pos["nu_in"], df_test2_pos["L1_rho"], marker='o', label='L1 rho')
axes[0].plot(df_test2_pos["nu_in"], df_test2_pos["L1_u"], marker='s', label='L1 u')
axes[0].plot(df_test2_pos["nu_in"], df_test2_pos["L1_p"], marker='^', label='L1 p')
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].set_xlabel('nu_in')
axes[0].set_ylabel('L1 error')
axes[0].set_title('Error vs collision frequency')
axes[0].legend()
axes[0].grid()

axes[1].plot(df_test2_pos["nu_in"], df_test2_pos["mean_drift_final"], marker='o', label='mean drift final')
axes[1].plot(df_test2_pos["nu_in"], df_test2_pos["max_drift_final"], marker='s', label='max drift final')
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].set_xlabel('nu_in')
axes[1].set_ylabel('final drift')
axes[1].set_title('Final drift vs collision frequency')
axes[1].legend()
axes[1].grid()

fig.suptitle('Test 2 summary plots', fontsize=14)

save_figure(fig, "04_test2_error_and_drift_vs_nu.png")
plt.show()
plt.close(fig)


# Step 1: Conservative Euler variables and Rusanov flux
# This part does not replace the previous code.
# It is added as a new finite-volume shock-capturing baseline.


def prim_to_state_E(rho, u, p, gamma=gamma):
    """
    Primitive variables -> conservative variables.

    Input:
        rho : density
        u   : velocity
        p   : pressure

    Output:
        U = (rho, m, E)
        where m = rho*u
        and E = p/(gamma-1) + 0.5*rho*u^2
    """
    m = rho * u
    E = p / (gamma - 1.0) + 0.5 * rho * u**2
    U = torch.stack([rho, m, E], dim=0)
    return U


def state_to_prim_E(U, gamma=gamma):
    """
    Conservative variables -> primitive variables.

    U = (rho, m, E)
    """
    rho = torch.clamp(U[0], min=eps)
    m   = U[1]
    E   = torch.clamp(U[2], min=eps)

    u = m / rho
    kinetic = 0.5 * rho * u**2
    p = (gamma - 1.0) * torch.clamp(E - kinetic, min=eps)

    return rho, u, p


def flux_E(U, gamma=gamma):
    """
    Conservative Euler flux.

    U_t + F(U)_x = 0

    F(U) = (rho*u, rho*u^2 + p, u*(E+p))
    """
    rho, u, p = state_to_prim_E(U, gamma=gamma)

    m = U[1]
    E = U[2]

    F_rho = m
    F_m   = m * u + p
    F_E   = u * (E + p)

    F = torch.stack([F_rho, F_m, F_E], dim=0)
    return F


def max_wave_speed_E(U, gamma=gamma):
    """
    Compute max local wave speed |u| + c.
    """
    rho, u, p = state_to_prim_E(U, gamma=gamma)
    c = torch.sqrt(gamma * p / rho)
    return torch.abs(u) + c


def rusanov_flux_E(U_L, U_R, gamma=gamma):
    """
    Rusanov numerical flux at cell interfaces.

    F_{j+1/2} = 0.5*(F_L + F_R) - 0.5*a*(U_R - U_L)

    where a = max(|u_L|+c_L, |u_R|+c_R)
    """
    F_L = flux_E(U_L, gamma=gamma)
    F_R = flux_E(U_R, gamma=gamma)

    a_L = max_wave_speed_E(U_L, gamma=gamma)
    a_R = max_wave_speed_E(U_R, gamma=gamma)

    a = torch.maximum(a_L, a_R)

    F_half = 0.5 * (F_L + F_R) - 0.5 * a.unsqueeze(0) * (U_R - U_L)

    return F_half


def apply_fixed_bc_one_fluid_E(U, U_L, U_R):
    """
    Fixed boundary condition for one fluid.
    """
    U[:, 0]  = U_L
    U[:, -1] = U_R
    return U


def rusanov_step_one_fluid_E(U, dx, dt, U_L, U_R, gamma=gamma):
    """
    One finite-volume Rusanov step for one fluid.

    This is first-order in space.
    It is more robust for shocks than centered difference.
    """
    U = apply_fixed_bc_one_fluid_E(U, U_L, U_R)

    # left and right states at interfaces
    U_left  = U[:, :-1]
    U_right = U[:, 1:]

    # numerical flux at interfaces j+1/2
    F_half = rusanov_flux_E(U_left, U_right, gamma=gamma)

    U_new = U.clone()

    # finite-volume update for interior cells
    U_new[:, 1:-1] = U[:, 1:-1] - dt / dx * (F_half[:, 1:] - F_half[:, :-1])

    # positivity protection
    U_new[0] = torch.clamp(U_new[0], min=eps)

    rho_new, u_new, p_new = state_to_prim_E(U_new, gamma=gamma)
    E_internal = p_new / (gamma - 1.0)
    E_kinetic  = 0.5 * rho_new * u_new**2
    U_new[2] = torch.clamp(E_internal + E_kinetic, min=eps)

    U_new = apply_fixed_bc_one_fluid_E(U_new, U_L, U_R)

    return U_new


def compute_dt_one_fluid_E(U, dx, gamma=gamma, cfl=0.5):
    """
    CFL time step for conservative Euler system.
    """
    speed = max_wave_speed_E(U, gamma=gamma)
    max_speed = torch.max(speed)
    dt = cfl * dx / max_speed
    return dt



# Debug for the new conservative variables

rho_test = torch.tensor([1.0, 0.125], dtype=dtype, device=device)
u_test   = torch.tensor([0.0, 0.0], dtype=dtype, device=device)
p_test   = torch.tensor([1.0, 0.1], dtype=dtype, device=device)

U_test = prim_to_state_E(rho_test, u_test, p_test)

rho_back, u_back, p_back = state_to_prim_E(U_test)

print("U_test =", U_test)
print("rho_back =", rho_back)
print("u_back =", u_back)
print("p_back =", p_back)



# Step 2: Two-fluid Rusanov + collision source splitting
# This does not replace the old code.
# It adds a conservative finite-volume two-fluid baseline.

def make_two_fluid_state_E(rho_i, u_i, p_i, rho_n, u_n, p_n, gamma=gamma):
    """
    Build two-fluid conservative state.

    Ion fluid:     U_i = (rho_i, m_i, E_i)
    Neutral fluid: U_n = (rho_n, m_n, E_n)
    """
    U_i = prim_to_state_E(rho_i, u_i, p_i, gamma=gamma)
    U_n = prim_to_state_E(rho_n, u_n, p_n, gamma=gamma)
    return U_i, U_n


def state_to_prim_two_fluid_E(U_i, U_n, gamma=gamma):
    """
    Convert two-fluid conservative variables to primitive variables.
    """
    rho_i, u_i, p_i = state_to_prim_E(U_i, gamma=gamma)
    rho_n, u_n, p_n = state_to_prim_E(U_n, gamma=gamma)
    return rho_i, u_i, p_i, rho_n, u_n, p_n


def compute_dt_two_fluid_E(U_i, U_n, dx, gamma=gamma, cfl=0.5):
    """
    Global CFL time step for the two-fluid system.

    For now we still use the minimum of ion and neutral dt.
    Later, this is exactly the part we will improve for the ADI/per-fluid method.
    """
    dt_i = compute_dt_one_fluid_E(U_i, dx, gamma=gamma, cfl=cfl)
    dt_n = compute_dt_one_fluid_E(U_n, dx, gamma=gamma, cfl=cfl)

    return torch.minimum(dt_i, dt_n)


def collision_source_exact_E(U_i, U_n, dt, nu_in, gamma=gamma):
    """
    Exact momentum relaxation step for the collision source.

    We solve:
        du_i/dt = nu_in * (u_n - u_i)
        du_n/dt = nu_ni * (u_i - u_n)

    with momentum conservation:
        rho_i * nu_in = rho_n * nu_ni

    The relative velocity satisfies:
        w' = -(nu_in + nu_ni) w

    Here we ignore heating, so pressure/internal energy is kept fixed
    during this source step.
    """
    rho_i, u_i, p_i = state_to_prim_E(U_i, gamma=gamma)
    rho_n, u_n, p_n = state_to_prim_E(U_n, gamma=gamma)

    rho_i = torch.clamp(rho_i, min=eps)
    rho_n = torch.clamp(rho_n, min=eps)

    # pointwise nu_ni from momentum conservation
    nu_ni = rho_i * nu_in / rho_n

    # total density and barycentric velocity
    rho_tot = rho_i + rho_n
    u_bar = (rho_i * u_i + rho_n * u_n) / torch.clamp(rho_tot, min=eps)

    # relative velocity
    w_old = u_i - u_n
    decay = torch.exp(-(nu_in + nu_ni) * dt)
    w_new = w_old * decay

    # reconstruct velocities while preserving total momentum
    u_i_new = u_bar + rho_n / torch.clamp(rho_tot, min=eps) * w_new
    u_n_new = u_bar - rho_i / torch.clamp(rho_tot, min=eps) * w_new

    # ignore heating: keep pressure fixed during collision step
    U_i_new = prim_to_state_E(rho_i, u_i_new, p_i, gamma=gamma)
    U_n_new = prim_to_state_E(rho_n, u_n_new, p_n, gamma=gamma)

    return U_i_new, U_n_new


def rusanov_step_two_fluid_OS_E(U_i, U_n, dx, dt,
                                U_i_L, U_i_R, U_n_L, U_n_R,
                                nu_in,
                                gamma=gamma):
    """
    One operator-splitting step for the two-fluid system.

    Step A: evolve ion fluid by Rusanov hydro step.
    Step B: evolve neutral fluid by Rusanov hydro step.
    Step C: apply exact collision momentum relaxation.

    This is not the final ADI method yet.
    It is the clean two-fluid Rusanov + source-splitting baseline.
    """

    # Hydro step for each fluid separately
    U_i_star = rusanov_step_one_fluid_E(U_i, dx, dt, U_i_L, U_i_R, gamma=gamma)
    U_n_star = rusanov_step_one_fluid_E(U_n, dx, dt, U_n_L, U_n_R, gamma=gamma)

    # Collision/source step
    U_i_new, U_n_new = collision_source_exact_E(U_i_star, U_n_star, dt, nu_in, gamma=gamma)

    # Apply fixed boundary conditions
    U_i_new = apply_fixed_bc_one_fluid_E(U_i_new, U_i_L, U_i_R)
    U_n_new = apply_fixed_bc_one_fluid_E(U_n_new, U_n_L, U_n_R)

    return U_i_new, U_n_new


# Debug test for Step 2: drift relaxation with Rusanov + source


N = 200
x = torch.linspace(-1.0, 1.0, N, dtype=dtype, device=device)
dx = x[1] - x[0]

rho_i0 = 0.2
rho_n0 = 0.8
u_i0   = 1.0
u_n0   = 0.0
p_i0   = 0.1
p_n0   = 0.1

rho_i = rho_i0 * torch.ones_like(x)
rho_n = rho_n0 * torch.ones_like(x)

u_i = u_i0 * torch.ones_like(x)
u_n = u_n0 * torch.ones_like(x)

p_i = p_i0 * torch.ones_like(x)
p_n = p_n0 * torch.ones_like(x)

U_i, U_n = make_two_fluid_state_E(rho_i, u_i, p_i, rho_n, u_n, p_n, gamma=gamma)

# fixed boundary states
U_i_L = U_i[:, 0].clone()
U_i_R = U_i[:, -1].clone()
U_n_L = U_n[:, 0].clone()
U_n_R = U_n[:, -1].clone()

nu_in = 1.0
nu_ni = rho_i0 * nu_in / rho_n0
lambda_exact = nu_in + nu_ni

print("nu_in =", nu_in)
print("nu_ni =", nu_ni)
print("lambda_exact =", lambda_exact)

t = 0.0
t_end = 0.5

t_hist = []
drift_hist = []
exact_hist = []

while t < t_end:
    rho_i_now, u_i_now, p_i_now, rho_n_now, u_n_now, p_n_now = state_to_prim_two_fluid_E(U_i, U_n)

    drift = torch.mean(torch.abs(u_i_now - u_n_now)).item()

    t_hist.append(t)
    drift_hist.append(drift)
    exact_hist.append(abs(u_i0 - u_n0) * math.exp(-lambda_exact * t))

    dt = compute_dt_two_fluid_E(U_i, U_n, dx, gamma=gamma, cfl=0.5).item()

    if t + dt > t_end:
        dt = t_end - t

    U_i, U_n = rusanov_step_two_fluid_OS_E(
        U_i, U_n, dx, dt,
        U_i_L, U_i_R, U_n_L, U_n_R,
        nu_in=nu_in,
        gamma=gamma
    )

    t += dt

rho_i_final, u_i_final, p_i_final, rho_n_final, u_n_final, p_n_final = state_to_prim_two_fluid_E(U_i, U_n)

print("final mean u_i =", torch.mean(u_i_final).item())
print("final mean u_n =", torch.mean(u_n_final).item())
print("final mean drift =", torch.mean(torch.abs(u_i_final - u_n_final)).item())

plt.figure(figsize=(8,5))
plt.semilogy(t_hist, drift_hist, label="numerical drift")
plt.semilogy(t_hist, exact_hist, "--", label="exact exponential")
plt.xlabel("t")
plt.ylabel("mean |u_i - u_n|")
plt.title("Step 2 test: drift relaxation with source splitting")
plt.grid()
plt.legend()
plt.close()


# Step 3A: Two-fluid Sod initial condition for conservative variables


def initial_condition_two_fluid_sod_E(x, x0=0.0, gamma=gamma,
                                      ion_frac_L=0.5, ion_frac_R=0.5,
                                      press_frac_i=0.5,
                                      drift_n_L=0.0, drift_n_R=0.0):
    """
    Two-fluid Sod shock tube initial condition.

    Total left/right states:
        left:  rho = 0.125, p = 0.125/gamma
        right: rho = 1.0,   p = 1.0/gamma

    Then split density and pressure into ion and neutral parts.
    """

    left_mask = x < x0

    rho_tot = torch.where(
        left_mask,
        torch.full_like(x, 0.125),
        torch.full_like(x, 1.0)
    )

    p_tot = torch.where(
        left_mask,
        torch.full_like(x, 0.125 / gamma),
        torch.full_like(x, 1.0 / gamma)
    )

    ion_frac = torch.where(
        left_mask,
        torch.full_like(x, ion_frac_L),
        torch.full_like(x, ion_frac_R)
    )

    rho_i = ion_frac * rho_tot
    rho_n = (1.0 - ion_frac) * rho_tot

    p_i = press_frac_i * p_tot
    p_n = (1.0 - press_frac_i) * p_tot

    u_i0 = torch.zeros_like(x)

    u_n0 = torch.where(
        left_mask,
        torch.full_like(x, drift_n_L),
        torch.full_like(x, drift_n_R)
    )

    U_i, U_n = make_two_fluid_state_E(
        rho_i, u_i0, p_i,
        rho_n, u_n0, p_n,
        gamma=gamma
    )

    return U_i, U_n


# Step 3B: Run two-fluid Sod with Rusanov + source splitting


def run_two_fluid_sod_case_OS_E(nu_in,
                                x_min=-5.0, x_max=5.0, N=1000,
                                x0=0.0, t_end=0.08,
                                ion_frac_L=0.5, ion_frac_R=0.5,
                                press_frac_i=0.5,
                                drift_n_L=0.0, drift_n_R=0.0,
                                cfl=0.4,
                                gamma=gamma,
                                max_steps=100000):
    """
    Run a two-fluid Sod shock tube using:

        1. Rusanov finite-volume hydro step for ions
        2. Rusanov finite-volume hydro step for neutrals
        3. exact collision/source relaxation step

    This is still not the final ADI method.
    This is the clean Rusanov + operator splitting baseline.
    """

    x = torch.linspace(x_min, x_max, N, dtype=dtype, device=device)
    dx = x[1] - x[0]

    U_i, U_n = initial_condition_two_fluid_sod_E(
        x, x0=x0, gamma=gamma,
        ion_frac_L=ion_frac_L,
        ion_frac_R=ion_frac_R,
        press_frac_i=press_frac_i,
        drift_n_L=drift_n_L,
        drift_n_R=drift_n_R
    )

    # fixed boundary states
    U_i_L = U_i[:, 0].clone()
    U_i_R = U_i[:, -1].clone()
    U_n_L = U_n[:, 0].clone()
    U_n_R = U_n[:, -1].clone()

    t = 0.0
    step = 0

    t_hist = []
    drift_hist = []
    dt_hist = []

    while t < t_end and step < max_steps:
        U_i = apply_fixed_bc_one_fluid_E(U_i, U_i_L, U_i_R)
        U_n = apply_fixed_bc_one_fluid_E(U_n, U_n_L, U_n_R)

        rho_i_now, u_i_now, p_i_now, rho_n_now, u_n_now, p_n_now = state_to_prim_two_fluid_E(
            U_i, U_n, gamma=gamma
        )

        drift_now = torch.mean(torch.abs(u_i_now - u_n_now)).item()

        t_hist.append(t)
        drift_hist.append(drift_now)

        dt = compute_dt_two_fluid_E(U_i, U_n, dx, gamma=gamma, cfl=cfl).item()

        if t + dt > t_end:
            dt = t_end - t

        if dt <= 0:
            print("dt <= 0")
            break

        dt_hist.append(dt)

        U_i, U_n = rusanov_step_two_fluid_OS_E(
            U_i, U_n, dx, dt,
            U_i_L, U_i_R,
            U_n_L, U_n_R,
            nu_in=nu_in,
            gamma=gamma
        )

        t += dt
        step += 1

    if step >= max_steps:
        print("Warning: max_steps reached")

    rho_i, u_i, p_i, rho_n, u_n, p_n = state_to_prim_two_fluid_E(
        U_i, U_n, gamma=gamma
    )

    rho_tot = rho_i + rho_n
    m_tot = U_i[1] + U_n[1]
    u_tot = m_tot / torch.clamp(rho_tot, min=eps)
    p_tot = p_i + p_n

    result = {
        "x": x,
        "U_i": U_i,
        "U_n": U_n,

        "rho_i": rho_i,
        "u_i": u_i,
        "p_i": p_i,

        "rho_n": rho_n,
        "u_n": u_n,
        "p_n": p_n,

        "rho_tot": rho_tot,
        "u_tot": u_tot,
        "p_tot": p_tot,

        "t_hist": t_hist,
        "drift_hist": drift_hist,
        "dt_hist": dt_hist,

        "nu_in": nu_in,
        "final_time": t,
        "steps": step,

        "diagnostics": {
            "min_rho_i": torch.min(rho_i).item(),
            "min_rho_n": torch.min(rho_n).item(),
            "min_p_i": torch.min(p_i).item(),
            "min_p_n": torch.min(p_n).item(),
            "mean_drift_final": torch.mean(torch.abs(u_i - u_n)).item(),
            "max_drift_final": torch.max(torch.abs(u_i - u_n)).item(),
        }
    }

    return result


# Step 3C: Plot two-fluid Sod result


def plot_two_fluid_sod_OS_E(result):
    x_np = result["x"].detach().cpu().numpy()

    rho_i_np = result["rho_i"].detach().cpu().numpy()
    rho_n_np = result["rho_n"].detach().cpu().numpy()
    rho_tot_np = result["rho_tot"].detach().cpu().numpy()

    u_i_np = result["u_i"].detach().cpu().numpy()
    u_n_np = result["u_n"].detach().cpu().numpy()
    u_tot_np = result["u_tot"].detach().cpu().numpy()

    p_i_np = result["p_i"].detach().cpu().numpy()
    p_n_np = result["p_n"].detach().cpu().numpy()
    p_tot_np = result["p_tot"].detach().cpu().numpy()

    nu_in = result["nu_in"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].plot(x_np, rho_i_np, label="rho_i")
    axes[0, 0].plot(x_np, rho_n_np, label="rho_n")
    axes[0, 0].plot(x_np, rho_tot_np, linewidth=2, label="rho_total")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("density")
    axes[0, 0].set_title("Density")
    axes[0, 0].grid()
    axes[0, 0].legend()

    axes[0, 1].plot(x_np, u_i_np, label="u_i")
    axes[0, 1].plot(x_np, u_n_np, label="u_n")
    axes[0, 1].plot(x_np, u_tot_np, linewidth=2, label="u_total")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("velocity")
    axes[0, 1].set_title("Velocity")
    axes[0, 1].grid()
    axes[0, 1].legend()

    axes[1, 0].plot(x_np, p_i_np, label="p_i")
    axes[1, 0].plot(x_np, p_n_np, label="p_n")
    axes[1, 0].plot(x_np, p_tot_np, linewidth=2, label="p_total")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("pressure")
    axes[1, 0].set_title("Pressure")
    axes[1, 0].grid()
    axes[1, 0].legend()

    axes[1, 1].plot(result["t_hist"], result["drift_hist"])
    axes[1, 1].set_xlabel("t")
    axes[1, 1].set_ylabel("mean |u_i - u_n|")
    axes[1, 1].set_title("Drift history")
    axes[1, 1].grid()

    fig.suptitle(f"Two-fluid Sod OS result, nu_in = {nu_in}", fontsize=14)

    save_figure(fig, f"05_OS_two_fluid_sod_nu_{nu_in}.png")
    plt.show()
    plt.close(fig)


# Step 3D: Run one test case


result_os = run_two_fluid_sod_case_OS_E(
    nu_in=0.5,
    N=1000,
    t_end=0.08,
    ion_frac_L=0.5,
    ion_frac_R=0.5,
    press_frac_i=0.5,
    drift_n_L=0.0,
    drift_n_R=0.0,
    cfl=0.4
)

print("final time =", result_os["final_time"])
print("number of steps =", result_os["steps"])
print("diagnostics:")
print(result_os["diagnostics"])

plot_two_fluid_sod_OS_E(result_os)


# Step 4A: Run several collision frequencies


nu_list_OS = [0.0, 0.01, 0.1, 0.5, 1.0, 5.0, 20.0]

results_OS = []

for nu in nu_list_OS:
    print(f"\nRunning OS Rusanov case: nu_in = {nu}")

    result = run_two_fluid_sod_case_OS_E(
        nu_in=nu,
        N=1000,
        t_end=0.08,
        ion_frac_L=0.8, # Use some different, this is good.
        ion_frac_R=0.2,
        press_frac_i=0.5,
        drift_n_L=0.2,
        drift_n_R=0.0,
        cfl=0.4
    )

    results_OS.append(result)

    print("final time =", result["final_time"])
    print("steps =", result["steps"])
    print("mean drift final =", result["diagnostics"]["mean_drift_final"])
    print("max drift final  =", result["diagnostics"]["max_drift_final"])
    print("min rho_i =", result["diagnostics"]["min_rho_i"])
    print("min rho_n =", result["diagnostics"]["min_rho_n"])
    print("min p_i   =", result["diagnostics"]["min_p_i"])
    print("min p_n   =", result["diagnostics"]["min_p_n"])



# Step 4B: Summary table


rows_OS = []

for r in results_OS:
    rows_OS.append({
        "nu_in": r["nu_in"],
        "steps": r["steps"],
        "final_time": r["final_time"],
        "mean_drift_final": r["diagnostics"]["mean_drift_final"],
        "max_drift_final": r["diagnostics"]["max_drift_final"],
        "min_rho_i": r["diagnostics"]["min_rho_i"],
        "min_rho_n": r["diagnostics"]["min_rho_n"],
        "min_p_i": r["diagnostics"]["min_p_i"],
        "min_p_n": r["diagnostics"]["min_p_n"],
    })

df_OS = pd.DataFrame(rows_OS)

print("Step 4 summary table:")
print(df_OS)



# Step 4C: Final drift versus collision frequency


df_OS_pos = df_OS[df_OS["nu_in"] > 0].copy()

plt.figure(figsize=(8,5))
plt.plot(df_OS_pos["nu_in"], df_OS_pos["mean_drift_final"], marker="o", label="mean final drift")
plt.plot(df_OS_pos["nu_in"], df_OS_pos["max_drift_final"], marker="s", label="max final drift")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("nu_in")
plt.ylabel("final drift")
plt.title("Final velocity drift vs collision frequency")
plt.grid()
plt.legend()
plt.close()


# Step 4D: Compare representative cases

selected_nu_OS = [0.0, 0.5, 20.0]
selected_results_OS = [r for r in results_OS if r["nu_in"] in selected_nu_OS]

plt.figure(figsize=(8,5))
for r in selected_results_OS:
    x_np = r["x"].detach().cpu().numpy()
    rho_tot_np = r["rho_tot"].detach().cpu().numpy()
    plt.plot(x_np, rho_tot_np, label=f'nu_in={r["nu_in"]}')

plt.xlabel("x")
plt.ylabel("rho_total")
plt.title("Total density for different collision frequencies")
plt.grid()
plt.legend()
plt.close()


plt.figure(figsize=(8,5))
for r in selected_results_OS:
    x_np = r["x"].detach().cpu().numpy()
    u_tot_np = r["u_tot"].detach().cpu().numpy()
    plt.plot(x_np, u_tot_np, label=f'nu_in={r["nu_in"]}')

plt.xlabel("x")
plt.ylabel("u_total")
plt.title("Total velocity for different collision frequencies")
plt.grid()
plt.legend()
plt.close()


plt.figure(figsize=(8,5))
for r in selected_results_OS:
    x_np = r["x"].detach().cpu().numpy()
    p_tot_np = r["p_tot"].detach().cpu().numpy()
    plt.plot(x_np, p_tot_np, label=f'nu_in={r["nu_in"]}')

plt.xlabel("x")
plt.ylabel("p_total")
plt.title("Total pressure for different collision frequencies")
plt.grid()
plt.legend()
plt.show()


# Step 4E: Drift history for representative cases


plt.figure(figsize=(8,5))

for r in selected_results_OS:
    plt.plot(r["t_hist"], r["drift_hist"], label=f'nu_in={r["nu_in"]}')

plt.xlabel("t")
plt.ylabel("mean |u_i - u_n|")
plt.title("Velocity drift history")
plt.grid()
plt.legend()
plt.close()


# Step 5A: Single-fluid exact Sod solution for comparison
# This is used only for diagnostics of the total variables.


def pressure_function_new(p, rho_k, u_k, p_k, gamma):
    a_k = math.sqrt(gamma * p_k / rho_k)
    A_k = 2.0 / ((gamma + 1.0) * rho_k)
    B_k = (gamma - 1.0) / (gamma + 1.0) * p_k

    if p > p_k:
        # shock
        f = (p - p_k) * math.sqrt(A_k / (p + B_k))
        fd = math.sqrt(A_k / (p + B_k)) * (
            1.0 - 0.5 * (p - p_k) / (p + B_k)
        )
    else:
        # rarefaction
        pr = p / p_k
        f = (2.0 * a_k / (gamma - 1.0)) * (
            pr**((gamma - 1.0) / (2.0 * gamma)) - 1.0
        )
        fd = (1.0 / (rho_k * a_k)) * pr**(
            -(gamma + 1.0) / (2.0 * gamma)
        )

    return f, fd


def star_pressure_velocity_new(rho_L, u_L, p_L,
                               rho_R, u_R, p_R,
                               gamma,
                               tol=1e-10,
                               max_iter=100):
    a_L = math.sqrt(gamma * p_L / rho_L)
    a_R = math.sqrt(gamma * p_R / rho_R)

    p_guess = 0.5 * (p_L + p_R) - 0.125 * (u_R - u_L) * (
        rho_L + rho_R
    ) * (a_L + a_R)

    p = max(1e-8, p_guess)

    for _ in range(max_iter):
        fL, fdL = pressure_function_new(p, rho_L, u_L, p_L, gamma)
        fR, fdR = pressure_function_new(p, rho_R, u_R, p_R, gamma)

        p_new = p - (fL + fR + u_R - u_L) / (fdL + fdR)
        p_new = max(1e-8, p_new)

        if abs(p_new - p) / (0.5 * (p_new + p)) < tol:
            p = p_new
            break

        p = p_new

    fL, _ = pressure_function_new(p, rho_L, u_L, p_L, gamma)
    fR, _ = pressure_function_new(p, rho_R, u_R, p_R, gamma)

    u_star = 0.5 * (u_L + u_R + fR - fL)

    return p, u_star


def exact_solution_sod_new(x, t, x0,
                           rho_L, u_L, p_L,
                           rho_R, u_R, p_R,
                           gamma):
    """
    Exact Sod solution for the case used in this project:
        left state:  low density / low pressure
        right state: high density / high pressure

    This gives a left-moving shock and right-moving rarefaction.
    """

    p_star, u_star = star_pressure_velocity_new(
        rho_L, u_L, p_L,
        rho_R, u_R, p_R,
        gamma
    )

    rho_ex = torch.zeros_like(x)
    u_ex = torch.zeros_like(x)
    p_ex = torch.zeros_like(x)

    a_L = math.sqrt(gamma * p_L / rho_L)
    a_R = math.sqrt(gamma * p_R / rho_R)

    rho_star_L = rho_L * (
        (p_star / p_L + (gamma - 1.0) / (gamma + 1.0)) /
        (((gamma - 1.0) / (gamma + 1.0)) * (p_star / p_L) + 1.0)
    )

    rho_star_R = rho_R * (p_star / p_R)**(1.0 / gamma)

    xi = (x - x0) / t

    # left shock
    S_L = u_L - a_L * math.sqrt(
        (gamma + 1.0) / (2.0 * gamma) * (p_star / p_L)
        + (gamma - 1.0) / (2.0 * gamma)
    )

    S_contact = u_star

    # right rarefaction
    a_star_R = a_R * (p_star / p_R)**((gamma - 1.0) / (2.0 * gamma))
    S_tail_R = u_star + a_star_R
    S_head_R = u_R + a_R

    # left state
    mask1 = xi <= S_L
    rho_ex[mask1] = rho_L
    u_ex[mask1] = u_L
    p_ex[mask1] = p_L

    # left star state
    mask2 = (xi > S_L) & (xi <= S_contact)
    rho_ex[mask2] = rho_star_L
    u_ex[mask2] = u_star
    p_ex[mask2] = p_star

    # right star state
    mask3 = (xi > S_contact) & (xi <= S_tail_R)
    rho_ex[mask3] = rho_star_R
    u_ex[mask3] = u_star
    p_ex[mask3] = p_star

    # right rarefaction fan
    mask4 = (xi > S_tail_R) & (xi < S_head_R)
    xi_fan = xi[mask4]

    u_fan = 2.0 / (gamma + 1.0) * (
        -a_R + 0.5 * (gamma - 1.0) * u_R + xi_fan
    )

    a_fan = 2.0 / (gamma + 1.0) * (
        a_R - 0.5 * (gamma - 1.0) * (u_R - xi_fan)
    )

    rho_fan = rho_R * (a_fan / a_R)**(2.0 / (gamma - 1.0))
    p_fan = p_R * (a_fan / a_R)**(2.0 * gamma / (gamma - 1.0))

    rho_ex[mask4] = rho_fan
    u_ex[mask4] = u_fan
    p_ex[mask4] = p_fan

    # right state
    mask5 = xi >= S_head_R
    rho_ex[mask5] = rho_R
    u_ex[mask5] = u_R
    p_ex[mask5] = p_R

    wave_speeds = {
        "S_L": S_L,
        "S_contact": S_contact,
        "S_tail_R": S_tail_R,
        "S_head_R": S_head_R,
        "p_star": p_star,
        "u_star": u_star,
    }

    return rho_ex, u_ex, p_ex, wave_speeds



# Step 5B: Add exact comparison and error computation


def add_exact_comparison_to_result(result, x0=0.0, t_end=0.08, gamma=gamma):
    """
    Add single-fluid exact solution and error diagnostics to a result dictionary.
    """

    x = result["x"]

    rho_L = 0.125
    u_L = 0.0
    p_L = 0.125 / gamma

    rho_R = 1.0
    u_R = 0.0
    p_R = 1.0 / gamma

    rho_ex, u_ex, p_ex, wave_speeds = exact_solution_sod_new(
        x, t_end, x0,
        rho_L, u_L, p_L,
        rho_R, u_R, p_R,
        gamma
    )

    rho_tot = result["rho_tot"]
    u_tot = result["u_tot"]
    p_tot = result["p_tot"]

    errors = {
        "L1_rho": torch.mean(torch.abs(rho_tot - rho_ex)).item(),
        "Linf_rho": torch.max(torch.abs(rho_tot - rho_ex)).item(),

        "L1_u": torch.mean(torch.abs(u_tot - u_ex)).item(),
        "Linf_u": torch.max(torch.abs(u_tot - u_ex)).item(),

        "L1_p": torch.mean(torch.abs(p_tot - p_ex)).item(),
        "Linf_p": torch.max(torch.abs(p_tot - p_ex)).item(),
    }

    result["rho_ex"] = rho_ex
    result["u_ex"] = u_ex
    result["p_ex"] = p_ex
    result["wave_speeds"] = wave_speeds
    result["errors_exact"] = errors

    return result




# Step 5C: Controlled comparison with single-fluid exact solution


result_exact_test = run_two_fluid_sod_case_OS_E(
    nu_in=0.5,
    N=1000,
    t_end=0.08,
    ion_frac_L=0.5,
    ion_frac_R=0.5,
    press_frac_i=0.5,
    drift_n_L=0.0,
    drift_n_R=0.0,
    cfl=0.4
)

result_exact_test = add_exact_comparison_to_result(
    result_exact_test,
    x0=0.0,
    t_end=0.08,
    gamma=gamma
)

print("final time =", result_exact_test["final_time"])
print("steps =", result_exact_test["steps"])

print("\nWave speeds:")
print(result_exact_test["wave_speeds"])

print("\nErrors compared with single-fluid exact solution:")
print(result_exact_test["errors_exact"])

print("\nDiagnostics:")
print(result_exact_test["diagnostics"])



# Step 5D: Plot numerical total variables against exact solution


def plot_exact_comparison_OS_E(result):
    x_np = result["x"].detach().cpu().numpy()

    rho_tot_np = result["rho_tot"].detach().cpu().numpy()
    u_tot_np = result["u_tot"].detach().cpu().numpy()
    p_tot_np = result["p_tot"].detach().cpu().numpy()

    rho_ex_np = result["rho_ex"].detach().cpu().numpy()
    u_ex_np = result["u_ex"].detach().cpu().numpy()
    p_ex_np = result["p_ex"].detach().cpu().numpy()

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    axes[0].plot(x_np, rho_tot_np, label="two-fluid total density")
    axes[0].plot(x_np, rho_ex_np, "--", label="single-fluid exact")
    axes[0].set_ylabel("density")
    axes[0].set_title("Density")
    axes[0].grid()
    axes[0].legend()

    axes[1].plot(x_np, u_tot_np, label="two-fluid total velocity")
    axes[1].plot(x_np, u_ex_np, "--", label="single-fluid exact")
    axes[1].set_ylabel("velocity")
    axes[1].set_title("Velocity")
    axes[1].grid()
    axes[1].legend()

    axes[2].plot(x_np, p_tot_np, label="two-fluid total pressure")
    axes[2].plot(x_np, p_ex_np, "--", label="single-fluid exact")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("pressure")
    axes[2].set_title("Pressure")
    axes[2].grid()
    axes[2].legend()

    fig.suptitle("Rusanov OS vs single-fluid exact solution", fontsize=14)

    save_figure(fig, "07_OS_exact_comparison.png")
    plt.show()
    plt.close(fig)


plot_exact_comparison_OS_E(result_exact_test)



# Step 5E: Error table for report


df_exact_error = pd.DataFrame([{
    "nu_in": result_exact_test["nu_in"],
    "steps": result_exact_test["steps"],
    "L1_rho": result_exact_test["errors_exact"]["L1_rho"],
    "Linf_rho": result_exact_test["errors_exact"]["Linf_rho"],
    "L1_u": result_exact_test["errors_exact"]["L1_u"],
    "Linf_u": result_exact_test["errors_exact"]["Linf_u"],
    "L1_p": result_exact_test["errors_exact"]["L1_p"],
    "Linf_p": result_exact_test["errors_exact"]["Linf_p"],
    "mean_drift_final": result_exact_test["diagnostics"]["mean_drift_final"],
}])

print(df_exact_error)


# Step 6A: One-fluid subcycling over one macro time interval


def subcycle_one_fluid_E(U, dx, macro_dt, U_L, U_R,
                         gamma=gamma, cfl=0.4,
                         max_substeps=100000):
    """
    Evolve one fluid over a macro time interval.

    The fluid chooses its own local CFL time step.

    This is the key building block for the per-fluid method:
        each fluid is advanced with its own stable time step,
        instead of forcing all fluids to use the smallest global dt.
    """

    t_local = 0.0
    substeps = 0

    U_now = U.clone()

    while t_local < macro_dt and substeps < max_substeps:
        U_now = apply_fixed_bc_one_fluid_E(U_now, U_L, U_R)

        dt_local = compute_dt_one_fluid_E(
            U_now, dx, gamma=gamma, cfl=cfl
        ).item()

        if t_local + dt_local > macro_dt:
            dt_local = macro_dt - t_local

        if dt_local <= 0:
            print("dt_local <= 0")
            break

        U_now = rusanov_step_one_fluid_E(
            U_now, dx, dt_local, U_L, U_R, gamma=gamma
        )

        t_local += dt_local
        substeps += 1

    if substeps >= max_substeps:
        print("Warning: max_substeps reached in subcycle_one_fluid_E")

    return U_now, substeps


# Step 6B: Per-fluid operator-splitting step

def per_fluid_step_two_fluid_E(U_i, U_n, dx, macro_dt,
                               U_i_L, U_i_R, U_n_L, U_n_R,
                               nu_in,
                               gamma=gamma,
                               cfl=0.4):
    """
    One per-fluid operator splitting step.

    Step A:
        Evolve ion fluid over macro_dt using its own substeps.

    Step B:
        Evolve neutral fluid over macro_dt using its own substeps.

    Step C:
        Apply exact collision relaxation over macro_dt.

    This is the simplified ADI/per-fluid baseline.
    """

    U_i_star, nsub_i = subcycle_one_fluid_E(
        U_i, dx, macro_dt,
        U_i_L, U_i_R,
        gamma=gamma,
        cfl=cfl
    )

    U_n_star, nsub_n = subcycle_one_fluid_E(
        U_n, dx, macro_dt,
        U_n_L, U_n_R,
        gamma=gamma,
        cfl=cfl
    )

    U_i_new, U_n_new = collision_source_exact_E(
        U_i_star, U_n_star, macro_dt, nu_in, gamma=gamma
    )

    U_i_new = apply_fixed_bc_one_fluid_E(U_i_new, U_i_L, U_i_R)
    U_n_new = apply_fixed_bc_one_fluid_E(U_n_new, U_n_L, U_n_R)

    return U_i_new, U_n_new, nsub_i, nsub_n



# Step 6C: Run two-fluid Sod with per-fluid subcycling


def run_two_fluid_sod_case_per_fluid_E(nu_in,
                                       x_min=-5.0, x_max=5.0, N=1000,
                                       x0=0.0, t_end=0.08,
                                       ion_frac_L=0.5, ion_frac_R=0.5,
                                       press_frac_i=0.5,
                                       drift_n_L=0.0, drift_n_R=0.0,
                                       cfl=0.4,
                                       macro_factor=1.0,
                                       gamma=gamma,
                                       max_macro_steps=100000):
    """
    Run two-fluid Sod using the simplified per-fluid method.

    macro_dt is chosen from the larger of the two fluid CFL steps:

        macro_dt = macro_factor * max(dt_i, dt_n)

    Then each fluid subcycles inside this macro step using its own CFL.
    """

    x = torch.linspace(x_min, x_max, N, dtype=dtype, device=device)
    dx = x[1] - x[0]

    U_i, U_n = initial_condition_two_fluid_sod_E(
        x, x0=x0, gamma=gamma,
        ion_frac_L=ion_frac_L,
        ion_frac_R=ion_frac_R,
        press_frac_i=press_frac_i,
        drift_n_L=drift_n_L,
        drift_n_R=drift_n_R
    )

    U_i_L = U_i[:, 0].clone()
    U_i_R = U_i[:, -1].clone()
    U_n_L = U_n[:, 0].clone()
    U_n_R = U_n[:, -1].clone()

    t = 0.0
    macro_step = 0

    t_hist = []
    drift_hist = []
    macro_dt_hist = []

    substeps_i_hist = []
    substeps_n_hist = []

    total_substeps_i = 0
    total_substeps_n = 0

    while t < t_end and macro_step < max_macro_steps:
        U_i = apply_fixed_bc_one_fluid_E(U_i, U_i_L, U_i_R)
        U_n = apply_fixed_bc_one_fluid_E(U_n, U_n_L, U_n_R)

        rho_i_now, u_i_now, p_i_now, rho_n_now, u_n_now, p_n_now = state_to_prim_two_fluid_E(
            U_i, U_n, gamma=gamma
        )

        drift_now = torch.mean(torch.abs(u_i_now - u_n_now)).item()

        t_hist.append(t)
        drift_hist.append(drift_now)

        dt_i = compute_dt_one_fluid_E(U_i, dx, gamma=gamma, cfl=cfl).item()
        dt_n = compute_dt_one_fluid_E(U_n, dx, gamma=gamma, cfl=cfl).item()

        # Main difference from global method:
        # use the larger fluid time step as macro time interval
        macro_dt = macro_factor * max(dt_i, dt_n)

        if t + macro_dt > t_end:
            macro_dt = t_end - t

        if macro_dt <= 0:
            print("macro_dt <= 0")
            break

        macro_dt_hist.append(macro_dt)

        U_i, U_n, nsub_i, nsub_n = per_fluid_step_two_fluid_E(
            U_i, U_n, dx, macro_dt,
            U_i_L, U_i_R,
            U_n_L, U_n_R,
            nu_in=nu_in,
            gamma=gamma,
            cfl=cfl
        )

        substeps_i_hist.append(nsub_i)
        substeps_n_hist.append(nsub_n)

        total_substeps_i += nsub_i
        total_substeps_n += nsub_n

        t += macro_dt
        macro_step += 1

    if macro_step >= max_macro_steps:
        print("Warning: max_macro_steps reached")

    rho_i, u_i, p_i, rho_n, u_n, p_n = state_to_prim_two_fluid_E(
        U_i, U_n, gamma=gamma
    )

    rho_tot = rho_i + rho_n
    m_tot = U_i[1] + U_n[1]
    u_tot = m_tot / torch.clamp(rho_tot, min=eps)
    p_tot = p_i + p_n

    result = {
        "x": x,
        "U_i": U_i,
        "U_n": U_n,

        "rho_i": rho_i,
        "u_i": u_i,
        "p_i": p_i,

        "rho_n": rho_n,
        "u_n": u_n,
        "p_n": p_n,

        "rho_tot": rho_tot,
        "u_tot": u_tot,
        "p_tot": p_tot,

        "t_hist": t_hist,
        "drift_hist": drift_hist,
        "macro_dt_hist": macro_dt_hist,

        "substeps_i_hist": substeps_i_hist,
        "substeps_n_hist": substeps_n_hist,

        "nu_in": nu_in,
        "final_time": t,
        "macro_steps": macro_step,

        "total_substeps_i": total_substeps_i,
        "total_substeps_n": total_substeps_n,
        "total_hydro_substeps": total_substeps_i + total_substeps_n,

        "diagnostics": {
            "min_rho_i": torch.min(rho_i).item(),
            "min_rho_n": torch.min(rho_n).item(),
            "min_p_i": torch.min(p_i).item(),
            "min_p_n": torch.min(p_n).item(),
            "mean_drift_final": torch.mean(torch.abs(u_i - u_n)).item(),
            "max_drift_final": torch.max(torch.abs(u_i - u_n)).item(),
        }
    }

    return result


# Step 6D: Test one per-fluid case



result_pf = run_two_fluid_sod_case_per_fluid_E(
    nu_in=0.5,
    N=1000,
    t_end=0.08,
    ion_frac_L=0.8,
    ion_frac_R=0.2,
    press_frac_i=0.5,
    drift_n_L=0.2,
    drift_n_R=0.0,
    cfl=0.4,
    macro_factor=1.0
)

print("final time =", result_pf["final_time"])
print("macro steps =", result_pf["macro_steps"])
print("total ion substeps =", result_pf["total_substeps_i"])
print("total neutral substeps =", result_pf["total_substeps_n"])
print("total hydro substeps =", result_pf["total_hydro_substeps"])
print("diagnostics:")
print(result_pf["diagnostics"])

plot_two_fluid_sod_OS_E(result_pf)


# Step 7A: Compare global OS and per-fluid method


import time

def compare_global_and_per_fluid(nu_in,
                                 N=1000,
                                 t_end=0.08,
                                 ion_frac_L=0.05,
                                 ion_frac_R=0.05,
                                 press_frac_i=0.5,
                                 drift_n_L=0.0,
                                 drift_n_R=0.0,
                                 cfl=0.4,
                                 macro_factor=1.0):
    """
    Compare:
        1. global OS method using dt = min(dt_i, dt_n)
        2. per-fluid subcycling method

    The density split is chosen so that the two fluids may have
    different characteristic speeds and therefore different CFL steps.
    """

    # Global OS method

    tic = time.perf_counter()

    result_global = run_two_fluid_sod_case_OS_E(
        nu_in=nu_in,
        N=N,
        t_end=t_end,
        ion_frac_L=ion_frac_L,
        ion_frac_R=ion_frac_R,
        press_frac_i=press_frac_i,
        drift_n_L=drift_n_L,
        drift_n_R=drift_n_R,
        cfl=cfl
    )

    time_global = time.perf_counter() - tic

    # In each global step, both ion and neutral are updated once.
    global_hydro_updates = 2 * result_global["steps"]


    # Per-fluid method

    tic = time.perf_counter()

    result_pf = run_two_fluid_sod_case_per_fluid_E(
        nu_in=nu_in,
        N=N,
        t_end=t_end,
        ion_frac_L=ion_frac_L,
        ion_frac_R=ion_frac_R,
        press_frac_i=press_frac_i,
        drift_n_L=drift_n_L,
        drift_n_R=drift_n_R,
        cfl=cfl,
        macro_factor=macro_factor
    )

    time_pf = time.perf_counter() - tic

    pf_hydro_updates = result_pf["total_hydro_substeps"]

    # -----------------------------
    # Difference between final results
    # -----------------------------
    rho_diff = torch.mean(torch.abs(result_global["rho_tot"] - result_pf["rho_tot"])).item()
    u_diff   = torch.mean(torch.abs(result_global["u_tot"]   - result_pf["u_tot"])).item()
    p_diff   = torch.mean(torch.abs(result_global["p_tot"]   - result_pf["p_tot"])).item()

    # estimated speed-up based on hydro update count
    update_speedup = global_hydro_updates / max(pf_hydro_updates, 1)

    # measured wall-clock speed-up
    runtime_speedup = time_global / max(time_pf, 1e-15)

    summary = {
        "nu_in": nu_in,
        "N": N,
        "ion_frac_L": ion_frac_L,
        "ion_frac_R": ion_frac_R,

        "global_steps": result_global["steps"],
        "global_hydro_updates": global_hydro_updates,
        "global_runtime": time_global,

        "pf_macro_steps": result_pf["macro_steps"],
        "pf_ion_substeps": result_pf["total_substeps_i"],
        "pf_neutral_substeps": result_pf["total_substeps_n"],
        "pf_hydro_updates": pf_hydro_updates,
        "pf_runtime": time_pf,

        "update_speedup": update_speedup,
        "runtime_speedup": runtime_speedup,

        "L1_diff_rho_total": rho_diff,
        "L1_diff_u_total": u_diff,
        "L1_diff_p_total": p_diff,

        "global_mean_drift": result_global["diagnostics"]["mean_drift_final"],
        "pf_mean_drift": result_pf["diagnostics"]["mean_drift_final"],
    }

    return result_global, result_pf, summary



# Step 7B: Run one speed-up comparison case


result_global_test, result_pf_test, summary_test = compare_global_and_per_fluid(
    nu_in=0.5,
    N=1000,
    t_end=0.08,
    ion_frac_L=0.05,
    ion_frac_R=0.05,
    press_frac_i=0.5,
    drift_n_L=0.0,
    drift_n_R=0.0,
    cfl=0.4,
    macro_factor=1.0
)

print("Speed-up comparison summary:")
for key, value in summary_test.items():
    print(key, "=", value)





# Step 7C: Plot global OS vs per-fluid final result

def plot_global_vs_per_fluid(result_global, result_pf):
    x_np = result_global["x"].detach().cpu().numpy()

    rho_g = result_global["rho_tot"].detach().cpu().numpy()
    u_g   = result_global["u_tot"].detach().cpu().numpy()
    p_g   = result_global["p_tot"].detach().cpu().numpy()

    rho_pf = result_pf["rho_tot"].detach().cpu().numpy()
    u_pf   = result_pf["u_tot"].detach().cpu().numpy()
    p_pf   = result_pf["p_tot"].detach().cpu().numpy()

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    axes[0].plot(x_np, rho_g, label="global OS")
    axes[0].plot(x_np, rho_pf, "--", label="per-fluid")
    axes[0].set_ylabel("rho_total")
    axes[0].set_title("Total density")
    axes[0].grid()
    axes[0].legend()

    axes[1].plot(x_np, u_g, label="global OS")
    axes[1].plot(x_np, u_pf, "--", label="per-fluid")
    axes[1].set_ylabel("u_total")
    axes[1].set_title("Total velocity")
    axes[1].grid()
    axes[1].legend()

    axes[2].plot(x_np, p_g, label="global OS")
    axes[2].plot(x_np, p_pf, "--", label="per-fluid")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("p_total")
    axes[2].set_title("Total pressure")
    axes[2].grid()
    axes[2].legend()

    fig.suptitle("Global OS vs per-fluid method", fontsize=14)

    save_figure(fig, "08_global_vs_per_fluid.png")
    plt.show()
    plt.close(fig)

plot_global_vs_per_fluid(result_global_test, result_pf_test)



# Step 7D: Summary table for one comparison


df_speedup_one = pd.DataFrame([summary_test])

print(df_speedup_one[[
    "nu_in",
    "ion_frac_L",
    "global_steps",
    "global_hydro_updates",
    "pf_macro_steps",
    "pf_ion_substeps",
    "pf_neutral_substeps",
    "pf_hydro_updates",
    "update_speedup",
    "runtime_speedup",
    "L1_diff_rho_total",
    "L1_diff_u_total",
    "L1_diff_p_total"
]])



# Step 8A: Speed-up test for different density splits


ion_frac_list = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]

speedup_summaries = []

for ion_frac in ion_frac_list:
    print(f"\nRunning density contrast case: ion_frac = {ion_frac}")

    result_global, result_pf, summary = compare_global_and_per_fluid(
        nu_in=0.5,
        N=500,
        t_end=0.08,
        ion_frac_L=ion_frac,
        ion_frac_R=ion_frac,
        press_frac_i=0.5,
        drift_n_L=0.0,
        drift_n_R=0.0,
        cfl=0.4,
        macro_factor=1.0
    )

    speedup_summaries.append(summary)

    print("global hydro updates =", summary["global_hydro_updates"])
    print("per-fluid hydro updates =", summary["pf_hydro_updates"])
    print("update speed-up =", summary["update_speedup"])
    print("runtime speed-up =", summary["runtime_speedup"])
    print("L1 diff rho =", summary["L1_diff_rho_total"])
    print("L1 diff u   =", summary["L1_diff_u_total"])
    print("L1 diff p   =", summary["L1_diff_p_total"])






# Step 8B: Summary table for density-contrast speed-up test


df_speedup = pd.DataFrame(speedup_summaries)

print("Density contrast speed-up table:")
print(df_speedup[[
    "ion_frac_L",
    "global_steps",
    "global_hydro_updates",
    "pf_macro_steps",
    "pf_ion_substeps",
    "pf_neutral_substeps",
    "pf_hydro_updates",
    "update_speedup",
    "runtime_speedup",
    "L1_diff_rho_total",
    "L1_diff_u_total",
    "L1_diff_p_total"
]])


# Step 8C: Plot speed-up versus ion density fraction


plt.figure(figsize=(8,5))
plt.plot(df_speedup["ion_frac_L"], df_speedup["update_speedup"], marker="o", label="hydro update speed-up")
plt.plot(df_speedup["ion_frac_L"], df_speedup["runtime_speedup"], marker="s", label="runtime speed-up")
plt.xscale("log")
plt.xlabel("ion density fraction")
plt.ylabel("speed-up")
plt.title("Speed-up versus ion density fraction")
plt.grid()
plt.legend()
plt.close()



# Step 8D: Hydro update count comparison


plt.figure(figsize=(8,5))
plt.plot(df_speedup["ion_frac_L"], df_speedup["global_hydro_updates"], marker="o", label="global OS hydro updates")
plt.plot(df_speedup["ion_frac_L"], df_speedup["pf_hydro_updates"], marker="s", label="per-fluid hydro updates")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("ion density fraction")
plt.ylabel("number of hydro updates")
plt.title("Hydro update count versus ion density fraction")
plt.grid()
plt.legend()
plt.close()



# Step 8E: Ion and neutral substeps in per-fluid method


plt.figure(figsize=(8,5))
plt.plot(df_speedup["ion_frac_L"], df_speedup["pf_ion_substeps"], marker="o", label="ion substeps")
plt.plot(df_speedup["ion_frac_L"], df_speedup["pf_neutral_substeps"], marker="s", label="neutral substeps")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("ion density fraction")
plt.ylabel("number of substeps")
plt.title("Per-fluid substeps versus ion density fraction")
plt.grid()
plt.legend()
plt.close()


# Step 8F: Difference between global OS and per-fluid method

plt.figure(figsize=(8,5))
plt.plot(df_speedup["ion_frac_L"], df_speedup["L1_diff_rho_total"], marker="o", label="L1 diff rho")
plt.plot(df_speedup["ion_frac_L"], df_speedup["L1_diff_u_total"], marker="s", label="L1 diff u")
plt.plot(df_speedup["ion_frac_L"], df_speedup["L1_diff_p_total"], marker="^", label="L1 diff p")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("ion density fraction")
plt.ylabel("L1 difference")
plt.title("Difference between global OS and per-fluid method")
plt.grid()
plt.legend()
plt.close()


# Step 9A: TVD MUSCL-Rusanov method
# This completes the TVD part of the project.


def minmod(a, b):
    """
    Minmod slope limiter.
    """
    same_sign = (a * b) > 0.0
    return torch.where(
        same_sign,
        torch.sign(a) * torch.minimum(torch.abs(a), torch.abs(b)),
        torch.zeros_like(a)
    )


def limited_slope(U):
    """
    Compute TVD limited slopes for U.
    U has shape (3, N).
    """
    slope = torch.zeros_like(U)

    d_left  = U[:, 1:-1] - U[:, :-2]
    d_right = U[:, 2:]   - U[:, 1:-1]

    slope[:, 1:-1] = minmod(d_left, d_right)

    return slope


def tvd_rusanov_step_one_fluid_E(U, dx, dt, U_L, U_R, gamma=gamma):
    """
    One TVD MUSCL-Rusanov finite-volume step for one fluid.

    Conservative variables:
        U = (rho, m, E)

    This is a second-order TVD reconstruction with minmod limiter.
    """

    U = apply_fixed_bc_one_fluid_E(U, U_L, U_R)

    slope = limited_slope(U)

    # Interface states at j+1/2
    U_left  = U[:, :-1] + 0.5 * slope[:, :-1]
    U_right = U[:, 1:]  - 0.5 * slope[:, 1:]

    # Rusanov flux at interfaces
    F_half = rusanov_flux_E(U_left, U_right, gamma=gamma)

    U_new = U.clone()

    # Finite-volume update for interior cells
    U_new[:, 1:-1] = U[:, 1:-1] - dt / dx * (
        F_half[:, 1:] - F_half[:, :-1]
    )

    # Positivity protection
    U_new[0] = torch.clamp(U_new[0], min=eps)

    rho_new, u_new, p_new = state_to_prim_E(U_new, gamma=gamma)
    E_internal = p_new / (gamma - 1.0)
    E_kinetic  = 0.5 * rho_new * u_new**2
    U_new[2] = torch.clamp(E_internal + E_kinetic, min=eps)

    U_new = apply_fixed_bc_one_fluid_E(U_new, U_L, U_R)

    return U_new


def tvd_step_two_fluid_OS_E(U_i, U_n, dx, dt,
                            U_i_L, U_i_R, U_n_L, U_n_R,
                            nu_in,
                            gamma=gamma):
    """
    One two-fluid TVD operator-splitting step.

    Step 1: TVD hydro step for ions.
    Step 2: TVD hydro step for neutrals.
    Step 3: exact collision momentum relaxation.
    """

    U_i_star = tvd_rusanov_step_one_fluid_E(
        U_i, dx, dt, U_i_L, U_i_R, gamma=gamma
    )

    U_n_star = tvd_rusanov_step_one_fluid_E(
        U_n, dx, dt, U_n_L, U_n_R, gamma=gamma
    )

    U_i_new, U_n_new = collision_source_exact_E(
        U_i_star, U_n_star, dt, nu_in, gamma=gamma
    )

    U_i_new = apply_fixed_bc_one_fluid_E(U_i_new, U_i_L, U_i_R)
    U_n_new = apply_fixed_bc_one_fluid_E(U_n_new, U_n_L, U_n_R)

    return U_i_new, U_n_new



# Step 9B: Run two-fluid Sod with TVD method


def run_two_fluid_sod_case_TVD_OS_E(nu_in,
                                    x_min=-5.0, x_max=5.0, N=1000,
                                    x0=0.0, t_end=0.08,
                                    ion_frac_L=0.5, ion_frac_R=0.5,
                                    press_frac_i=0.5,
                                    drift_n_L=0.0, drift_n_R=0.0,
                                    cfl=0.35,
                                    gamma=gamma,
                                    max_steps=100000):

    x = torch.linspace(x_min, x_max, N, dtype=dtype, device=device)
    dx = x[1] - x[0]

    U_i, U_n = initial_condition_two_fluid_sod_E(
        x, x0=x0, gamma=gamma,
        ion_frac_L=ion_frac_L,
        ion_frac_R=ion_frac_R,
        press_frac_i=press_frac_i,
        drift_n_L=drift_n_L,
        drift_n_R=drift_n_R
    )

    U_i_L = U_i[:, 0].clone()
    U_i_R = U_i[:, -1].clone()
    U_n_L = U_n[:, 0].clone()
    U_n_R = U_n[:, -1].clone()

    t = 0.0
    step = 0

    t_hist = []
    drift_hist = []
    dt_hist = []

    while t < t_end and step < max_steps:
        U_i = apply_fixed_bc_one_fluid_E(U_i, U_i_L, U_i_R)
        U_n = apply_fixed_bc_one_fluid_E(U_n, U_n_L, U_n_R)

        rho_i_now, u_i_now, p_i_now, rho_n_now, u_n_now, p_n_now = state_to_prim_two_fluid_E(
            U_i, U_n, gamma=gamma
        )

        drift_now = torch.mean(torch.abs(u_i_now - u_n_now)).item()

        t_hist.append(t)
        drift_hist.append(drift_now)

        dt = compute_dt_two_fluid_E(U_i, U_n, dx, gamma=gamma, cfl=cfl).item()

        if t + dt > t_end:
            dt = t_end - t

        if dt <= 0:
            print("dt <= 0")
            break

        dt_hist.append(dt)

        U_i, U_n = tvd_step_two_fluid_OS_E(
            U_i, U_n, dx, dt,
            U_i_L, U_i_R,
            U_n_L, U_n_R,
            nu_in=nu_in,
            gamma=gamma
        )

        t += dt
        step += 1

    if step >= max_steps:
        print("Warning: max_steps reached")

    rho_i, u_i, p_i, rho_n, u_n, p_n = state_to_prim_two_fluid_E(
        U_i, U_n, gamma=gamma
    )

    rho_tot = rho_i + rho_n
    m_tot = U_i[1] + U_n[1]
    u_tot = m_tot / torch.clamp(rho_tot, min=eps)
    p_tot = p_i + p_n

    result = {
        "x": x,
        "U_i": U_i,
        "U_n": U_n,

        "rho_i": rho_i,
        "u_i": u_i,
        "p_i": p_i,

        "rho_n": rho_n,
        "u_n": u_n,
        "p_n": p_n,

        "rho_tot": rho_tot,
        "u_tot": u_tot,
        "p_tot": p_tot,

        "t_hist": t_hist,
        "drift_hist": drift_hist,
        "dt_hist": dt_hist,

        "nu_in": nu_in,
        "final_time": t,
        "steps": step,

        "diagnostics": {
            "min_rho_i": torch.min(rho_i).item(),
            "min_rho_n": torch.min(rho_n).item(),
            "min_p_i": torch.min(p_i).item(),
            "min_p_n": torch.min(p_n).item(),
            "mean_drift_final": torch.mean(torch.abs(u_i - u_n)).item(),
            "max_drift_final": torch.max(torch.abs(u_i - u_n)).item(),
        }
    }

    return result



# Step 9C: TVD test with exact single-fluid comparison


result_tvd = run_two_fluid_sod_case_TVD_OS_E(
    nu_in=0.5,
    N=1000,
    t_end=0.08,
    ion_frac_L=0.5,
    ion_frac_R=0.5,
    press_frac_i=0.5,
    drift_n_L=0.0,
    drift_n_R=0.0,
    cfl=0.35
)

result_tvd = add_exact_comparison_to_result(
    result_tvd,
    x0=0.0,
    t_end=0.08,
    gamma=gamma
)

print("\nTVD test:")
print("final time =", result_tvd["final_time"])
print("steps =", result_tvd["steps"])
print("diagnostics =", result_tvd["diagnostics"])
print("errors exact =", result_tvd["errors_exact"])



# Step 9D: Plot TVD result against exact solution


def plot_exact_comparison_TVD_E(result):
    x_np = result["x"].detach().cpu().numpy()

    rho_tot_np = result["rho_tot"].detach().cpu().numpy()
    u_tot_np   = result["u_tot"].detach().cpu().numpy()
    p_tot_np   = result["p_tot"].detach().cpu().numpy()

    rho_ex_np = result["rho_ex"].detach().cpu().numpy()
    u_ex_np   = result["u_ex"].detach().cpu().numpy()
    p_ex_np   = result["p_ex"].detach().cpu().numpy()

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    axes[0].plot(x_np, rho_tot_np, label="TVD two-fluid total density")
    axes[0].plot(x_np, rho_ex_np, "--", label="single-fluid exact")
    axes[0].set_ylabel("density")
    axes[0].set_title("TVD two-fluid result vs exact solution")
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(x_np, u_tot_np, label="TVD two-fluid total velocity")
    axes[1].plot(x_np, u_ex_np, "--", label="single-fluid exact")
    axes[1].set_ylabel("velocity")
    axes[1].legend()
    axes[1].grid()

    axes[2].plot(x_np, p_tot_np, label="TVD two-fluid total pressure")
    axes[2].plot(x_np, p_ex_np, "--", label="single-fluid exact")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("pressure")
    axes[2].legend()
    axes[2].grid()

    save_figure(fig, "19_TVD_twofluid_vs_exact.png")
    plt.close()


plot_exact_comparison_TVD_E(result_tvd)



# Step 9E: TVD collision-frequency study


nu_list_TVD = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]

results_TVD = []

for nu in nu_list_TVD:
    print(f"\nRunning TVD case: nu_in = {nu}")

    result = run_two_fluid_sod_case_TVD_OS_E(
        nu_in=nu,
        N=1000,
        t_end=0.08,
        ion_frac_L=0.8,
        ion_frac_R=0.2,
        press_frac_i=0.5,
        drift_n_L=0.2,
        drift_n_R=0.0,
        cfl=0.35
    )

    result = add_exact_comparison_to_result(
        result,
        x0=0.0,
        t_end=0.08,
        gamma=gamma
    )

    results_TVD.append(result)

    print("steps =", result["steps"])
    print("mean drift final =", result["diagnostics"]["mean_drift_final"])
    print("max drift final  =", result["diagnostics"]["max_drift_final"])
    print("L1 rho exact =", result["errors_exact"]["L1_rho"])
    print("L1 u exact   =", result["errors_exact"]["L1_u"])
    print("L1 p exact   =", result["errors_exact"]["L1_p"])


TVD_table = []

for r in results_TVD:
    TVD_table.append({
        "nu_in": r["nu_in"],
        "steps": r["steps"],
        "L1_rho": r["errors_exact"]["L1_rho"],
        "L1_u": r["errors_exact"]["L1_u"],
        "L1_p": r["errors_exact"]["L1_p"],
        "mean_drift_final": r["diagnostics"]["mean_drift_final"],
        "max_drift_final": r["diagnostics"]["max_drift_final"],
        "min_rho_i": r["diagnostics"]["min_rho_i"],
        "min_rho_n": r["diagnostics"]["min_rho_n"],
        "min_p_i": r["diagnostics"]["min_p_i"],
        "min_p_n": r["diagnostics"]["min_p_n"],
    })
def save_figure(fig, filename):
    path = fig_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {path}")
def to_numpy(a):
    """
    Convert torch tensor to numpy array safely.
    This also works if the tensor is on GPU.
    """
    if torch.is_tensor(a):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def get_column(rows, key):
    """
    Extract one column from a list of dictionaries.
    """
    return np.array([row[key] for row in rows])


def print_table(rows, columns=None, title=None):
    """
    Simple table printer.
    rows should be a list of dictionaries.
    """
    if title is not None:
        print("\n" + title)

    if len(rows) == 0:
        print("Empty table")
        return

    if columns is None:
        columns = list(rows[0].keys())

    header = " | ".join(columns)
    print(header)
    print("-" * len(header))

    for row in rows:
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6e}")
            else:
                values.append(str(value))
        print(" | ".join(values))

print_table(
    TVD_table,
    columns=[
        "nu_in",
        "steps",
        "L1_rho",
        "L1_u",
        "L1_p",
        "mean_drift_final",
        "max_drift_final",
        "min_rho_i",
        "min_rho_n",
        "min_p_i",
        "min_p_n"
    ],
    title="TVD collision-frequency table"
)


# Step 9F: Plot TVD error and drift versus collision frequency

TVD_table_pos = [row for row in TVD_table if row["nu_in"] > 0]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(
    get_column(TVD_table_pos, "nu_in"),
    get_column(TVD_table_pos, "L1_rho"),
    marker="o",
    label="L1 rho"
)

axes[0].plot(
    get_column(TVD_table_pos, "nu_in"),
    get_column(TVD_table_pos, "L1_u"),
    marker="s",
    label="L1 u"
)

axes[0].plot(
    get_column(TVD_table_pos, "nu_in"),
    get_column(TVD_table_pos, "L1_p"),
    marker="^",
    label="L1 p"
)

axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel("nu_in")
axes[0].set_ylabel("L1 error")
axes[0].set_title("TVD error vs collision frequency")
axes[0].grid()
axes[0].legend()

axes[1].plot(
    get_column(TVD_table_pos, "nu_in"),
    get_column(TVD_table_pos, "mean_drift_final"),
    marker="o",
    label="mean final drift"
)

axes[1].plot(
    get_column(TVD_table_pos, "nu_in"),
    get_column(TVD_table_pos, "max_drift_final"),
    marker="s",
    label="max final drift"
)

axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("nu_in")
axes[1].set_ylabel("final drift")
axes[1].set_title("TVD final drift vs collision frequency")
axes[1].grid()
axes[1].legend()

fig.suptitle("TVD collision-frequency study", fontsize=14)

save_figure(fig, "10_TVD_error_and_drift_vs_collision_frequency.png")
plt.show()
plt.close(fig)



# Step 9G: Representative TVD cases


selected_nu_TVD = [0.0, 1.0, 100.0]
selected_results_TVD = [r for r in results_TVD if r["nu_in"] in selected_nu_TVD]

fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

for r in selected_results_TVD:
    x_np = r["x"].detach().cpu().numpy()

    axes[0].plot(
        x_np,
        r["rho_tot"].detach().cpu().numpy(),
        label=f'nu_in={r["nu_in"]}'
    )

    axes[1].plot(
        x_np,
        r["u_tot"].detach().cpu().numpy(),
        label=f'nu_in={r["nu_in"]}'
    )

    axes[2].plot(
        x_np,
        r["p_tot"].detach().cpu().numpy(),
        label=f'nu_in={r["nu_in"]}'
    )

# exact reference from first result
x_ref = results_TVD[0]["x"].detach().cpu().numpy()
axes[0].plot(x_ref, results_TVD[0]["rho_ex"].detach().cpu().numpy(), "k--", label="exact")
axes[1].plot(x_ref, results_TVD[0]["u_ex"].detach().cpu().numpy(), "k--", label="exact")
axes[2].plot(x_ref, results_TVD[0]["p_ex"].detach().cpu().numpy(), "k--", label="exact")

axes[0].set_ylabel("density")
axes[0].set_title("Representative TVD cases")
axes[0].legend()
axes[0].grid()

axes[1].set_ylabel("velocity")
axes[1].legend()
axes[1].grid()

axes[2].set_xlabel("x")
axes[2].set_ylabel("pressure")
axes[2].legend()
axes[2].grid()

save_figure(fig, "33_TVD_representative_cases.png")
plt.show()
plt.close(fig)


# Save TVD table as LaTeX and PDF


df_TVD = pd.DataFrame(TVD_table)

# Save as LaTeX table
latex_path = fig_dir / "TVD_collision_frequency_table.tex"

df_TVD.to_latex(
    latex_path,
    index=False,
    float_format="%.4e",
    caption="TVD collision-frequency study.",
    label="tab:tvd_collision_frequency"
)

print(f"Saved LaTeX table: {latex_path}")


# Save as PDF table
fig, ax = plt.subplots(figsize=(14, 4))
ax.axis("off")

table = ax.table(
    cellText=df_TVD.round(4).values,
    colLabels=df_TVD.columns,
    loc="center",
    cellLoc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.0, 1.4)

ax.set_title("TVD collision-frequency table", fontsize=12)

pdf_table_path = fig_dir / "TVD_collision_frequency_table.pdf"
fig.tight_layout()
fig.savefig(pdf_table_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)
print(f"Saved PDF table: {pdf_table_path}")



# Step 10: ADI-like per-fluid TVD operator splitting method

#
# This part is appended to the previous code.
# It does not replace the old global TVD solver.
#
# Idea:
#   Global TVD OS:
#       dt = min(dt_i, dt_n)
#       update both fluids once per global step
#
#   New ADI-like per-fluid method:
#       choose a macro time interval
#       each fluid evolves over this macro interval using its own CFL substeps
#       then apply collision/source relaxation
#
# This is "per-fluid" splitting:
#       not per equation component,
#       not per source term,
#       but per fluid system.
#
# Since this is a 1D problem, this is not a classical x/y/z ADI.
# It is an ADI-inspired / per-fluid operator splitting method.



# Step 10A: TVD subcycling for one fluid


def subcycle_one_fluid_TVD_E(U, dx, macro_dt, U_L, U_R,
                             gamma=gamma,
                             cfl=0.35,
                             max_substeps=100000):
    """
    Evolve one fluid over a given macro time interval using
    TVD MUSCL-Rusanov substeps.

    The fluid chooses its own local CFL time step.

    This is the key building block of the per-fluid ADI-like method.
    """

    t_local = 0.0
    substeps = 0

    U_now = U.clone()

    while t_local < macro_dt and substeps < max_substeps:

        U_now = apply_fixed_bc_one_fluid_E(U_now, U_L, U_R)

        dt_local = compute_dt_one_fluid_E(
            U_now,
            dx,
            gamma=gamma,
            cfl=cfl
        ).item()

        if t_local + dt_local > macro_dt:
            dt_local = macro_dt - t_local

        if dt_local <= 0:
            print("dt_local <= 0 in subcycle_one_fluid_TVD_E")
            break

        U_now = tvd_rusanov_step_one_fluid_E(
            U_now,
            dx,
            dt_local,
            U_L,
            U_R,
            gamma=gamma
        )

        t_local += dt_local
        substeps += 1

    if substeps >= max_substeps:
        print("Warning: max_substeps reached in subcycle_one_fluid_TVD_E")

    return U_now, substeps


# Step 10B: One ADI-like per-fluid TVD step

def adi_per_fluid_step_two_fluid_TVD_E(U_i, U_n, dx, macro_dt,
                                       U_i_L, U_i_R,
                                       U_n_L, U_n_R,
                                       nu_in,
                                       gamma=gamma,
                                       cfl=0.35,
                                       use_strang=True):
    """
    One ADI-like per-fluid operator splitting step.

    If use_strang=True:

        1. half collision step
        2. ion fluid evolves over macro_dt using its own TVD substeps
        3. neutral fluid evolves over macro_dt using its own TVD substeps
        4. half collision step

    This is more symmetric than the simple Lie splitting.

    If use_strang=False:

        1. ion fluid hydro step over macro_dt
        2. neutral fluid hydro step over macro_dt
        3. full collision step
    """

    # Fixed boundary before starting
    U_i = apply_fixed_bc_one_fluid_E(U_i, U_i_L, U_i_R)
    U_n = apply_fixed_bc_one_fluid_E(U_n, U_n_L, U_n_R)

    if use_strang:
        # Half source step
        U_i_half, U_n_half = collision_source_exact_E(
            U_i,
            U_n,
            0.5 * macro_dt,
            nu_in,
            gamma=gamma
        )

        U_i_half = apply_fixed_bc_one_fluid_E(U_i_half, U_i_L, U_i_R)
        U_n_half = apply_fixed_bc_one_fluid_E(U_n_half, U_n_L, U_n_R)

        # Per-fluid hydro subcycling
        U_i_star, nsub_i = subcycle_one_fluid_TVD_E(
            U_i_half,
            dx,
            macro_dt,
            U_i_L,
            U_i_R,
            gamma=gamma,
            cfl=cfl
        )

        U_n_star, nsub_n = subcycle_one_fluid_TVD_E(
            U_n_half,
            dx,
            macro_dt,
            U_n_L,
            U_n_R,
            gamma=gamma,
            cfl=cfl
        )

        # Another half source step
        U_i_new, U_n_new = collision_source_exact_E(
            U_i_star,
            U_n_star,
            0.5 * macro_dt,
            nu_in,
            gamma=gamma
        )

    else:
        # Simple Lie splitting
        U_i_star, nsub_i = subcycle_one_fluid_TVD_E(
            U_i,
            dx,
            macro_dt,
            U_i_L,
            U_i_R,
            gamma=gamma,
            cfl=cfl
        )

        U_n_star, nsub_n = subcycle_one_fluid_TVD_E(
            U_n,
            dx,
            macro_dt,
            U_n_L,
            U_n_R,
            gamma=gamma,
            cfl=cfl
        )

        U_i_new, U_n_new = collision_source_exact_E(
            U_i_star,
            U_n_star,
            macro_dt,
            nu_in,
            gamma=gamma
        )

    U_i_new = apply_fixed_bc_one_fluid_E(U_i_new, U_i_L, U_i_R)
    U_n_new = apply_fixed_bc_one_fluid_E(U_n_new, U_n_L, U_n_R)

    return U_i_new, U_n_new, nsub_i, nsub_n



# Step 10C: Run two-fluid Sod with ADI-like per-fluid TVD method


def run_two_fluid_sod_case_ADI_TVD_E(nu_in,
                                     x_min=-5.0,
                                     x_max=5.0,
                                     N=1000,
                                     x0=0.0,
                                     t_end=0.08,
                                     ion_frac_L=0.5,
                                     ion_frac_R=0.5,
                                     press_frac_i=0.5,
                                     drift_n_L=0.0,
                                     drift_n_R=0.0,
                                     cfl=0.35,
                                     macro_factor=1.0,
                                     gamma=gamma,
                                     use_strang=True,
                                     max_macro_steps=100000):
    """
    Run a two-fluid Sod shock tube using the new ADI-like per-fluid TVD method.

    macro_dt is chosen by

        macro_dt = macro_factor * max(dt_i, dt_n)

    Then each fluid subcycles inside this macro interval using its own CFL.

    This allows the two fluids to use different effective time steps.
    """

    x = torch.linspace(x_min, x_max, N, dtype=dtype, device=device)
    dx = x[1] - x[0]

    U_i, U_n = initial_condition_two_fluid_sod_E(
        x,
        x0=x0,
        gamma=gamma,
        ion_frac_L=ion_frac_L,
        ion_frac_R=ion_frac_R,
        press_frac_i=press_frac_i,
        drift_n_L=drift_n_L,
        drift_n_R=drift_n_R
    )

    # Fixed boundary states
    U_i_L = U_i[:, 0].clone()
    U_i_R = U_i[:, -1].clone()
    U_n_L = U_n[:, 0].clone()
    U_n_R = U_n[:, -1].clone()

    t = 0.0
    macro_step = 0

    t_hist = []
    drift_hist = []
    macro_dt_hist = []
    dt_i_hist = []
    dt_n_hist = []

    substeps_i_hist = []
    substeps_n_hist = []

    total_substeps_i = 0
    total_substeps_n = 0

    while t < t_end and macro_step < max_macro_steps:

        U_i = apply_fixed_bc_one_fluid_E(U_i, U_i_L, U_i_R)
        U_n = apply_fixed_bc_one_fluid_E(U_n, U_n_L, U_n_R)

        rho_i_now, u_i_now, p_i_now, rho_n_now, u_n_now, p_n_now = state_to_prim_two_fluid_E(
            U_i,
            U_n,
            gamma=gamma
        )

        drift_now = torch.mean(torch.abs(u_i_now - u_n_now)).item()

        t_hist.append(t)
        drift_hist.append(drift_now)

        dt_i = compute_dt_one_fluid_E(
            U_i,
            dx,
            gamma=gamma,
            cfl=cfl
        ).item()

        dt_n = compute_dt_one_fluid_E(
            U_n,
            dx,
            gamma=gamma,
            cfl=cfl
        ).item()

        dt_i_hist.append(dt_i)
        dt_n_hist.append(dt_n)

        # Key difference from global method:
        # global method uses min(dt_i, dt_n)
        # here we use max(dt_i, dt_n) as macro interval
        macro_dt = macro_factor * max(dt_i, dt_n)

        if t + macro_dt > t_end:
            macro_dt = t_end - t

        if macro_dt <= 0:
            print("macro_dt <= 0 in run_two_fluid_sod_case_ADI_TVD_E")
            break

        macro_dt_hist.append(macro_dt)

        U_i, U_n, nsub_i, nsub_n = adi_per_fluid_step_two_fluid_TVD_E(
            U_i,
            U_n,
            dx,
            macro_dt,
            U_i_L,
            U_i_R,
            U_n_L,
            U_n_R,
            nu_in=nu_in,
            gamma=gamma,
            cfl=cfl,
            use_strang=use_strang
        )

        substeps_i_hist.append(nsub_i)
        substeps_n_hist.append(nsub_n)

        total_substeps_i += nsub_i
        total_substeps_n += nsub_n

        t += macro_dt
        macro_step += 1

    if macro_step >= max_macro_steps:
        print("Warning: max_macro_steps reached in ADI TVD run")

    rho_i, u_i, p_i, rho_n, u_n, p_n = state_to_prim_two_fluid_E(
        U_i,
        U_n,
        gamma=gamma
    )

    rho_tot = rho_i + rho_n
    m_tot = U_i[1] + U_n[1]
    u_tot = m_tot / torch.clamp(rho_tot, min=eps)
    p_tot = p_i + p_n

    result = {
        "method": "ADI-like per-fluid TVD",
        "x": x,
        "U_i": U_i,
        "U_n": U_n,

        "rho_i": rho_i,
        "u_i": u_i,
        "p_i": p_i,

        "rho_n": rho_n,
        "u_n": u_n,
        "p_n": p_n,

        "rho_tot": rho_tot,
        "u_tot": u_tot,
        "p_tot": p_tot,

        "t_hist": t_hist,
        "drift_hist": drift_hist,
        "macro_dt_hist": macro_dt_hist,
        "dt_i_hist": dt_i_hist,
        "dt_n_hist": dt_n_hist,

        "substeps_i_hist": substeps_i_hist,
        "substeps_n_hist": substeps_n_hist,

        "nu_in": nu_in,
        "final_time": t,
        "macro_steps": macro_step,

        "total_substeps_i": total_substeps_i,
        "total_substeps_n": total_substeps_n,
        "total_hydro_substeps": total_substeps_i + total_substeps_n,

        "diagnostics": {
            "min_rho_i": torch.min(rho_i).item(),
            "min_rho_n": torch.min(rho_n).item(),
            "min_p_i": torch.min(p_i).item(),
            "min_p_n": torch.min(p_n).item(),
            "mean_drift_final": torch.mean(torch.abs(u_i - u_n)).item(),
            "max_drift_final": torch.max(torch.abs(u_i - u_n)).item(),
        }
    }

    return result


# Step 10D: Plot ADI-like per-fluid TVD result

def plot_ADI_TVD_result(result):
    """
    Plot ion, neutral, and total variables for the ADI-like TVD method.
    """

    x_np = result["x"].detach().cpu().numpy()

    rho_i_np = result["rho_i"].detach().cpu().numpy()
    rho_n_np = result["rho_n"].detach().cpu().numpy()
    rho_tot_np = result["rho_tot"].detach().cpu().numpy()

    u_i_np = result["u_i"].detach().cpu().numpy()
    u_n_np = result["u_n"].detach().cpu().numpy()
    u_tot_np = result["u_tot"].detach().cpu().numpy()

    p_i_np = result["p_i"].detach().cpu().numpy()
    p_n_np = result["p_n"].detach().cpu().numpy()
    p_tot_np = result["p_tot"].detach().cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].plot(x_np, rho_i_np, label="rho_i")
    axes[0, 0].plot(x_np, rho_n_np, label="rho_n")
    axes[0, 0].plot(x_np, rho_tot_np, linewidth=2, label="rho_total")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("density")
    axes[0, 0].set_title("Density")
    axes[0, 0].grid()
    axes[0, 0].legend()

    axes[0, 1].plot(x_np, u_i_np, label="u_i")
    axes[0, 1].plot(x_np, u_n_np, label="u_n")
    axes[0, 1].plot(x_np, u_tot_np, linewidth=2, label="u_total")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("velocity")
    axes[0, 1].set_title("Velocity")
    axes[0, 1].grid()
    axes[0, 1].legend()

    axes[1, 0].plot(x_np, p_i_np, label="p_i")
    axes[1, 0].plot(x_np, p_n_np, label="p_n")
    axes[1, 0].plot(x_np, p_tot_np, linewidth=2, label="p_total")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("pressure")
    axes[1, 0].set_title("Pressure")
    axes[1, 0].grid()
    axes[1, 0].legend()

    axes[1, 1].plot(result["t_hist"], result["drift_hist"])
    axes[1, 1].set_xlabel("t")
    axes[1, 1].set_ylabel("mean |u_i - u_n|")
    axes[1, 1].set_title("Velocity drift")
    axes[1, 1].grid()

    fig.suptitle(
        f"ADI-like per-fluid TVD method, nu_in = {result['nu_in']}",
        fontsize=14
    )

    save_figure(fig, f"40_ADI_TVD_result_nu_{result['nu_in']}.png")
    plt.show()
    plt.close(fig)



# Step 10E: Compare global TVD OS and ADI-like per-fluid TVD

def compare_global_TVD_and_ADI_TVD(nu_in,
                                   N=1000,
                                   t_end=0.08,
                                   ion_frac_L=0.05,
                                   ion_frac_R=0.05,
                                   press_frac_i=0.5,
                                   drift_n_L=0.0,
                                   drift_n_R=0.0,
                                   cfl=0.35,
                                   macro_factor=1.0,
                                   use_strang=True):
    """
    Compare two methods:

        1. Global TVD OS:
           dt = min(dt_i, dt_n)

        2. ADI-like per-fluid TVD:
           each fluid evolves with its own CFL substeps
           over a macro time interval.

    The speed-up is measured in two ways:

        a) hydro update count speed-up
        b) wall-clock runtime speed-up
    """

    import time


    # Global TVD OS method
  
    tic = time.perf_counter()

    result_global = run_two_fluid_sod_case_TVD_OS_E(
        nu_in=nu_in,
        N=N,
        t_end=t_end,
        ion_frac_L=ion_frac_L,
        ion_frac_R=ion_frac_R,
        press_frac_i=press_frac_i,
        drift_n_L=drift_n_L,
        drift_n_R=drift_n_R,
        cfl=cfl
    )

    time_global = time.perf_counter() - tic

    # In every global step, both fluids are updated once.
    global_hydro_updates = 2 * result_global["steps"]

  
    # ADI-like per-fluid TVD method

    tic = time.perf_counter()

    result_adi = run_two_fluid_sod_case_ADI_TVD_E(
        nu_in=nu_in,
        N=N,
        t_end=t_end,
        ion_frac_L=ion_frac_L,
        ion_frac_R=ion_frac_R,
        press_frac_i=press_frac_i,
        drift_n_L=drift_n_L,
        drift_n_R=drift_n_R,
        cfl=cfl,
        macro_factor=macro_factor,
        use_strang=use_strang
    )

    time_adi = time.perf_counter() - tic

    adi_hydro_updates = result_adi["total_hydro_substeps"]

    # Difference between final results
  
    L1_diff_rho = torch.mean(
        torch.abs(result_global["rho_tot"] - result_adi["rho_tot"])
    ).item()

    L1_diff_u = torch.mean(
        torch.abs(result_global["u_tot"] - result_adi["u_tot"])
    ).item()

    L1_diff_p = torch.mean(
        torch.abs(result_global["p_tot"] - result_adi["p_tot"])
    ).item()

    update_speedup = global_hydro_updates / max(adi_hydro_updates, 1)
    runtime_speedup = time_global / max(time_adi, 1e-15)

    summary = {
        "nu_in": nu_in,
        "N": N,
        "ion_frac_L": ion_frac_L,
        "ion_frac_R": ion_frac_R,

        "global_steps": result_global["steps"],
        "global_hydro_updates": global_hydro_updates,
        "global_runtime": time_global,

        "adi_macro_steps": result_adi["macro_steps"],
        "adi_ion_substeps": result_adi["total_substeps_i"],
        "adi_neutral_substeps": result_adi["total_substeps_n"],
        "adi_hydro_updates": adi_hydro_updates,
        "adi_runtime": time_adi,

        "update_speedup": update_speedup,
        "runtime_speedup": runtime_speedup,

        "L1_diff_rho_total": L1_diff_rho,
        "L1_diff_u_total": L1_diff_u,
        "L1_diff_p_total": L1_diff_p,

        "global_mean_drift": result_global["diagnostics"]["mean_drift_final"],
        "adi_mean_drift": result_adi["diagnostics"]["mean_drift_final"],

        "global_min_rho_i": result_global["diagnostics"]["min_rho_i"],
        "global_min_rho_n": result_global["diagnostics"]["min_rho_n"],
        "global_min_p_i": result_global["diagnostics"]["min_p_i"],
        "global_min_p_n": result_global["diagnostics"]["min_p_n"],

        "adi_min_rho_i": result_adi["diagnostics"]["min_rho_i"],
        "adi_min_rho_n": result_adi["diagnostics"]["min_rho_n"],
        "adi_min_p_i": result_adi["diagnostics"]["min_p_i"],
        "adi_min_p_n": result_adi["diagnostics"]["min_p_n"],
    }

    return result_global, result_adi, summary



# Step 10F: Plot global TVD OS versus ADI-like TVD

def plot_global_TVD_vs_ADI_TVD(result_global, result_adi):
    """
    Plot total variables from global TVD OS and ADI-like per-fluid TVD.
    """

    x_np = result_global["x"].detach().cpu().numpy()

    rho_g = result_global["rho_tot"].detach().cpu().numpy()
    u_g = result_global["u_tot"].detach().cpu().numpy()
    p_g = result_global["p_tot"].detach().cpu().numpy()

    rho_a = result_adi["rho_tot"].detach().cpu().numpy()
    u_a = result_adi["u_tot"].detach().cpu().numpy()
    p_a = result_adi["p_tot"].detach().cpu().numpy()

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    axes[0].plot(x_np, rho_g, label="global TVD OS")
    axes[0].plot(x_np, rho_a, "--", label="ADI-like per-fluid TVD")
    axes[0].set_ylabel("rho_total")
    axes[0].set_title("Total density")
    axes[0].grid()
    axes[0].legend()

    axes[1].plot(x_np, u_g, label="global TVD OS")
    axes[1].plot(x_np, u_a, "--", label="ADI-like per-fluid TVD")
    axes[1].set_ylabel("u_total")
    axes[1].set_title("Total velocity")
    axes[1].grid()
    axes[1].legend()

    axes[2].plot(x_np, p_g, label="global TVD OS")
    axes[2].plot(x_np, p_a, "--", label="ADI-like per-fluid TVD")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("p_total")
    axes[2].set_title("Total pressure")
    axes[2].grid()
    axes[2].legend()

    fig.suptitle("Global TVD OS vs ADI-like per-fluid TVD", fontsize=14)

    save_figure(fig, "41_global_TVD_vs_ADI_TVD.png")
    plt.show()
    plt.close(fig)


# Step 10G: One single ADI test


print("\n==============================")
print("Running one ADI-like TVD test")
print("==============================")

result_adi_test = run_two_fluid_sod_case_ADI_TVD_E(
    nu_in=0.5,
    N=1000,
    t_end=0.08,
    ion_frac_L=0.8,
    ion_frac_R=0.2,
    press_frac_i=0.5,
    drift_n_L=0.2,
    drift_n_R=0.0,
    cfl=0.35,
    macro_factor=1.0,
    use_strang=True
)

print("ADI final time =", result_adi_test["final_time"])
print("ADI macro steps =", result_adi_test["macro_steps"])
print("ADI ion substeps =", result_adi_test["total_substeps_i"])
print("ADI neutral substeps =", result_adi_test["total_substeps_n"])
print("ADI total hydro substeps =", result_adi_test["total_hydro_substeps"])
print("ADI diagnostics:")
print(result_adi_test["diagnostics"])

plot_ADI_TVD_result(result_adi_test)



# Step 10H: Compare global TVD OS and ADI TVD for one case


print("\n==========================================")
print("Comparing global TVD OS and ADI-like TVD")
print("==========================================")

result_global_cmp, result_adi_cmp, summary_cmp = compare_global_TVD_and_ADI_TVD(
    nu_in=0.5,
    N=1000,
    t_end=0.08,
    ion_frac_L=0.05,
    ion_frac_R=0.05,
    press_frac_i=0.5,
    drift_n_L=0.0,
    drift_n_R=0.0,
    cfl=0.35,
    macro_factor=1.0,
    use_strang=True
)

print("\nComparison summary:")
for key, value in summary_cmp.items():
    print(key, "=", value)

plot_global_TVD_vs_ADI_TVD(result_global_cmp, result_adi_cmp)

df_ADI_one = pd.DataFrame([summary_cmp])

print("\nOne-case ADI speed-up table:")
print(df_ADI_one[[
    "nu_in",
    "ion_frac_L",
    "global_steps",
    "global_hydro_updates",
    "adi_macro_steps",
    "adi_ion_substeps",
    "adi_neutral_substeps",
    "adi_hydro_updates",
    "update_speedup",
    "runtime_speedup",
    "L1_diff_rho_total",
    "L1_diff_u_total",
    "L1_diff_p_total",
    "global_mean_drift",
    "adi_mean_drift"
]])


# Step 10I: Speed-up study for different density splits


print("\n======================================")
print("ADI speed-up study: density contrast")
print("======================================")

ion_frac_list_ADI = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]

ADI_speedup_summaries = []

for ion_frac in ion_frac_list_ADI:

    print(f"\nRunning ADI speed-up case: ion_frac = {ion_frac}")

    result_global, result_adi, summary = compare_global_TVD_and_ADI_TVD(
        nu_in=0.5,
        N=500,
        t_end=0.08,
        ion_frac_L=ion_frac,
        ion_frac_R=ion_frac,
        press_frac_i=0.5,
        drift_n_L=0.0,
        drift_n_R=0.0,
        cfl=0.35,
        macro_factor=1.0,
        use_strang=True
    )

    ADI_speedup_summaries.append(summary)

    print("global hydro updates =", summary["global_hydro_updates"])
    print("ADI hydro updates    =", summary["adi_hydro_updates"])
    print("update speed-up      =", summary["update_speedup"])
    print("runtime speed-up     =", summary["runtime_speedup"])
    print("L1 diff rho          =", summary["L1_diff_rho_total"])
    print("L1 diff u            =", summary["L1_diff_u_total"])
    print("L1 diff p            =", summary["L1_diff_p_total"])


df_ADI_speedup = pd.DataFrame(ADI_speedup_summaries)

print("\nADI density-contrast speed-up table:")
print(df_ADI_speedup[[
    "ion_frac_L",
    "global_steps",
    "global_hydro_updates",
    "adi_macro_steps",
    "adi_ion_substeps",
    "adi_neutral_substeps",
    "adi_hydro_updates",
    "update_speedup",
    "runtime_speedup",
    "L1_diff_rho_total",
    "L1_diff_u_total",
    "L1_diff_p_total"
]])



# Step 10J: Save ADI speed-up table


ADI_table_path = fig_dir / "ADI_speedup_density_contrast_table.csv"
df_ADI_speedup.to_csv(ADI_table_path, index=False)
print(f"Saved ADI speed-up table: {ADI_table_path}")

ADI_latex_path = fig_dir / "ADI_speedup_density_contrast_table.tex"

df_ADI_speedup.to_latex(
    ADI_latex_path,
    index=False,
    float_format="%.4e",
    caption="Speed-up comparison between global TVD operator splitting and ADI-like per-fluid TVD method.",
    label="tab:adi_speedup_density_contrast"
)

print(f"Saved ADI LaTeX table: {ADI_latex_path}")



# Step 10K: Plot ADI speed-up results




fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Speed-up
axes[0, 0].plot(
    df_ADI_speedup["ion_frac_L"],
    df_ADI_speedup["update_speedup"],
    marker="o",
    label="hydro update speed-up"
)

axes[0, 0].plot(
    df_ADI_speedup["ion_frac_L"],
    df_ADI_speedup["runtime_speedup"],
    marker="s",
    label="runtime speed-up"
)

axes[0, 0].set_xscale("log")
axes[0, 0].set_xlabel("ion density fraction")
axes[0, 0].set_ylabel("speed-up")
axes[0, 0].set_title("ADI speed-up")
axes[0, 0].grid()
axes[0, 0].legend()

# Hydro update counts
axes[0, 1].plot(
    df_ADI_speedup["ion_frac_L"],
    df_ADI_speedup["global_hydro_updates"],
    marker="o",
    label="global TVD OS"
)

axes[0, 1].plot(
    df_ADI_speedup["ion_frac_L"],
    df_ADI_speedup["adi_hydro_updates"],
    marker="s",
    label="ADI-like TVD"
)

axes[0, 1].set_xscale("log")
axes[0, 1].set_yscale("log")
axes[0, 1].set_xlabel("ion density fraction")
axes[0, 1].set_ylabel("hydro update count")
axes[0, 1].set_title("Hydro update count")
axes[0, 1].grid()
axes[0, 1].legend()

# ADI substeps
axes[1, 0].plot(
    df_ADI_speedup["ion_frac_L"],
    df_ADI_speedup["adi_ion_substeps"],
    marker="o",
    label="ion substeps"
)

axes[1, 0].plot(
    df_ADI_speedup["ion_frac_L"],
    df_ADI_speedup["adi_neutral_substeps"],
    marker="s",
    label="neutral substeps"
)

axes[1, 0].set_xscale("log")
axes[1, 0].set_yscale("log")
axes[1, 0].set_xlabel("ion density fraction")
axes[1, 0].set_ylabel("number of substeps")
axes[1, 0].set_title("ADI per-fluid substeps")
axes[1, 0].grid()
axes[1, 0].legend()

# Difference between methods
axes[1, 1].plot(
    df_ADI_speedup["ion_frac_L"],
    df_ADI_speedup["L1_diff_rho_total"],
    marker="o",
    label="L1 diff rho"
)

axes[1, 1].plot(
    df_ADI_speedup["ion_frac_L"],
    df_ADI_speedup["L1_diff_u_total"],
    marker="s",
    label="L1 diff u"
)

axes[1, 1].plot(
    df_ADI_speedup["ion_frac_L"],
    df_ADI_speedup["L1_diff_p_total"],
    marker="^",
    label="L1 diff p"
)

axes[1, 1].set_xscale("log")
axes[1, 1].set_yscale("log")
axes[1, 1].set_xlabel("ion density fraction")
axes[1, 1].set_ylabel("L1 difference")
axes[1, 1].set_title("Difference between global and ADI")
axes[1, 1].grid()
axes[1, 1].legend()

fig.suptitle("ADI-like per-fluid TVD speed-up study", fontsize=14)

save_figure(fig, "42_ADI_TVD_speedup_density_contrast.png")
plt.show()
plt.close(fig)



# Step 10L: Compare ADI TVD with exact single-fluid solution


print("\n=======================================")
print("ADI TVD exact single-fluid comparison")
print("=======================================")

result_adi_exact = run_two_fluid_sod_case_ADI_TVD_E(
    nu_in=100.0,
    N=1000,
    t_end=0.08,
    ion_frac_L=0.5,
    ion_frac_R=0.5,
    press_frac_i=0.5,
    drift_n_L=0.0,
    drift_n_R=0.0,
    cfl=0.35,
    macro_factor=1.0,
    use_strang=True
)

result_adi_exact = add_exact_comparison_to_result(
    result_adi_exact,
    x0=0.0,
    t_end=0.08,
    gamma=gamma
)

print("ADI exact comparison:")
print("final time =", result_adi_exact["final_time"])
print("macro steps =", result_adi_exact["macro_steps"])
print("errors exact =", result_adi_exact["errors_exact"])
print("diagnostics =", result_adi_exact["diagnostics"])


def plot_ADI_TVD_exact_comparison(result):
    """
    Plot ADI-like TVD total variables against single-fluid exact solution.
    """

    x_np = result["x"].detach().cpu().numpy()

    rho_tot_np = result["rho_tot"].detach().cpu().numpy()
    u_tot_np = result["u_tot"].detach().cpu().numpy()
    p_tot_np = result["p_tot"].detach().cpu().numpy()

    rho_ex_np = result["rho_ex"].detach().cpu().numpy()
    u_ex_np = result["u_ex"].detach().cpu().numpy()
    p_ex_np = result["p_ex"].detach().cpu().numpy()

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    axes[0].plot(x_np, rho_tot_np, label="ADI TVD total density")
    axes[0].plot(x_np, rho_ex_np, "--", label="single-fluid exact")
    axes[0].set_ylabel("density")
    axes[0].set_title("Density")
    axes[0].grid()
    axes[0].legend()

    axes[1].plot(x_np, u_tot_np, label="ADI TVD total velocity")
    axes[1].plot(x_np, u_ex_np, "--", label="single-fluid exact")
    axes[1].set_ylabel("velocity")
    axes[1].set_title("Velocity")
    axes[1].grid()
    axes[1].legend()

    axes[2].plot(x_np, p_tot_np, label="ADI TVD total pressure")
    axes[2].plot(x_np, p_ex_np, "--", label="single-fluid exact")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("pressure")
    axes[2].set_title("Pressure")
    axes[2].grid()
    axes[2].legend()

    fig.suptitle("ADI-like per-fluid TVD vs single-fluid exact solution", fontsize=14)

    save_figure(fig, "43_ADI_TVD_vs_exact_singlefluid.png")
    plt.show()
    plt.close(fig)


plot_ADI_TVD_exact_comparison(result_adi_exact)



# Step 10M: Final ADI table for report


df_ADI_exact = pd.DataFrame([{
    "nu_in": result_adi_exact["nu_in"],
    "macro_steps": result_adi_exact["macro_steps"],
    "ion_substeps": result_adi_exact["total_substeps_i"],
    "neutral_substeps": result_adi_exact["total_substeps_n"],
    "total_hydro_substeps": result_adi_exact["total_hydro_substeps"],
    "L1_rho": result_adi_exact["errors_exact"]["L1_rho"],
    "L1_u": result_adi_exact["errors_exact"]["L1_u"],
    "L1_p": result_adi_exact["errors_exact"]["L1_p"],
    "Linf_rho": result_adi_exact["errors_exact"]["Linf_rho"],
    "Linf_u": result_adi_exact["errors_exact"]["Linf_u"],
    "Linf_p": result_adi_exact["errors_exact"]["Linf_p"],
    "mean_drift_final": result_adi_exact["diagnostics"]["mean_drift_final"],
    "max_drift_final": result_adi_exact["diagnostics"]["max_drift_final"],
}])

print("\nADI exact-comparison table:")
print(df_ADI_exact)

ADI_exact_latex_path = fig_dir / "ADI_TVD_exact_comparison_table.tex"

df_ADI_exact.to_latex(
    ADI_exact_latex_path,
    index=False,
    float_format="%.4e",
    caption="ADI-like per-fluid TVD method compared with the single-fluid exact Sod solution.",
    label="tab:adi_tvd_exact_comparison"
)

print(f"Saved ADI exact-comparison LaTeX table: {ADI_exact_latex_path}")



# Step 10N: Short final message


print("\n====================================")
print("ADI-like per-fluid TVD section done")
print("Generated figures:")
print("  40_ADI_TVD_result_nu_*.png")
print("  41_global_TVD_vs_ADI_TVD.png")
print("  42_ADI_TVD_speedup_density_contrast.png")
print("  43_ADI_TVD_vs_exact_singlefluid.png")
print("Generated tables:")
print("  ADI_speedup_density_contrast_table.csv")
print("  ADI_speedup_density_contrast_table.tex")
print("  ADI_TVD_exact_comparison_table.tex")
print("====================================")

# Step 11: Save all norm/error/speed-up tables into one PDF

# This follows the same style as save_figure(fig, filename).
# It does not change previous computations.
# It only collects existing DataFrames and saves them.


from matplotlib.backends.backend_pdf import PdfPages


def format_table_value(value):
    """
    Format numbers in a clean way for PDF tables.
    """
    if isinstance(value, (float, np.floating)):
        return f"{value:.4e}"
    if isinstance(value, (int, np.integer)):
        return str(value)
    return str(value)


def save_dataframe_to_pdf_pages(pdf, df, title, max_cols_per_page=7):
    """
    Save one pandas DataFrame into the opened PdfPages object.

    If the table has too many columns, split it over several pages.
    """

    if df is None:
        return

    if len(df) == 0:
        return

    columns = list(df.columns)

    column_blocks = [
        columns[i:i + max_cols_per_page]
        for i in range(0, len(columns), max_cols_per_page)
    ]

    for block_id, cols in enumerate(column_blocks, start=1):
        df_block = df[cols].copy()

        cell_text = []
        for _, row in df_block.iterrows():
            cell_text.append([format_table_value(v) for v in row.values])

        fig, ax = plt.subplots(figsize=(16, 8.5))
        ax.axis("off")

        if len(column_blocks) == 1:
            page_title = title
        else:
            page_title = f"{title}  part {block_id}/{len(column_blocks)}"

        ax.set_title(page_title, fontsize=14, pad=20)

        table = ax.table(
            cellText=cell_text,
            colLabels=list(df_block.columns),
            cellLoc="center",
            loc="center"
        )

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.35)

        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def save_all_norm_summary_pdf(fig_dir):
    """
    Collect all existing summary tables and save them into one PDF.
    """

    tables = []

    if "df_drift" in globals():
        tables.append(("Test 1: drift relaxation slope", df_drift))

    if "df_test2" in globals():
        tables.append(("Test 2: early two-fluid Sod errors", df_test2))

    if "df_OS" in globals():
        tables.append(("Rusanov operator-splitting collision study", df_OS))

    if "df_exact_error" in globals():
        tables.append(("Rusanov OS exact-solution comparison", df_exact_error))

    if "df_speedup_one" in globals():
        tables.append(("One-case per-fluid speed-up comparison", df_speedup_one))

    if "df_speedup" in globals():
        tables.append(("Density contrast speed-up study", df_speedup))

    if "df_TVD" in globals():
        tables.append(("TVD collision-frequency study", df_TVD))

    if "df_ADI_one" in globals():
        tables.append(("One-case ADI-like TVD speed-up", df_ADI_one))

    if "df_ADI_speedup" in globals():
        tables.append(("ADI-like TVD density contrast speed-up", df_ADI_speedup))

    if "df_ADI_exact" in globals():
        tables.append(("ADI-like TVD exact single-fluid comparison", df_ADI_exact))

    if len(tables) == 0:
        print("No summary tables found. Nothing saved.")
        return

    pdf_path = fig_dir / "AST5110_norm_error_speedup_summary.pdf"

    with PdfPages(pdf_path) as pdf:

        # Cover page
        fig, ax = plt.subplots(figsize=(16, 8.5))
        ax.axis("off")

        ax.text(
            0.5,
            0.62,
            "AST5110 Two-Fluid Project",
            ha="center",
            va="center",
            fontsize=24,
            weight="bold"
        )

        ax.text(
            0.5,
            0.50,
            "Norm / Error / Drift / Speed-up Summary",
            ha="center",
            va="center",
            fontsize=18
        )

        ax.text(
            0.5,
            0.36,
            "Automatically generated from the Python simulation script.",
            ha="center",
            va="center",
            fontsize=12
        )

        ax.text(
            0.5,
            0.26,
            f"Saved in:\n{fig_dir}",
            ha="center",
            va="center",
            fontsize=9
        )

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # DataFrame pages
        for title, df in tables:
            save_dataframe_to_pdf_pages(
                pdf,
                df,
                title,
                max_cols_per_page=7
            )

    print(f"Saved summary PDF: {pdf_path}")


# Actually save the PDF
save_all_norm_summary_pdf(fig_dir)