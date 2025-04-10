
import numpy as np
import torch
from numerical_integration import *
from tqdm import tqdm


class KeplerSystem:
    def __init__(self, seed = 123):
        self.nstates = 4
        self.S = np.array([[0.,0.,1.,0.],[0.,0.,0.,1.],[-1.,0.,0.,0.],[0.,-1.,0.,0.]], dtype=np.float32)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.name_system = "Kepler"


    def Hamiltonian(self,u):
        """The Hamiltonian H of the system. Callable taking a
        torch tensor input of shape (nsamples, nstates) and
        returning a torch tensor of shape (nsamples, 1)."""
     
        A = np.array([[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,1]])
        B = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]])

        if not isinstance(u, np.ndarray):
            if isinstance(A, np.ndarray):
                A = torch.tensor(A)
            if isinstance(B, np.ndarray):
                B = torch.tensor(B)
            A = A.to(u.dtype)
            B = B.to(u.dtype)

        H = 1/2*u.T@A@u - 1/np.sqrt(u.T@B@u)
        return H
   
    
    def Hamiltonian_grad(self,u):
        """The gradient of the Hamiltonian H of the system. Callable
            taking an ndarray input of shape (nsamples, nstates) and
            returning a torch tensor of shape (nsamples, nstates)."""

        u = u.reshape(-1)
        if u.ndim ==1:
            x,y,px,py = u[0],u[1],u[2],u[3]
        else:
            x,y,px,py = u[:,0],u[:,1],u[:,2],u[:,3]
        if isinstance(u, np.ndarray):
            dHdu = np.array([x/(x**2+y**2)**(3/2),y/(x**2+y**2)**(3/2),px,py])
        else: 
            dHdu = torch.tensor([x/(x**2+y**2)**(3/2),y/(x**2+y**2)**(3/2),px,py])
        return dHdu
    

    def Angular_Momentum(self,u):
        A = np.array([[0,0,0,1],[0,0,0,0],[0,-1,0,0],[0,0,0,0]])
        if not isinstance(u, np.ndarray):
            if isinstance(A, np.ndarray):
                A = torch.tensor(A)
            A = A.to(u.dtype)
        L = u@A@u.T
        return L
    
    def L_dot(self,u):
        #u = u.reshape(-1)
        dudt = self.u_dot(u)
      
        if u.ndim ==1:
            x,y,px,py = u[0],u[1],u[2],u[3]
            x_dot,y_dot,px_dot,py_dot = dudt[0],dudt[1],dudt[2],dudt[3]
        else:
            x,y,px,py = u[:,0],u[:,1],u[:,2],u[:,3]
            x_dot,y_dot,px_dot,py_dot = dudt[:,0],dudt[:,1],dudt[:,2],dudt[:,3]
        return py*x_dot-px*y_dot-y*px_dot+x*py_dot
    """
        def L_dot(self,u):
            A = np.array([[0,0,0,1],[0,0,-1,0],[0,-1,0,0],[1,0,0,0]])
            if not isinstance(u, np.ndarray):
                if isinstance(A, np.ndarray):
                    A = torch.tensor(A)
                A = A.to(u.dtype)
            L_dot = u@A@self.u_dot(u).T
            return L_dot
        """
    """
    def initial_condition(self,e=None):
        while True:
            # Sample L0 with a lower bound to avoid chaotic regions
            #L0 = np.random.uniform(0.1, np.sqrt(a))  # Set lower bound > 0 (e.g., 0.1)
            #L0 = np.random.uniform(np.sqrt(-a*H0), np.sqrt(-a*H0))  
            #e = np.sqrt(1 - L0**2 / a)
            H0 = np.random.uniform(-1.5,-1e-3)
            e = np.random.uniform(0.1,0.7)
            a = -1/(2*H0)
            L0 = np.sqrt((e**2-1)/(2*H0))
            
            x0, y0 = a * (1 - e), 0
            px0 = 0
            py0 = L0/x0

            if py0 > 2.0:
                continue

            if L0 > 1.0:
                continue
            
            H = 0.5 * (px0**2 + py0**2) - 1 / np.sqrt(x0**2 + y0**2)
            if np.isclose(H, H0,atol=1e-6): 
                break
        return np.array([x0, y0, px0, py0]).flatten()
    """
    def initial_condition(self):
        while True:
            H0 = np.random.uniform(-1.5, -0.01)
            e = np.random.uniform(0.1, 0.7)
            a = -1 / (2 * H0)

            L0 = np.sqrt((e**2 - 1) / (2 * H0))
            if L0 > 1.0:
                continue

            theta = np.random.uniform(0, 2*np.pi)
            r = a * (1 - e**2) / (1 + e * np.cos(theta))

            x0 = r * np.cos(theta)
            y0 = r * np.sin(theta)

            v = np.sqrt(2 * (1/r + H0))
            vx_dir = -np.sin(theta)
            vy_dir = np.cos(theta)

            px0 = v * vx_dir
            py0 = v * vy_dir

            # Optional: rotate whole orbit
            angle = np.random.uniform(0, 2*np.pi)
            R = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle),  np.cos(angle)]
            ])
            x0, y0 = R @ np.array([x0, y0])
            px0, py0 = R @ np.array([px0, py0])

            # Check Hamiltonian consistency
            H = 0.5 * (px0**2 + py0**2) - 1 / np.sqrt(x0**2 + y0**2)
            if np.isclose(H, H0, atol=1e-5):
                return np.array([x0, y0, px0, py0])

    
    def u_dot(self,u):
        dH = self.Hamiltonian_grad(u.T).T
        u_dot = dH@self.S.T
        return u_dot

    
    def sample_trajectory(self,t,u0= None, integrator = "RK4"):
        if u0 is None:
            u0 = self.initial_condition()

        #Initializing solution and its derivative
        u = np.zeros([t.shape[0],self.nstates])
        dudt = np.zeros_like(u)
        
        #Setting initial conditions
        u[0, :] = u0

        for i, time_step in enumerate(t[:-1]):
            dt = t[i+1]-t[i]
            if integrator == "RK4":
                dudt[i,:] = RK4_time_derivative(self.u_dot,u_start = u[i : i + 1, :], dt = dt)
            elif integrator == "midpoint":
                dudt[i,:] = explicit_midpoint_time_derivative(self.u_dot,u_start = u[i : i +1, :], dt = dt)
            elif integrator == "symplectic midpoint":
                dudt[i,:] = symplectic_midpoint_time_derivative(self.u_dot,u_start = u[i : i +1, :],dt = dt)
            elif integrator == "symplectic euler":
                dudt[i,:] = symplectic_euler(self.u_dot,u_start = u[i : i +1, :],dt = dt)

            u[i+1,:] = u[i,:] + dt*dudt[i,:]

        return u, dudt, t, u0
    

def generate_data(ntrajectories, t_sample,system= KeplerSystem(),integrator = "RK4",true_derivatives = False,H0=None,u0s=None):
    data_type = torch.float32
    #Parameters
    nstates = system.nstates
    traj_length = t_sample.shape[0] 

    #Initializing 
    u = np.zeros((ntrajectories,traj_length,nstates))
    dudt = np.zeros_like(u)
    t = np.zeros((ntrajectories,traj_length))

    u0_ = np.zeros((ntrajectories,nstates))

    for i in tqdm(range(ntrajectories)):
        if u0s is not None:
            u0 = np.array(u0s[i])
            u[i], dudt[i], t[i],u0_[i] = system.sample_trajectory(t = t_sample,u0 = u0,integrator=integrator)
        else:
        
            u[i], dudt[i], t[i],u0_[i] = system.sample_trajectory(t=t_sample,integrator=integrator)
    
    #Reshaping
    dt = torch.tensor([t[0, 1] - t[0, 0]], dtype=data_type)
    u_start = torch.tensor(u[:, :-1], dtype=data_type).reshape(-1, nstates)
    u_end = torch.tensor(u[:, 1:], dtype=data_type).reshape(-1, nstates)
    t_start = torch.tensor(t[:, :-1], dtype=data_type).reshape(-1, 1)
    t_end = torch.tensor(t[:, 1:], dtype=data_type).reshape(-1, 1)
    dt = dt * torch.ones_like(t_start, dtype=data_type)
    u_ex = torch.zeros_like(u_start, dtype=data_type)

    if true_derivatives:
        dudt = torch.tensor(dudt[:, :-1], dtype=data_type).reshape(-1, 1, nstates)
    else:
        dudt = (u_end - u_start).clone().detach() / dt[0, 0]


    return (u_start, u_end, t_start, t_end, dt, u_ex), dudt, u,t, H0, u0_