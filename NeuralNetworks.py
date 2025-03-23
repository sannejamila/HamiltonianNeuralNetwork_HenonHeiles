import numpy as np
import torch
import torch.nn as nn
import numpy.polynomial.hermite as hermite
from numerical_integration import *


class Sin(nn.Module):
    @staticmethod
    def forward(u):
        return torch.sin(u)
    
class PAU(nn.Module):
    def __init__(self, n_numerator=5, n_denominator=4):
        super(PAU, self).__init__()
        self.n_numerator = n_numerator
        self.n_denominator = n_denominator
        self.numerator = nn.Parameter(torch.randn(n_numerator + 1))  #i = 0,...m
        self.denominator = nn.Parameter(torch.randn(n_denominator)) #i = 1,...n
        

    def forward(self, u):
        num = sum(self.numerator[i] * u**i for i in range(self.n_numerator + 1))
        den = 1 + abs(sum(self.denominator[i] * u**(i+1) for i in range(self.n_denominator)))
        return num / den

class PadeTypeActivation(nn.Module):
    def __init__(self, L=3, M=2):
        #L-M = 0 or 1
        super(PadeTypeActivation, self).__init__()
        self.L = L
        self.M = M
        self.c = nn.Parameter(torch.randn(L + 1)) # i = 0,..L
        
        # Choose polynomial degree to ensure denominator degree = M
        self.degree = M // 2 if M % 2 == 0 else (M - 1) // 2  
        self.polynomial_coeff = np.random.rand(self.degree)

    def polynomial(self,u):
        p = 0
        for i,coef in enumerate(self.polynomial_coeff):
            p += coef*u**(i+1)
        return p
    
    def forward(self, u):
        numerator = sum(self.c[j] * u**j for j in range(self.L + 1))
        denominator = 1 + self.polynomial(u)**2  #Ensures no real roots
        return numerator / torch.tensor(denominator, dtype=u.dtype, device=u.device)
    

class BaseHamiltonianNeuralNetwork(nn.Module):

    def __init__(self, nstates,noutputs = 1,hidden_dim=100, act_1 = Sin(), act_2 = nn.Softplus()):

        super().__init__()
        self.nstates = nstates
        self.noutputs = 1
        self.hidden_dim = hidden_dim
        self.act_1 = act_1
        self.act_2 = act_2

        linear1 = nn.Linear(nstates, hidden_dim) #nstates is input dim
        linear2 = nn.Linear(hidden_dim, hidden_dim)
        linear3 = nn.Linear(hidden_dim, noutputs)

        for lin in [linear1, linear2, linear3]:
            nn.init.orthogonal_(lin.weight)  #Fill the input Tensor with a (semi) orthogonal matrix.
            #nn.init.zeros_(lin.bias) #Initializing bias to 0

        self.model = nn.Sequential(
            linear1,
            self.act_1,
            #Sin(),
            linear2,
            self.act_2,
            #nn.Softplus(),
            linear3,
        )

    def forward(self,u=None):
        return self.model(u)
  


class PseudoHamiltonianNeuralNetwork(nn.Module):
    def __init__(self, nstates, S, Hamiltonian_True=None, Hamiltonian_Grad=None,Hamiltonian_estimated=None, initial_condition_sampler=None,**kwargs):
        super(PseudoHamiltonianNeuralNetwork,self).__init__()
     
        self.S = torch.tensor(S,dtype=torch.float32)
        self.Hamiltonian = None
        self.nstates = nstates
        self.Hamiltonian_True = Hamiltonian_True
        self.Hamiltonian_Grad = Hamiltonian_Grad
        

        if initial_condition_sampler is not None:
            self.initial_condition_sampler = initial_condition_sampler
       
        if Hamiltonian_True is not None:
            if Hamiltonian_Grad is None:
                #We have the true Hamiltonian, but not its gradient
                self.Hamiltonian = Hamiltonian_True
                self.dH = self._dH_hamiltonian_true
            else:
                #We have both the true Hamiltonian and its gradient
                self.Hamiltonian = self._hamiltonian_true
                self.dH = self._grad_hamiltonian_true
        
        elif Hamiltonian_Grad is not None:
            #We have the true gradient but not the true Hamiltonian
            self.dH = self._grad_hamiltonian_true
        else:
            #We do not have the true Hamiltonian or the true gradient
            if Hamiltonian_estimated is not None:
                self.Hamiltonian = Hamiltonian_estimated #HNN
            else:
                self.Hamiltonian = BaseHamiltonianNeuralNetwork(nstates = nstates, act_1 = Sin(), act_2 = nn.Softplus()) #HNN
            self.dH = self._dH_hamiltonian_est
            self.act1 = self.Hamiltonian.act_1
            self.act2 = self.Hamiltonian.act_2

    def _hamiltonian_true(self, u):
        return self.Hamiltonian_true(u).detach()
        
    def _grad_hamiltonian_true(self, u):
        return self.Hamiltonian_Grad(u).detach()
    
    def _dH_hamiltonian_est(self, u):
        #u = u.detach().requires_grad_()
        u = u.requires_grad_()
        return torch.autograd.grad(
            self.Hamiltonian(u).sum(),
            u,
            retain_graph=self.training,
            create_graph=self.training,
        )[0]
    
    def _dH_hamiltonian_true(self, u):
        u = u.detach().requires_grad_()
        return torch.autograd.grad(
            self.Hamiltonian(u).sum(), u, retain_graph=False, create_graph=False
        )[0].detach()
    
    def u_dot(self,u):
        return self.dH(u)@self.S.T
    
    def time_derivative_step(self,integrator,u_start,dt,u_end = None):
        if integrator == "RK4":
            dudt = RK4_time_derivative(self.u_dot,u_start, dt = dt)
        elif integrator == "midpoint":
            dudt = explicit_midpoint_time_derivative(self.u_dot,u_start, dt = dt)
        elif integrator == "symplectic midpoint":
            dudt = symplectic_midpoint_time_derivative(self.u_dot,u_start, dt = dt, u_end = u_end)
        elif integrator == "symplectic euler":
            dudt = symplectic_euler(self.u_dot,u_start,dt = dt)
        return dudt
    
    def simulate_trajectory(self,integrator,t_sample,dt,u0=None,H0=None):
        if u0 is None:
            u0 = self.initial_condition_sampler(H0)
        #Reshaping
        u0 = torch.tensor(u0,dtype = torch.float32)
        u0 = u0.reshape(1,u0.shape[-1])

        t_sample = torch.tensor(t_sample,dtype = torch.float32)
        t_shape = t_sample.shape[-1]

        #Initializing solution 
        u = torch.zeros([t_sample.shape[-1],self.nstates])
        dudt = torch.zeros_like(u)
        #Setting initial conditions
        u[0, :] = u0

        for i in range(t_shape-1):
            dudt[i,:] = self.time_derivative_step(integrator=integrator,u_start = u[i : i + 1, :], dt = dt)
            u[i+1,:] = u[i,:] + dt*dudt[i,:]

        return u,dudt,u0
    
    def generate_trajectories(self,ntrajectories, t_sample,integrator = "midpoint",u0s=None):
        if u0s.any() == None:
            u0s = self.initial_condition_sampler(ntrajectories)
        
        #Reshaping
        u0s = torch.tensor(u0s,dtype = torch.float32)
        u0s = u0s.reshape(ntrajectories, self.nstates)
        t_sample = torch.tensor(t_sample,dtype = torch.float32)
        if len(t_sample.shape) == 1:
                #Reshaping time
                t_sample = np.tile(t_sample, (ntrajectories, 1))

        dt = t_sample[0, 1] - t_sample[0, 0]
        traj_length = t_sample.shape[-1]

        #Initializng u and setting initial conditions
        u = torch.zeros([ntrajectories, traj_length, self.nstates])
        u[:,0,:] = u0s

        for i in range(ntrajectories):
            u[i] = self.simulate_trajectory(integrator = integrator,t_sample = t_sample, u0 = u0s[i],dt=dt)[0]
   
        return u, t_sample