'''
Generalized T-matrix code for calculating decoherence of dynamically decoupled electron spin in Droplet Etched GaAs QDs
Author: Leon Zaporski, 21 Oct 2022
Refer to Supplementary Information and Methods for physical parameters used here. 
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.special import erf,owens_t
import math

          
def main():

     def normal(x):
          return (1/math.sqrt(2*math.pi))*np.exp(-x**2/2)

     def normal_cdf(x):
          return 0.5*(1+erf(x/math.sqrt(2)))

     def skew_normal(x,mu,sigma,alpha):
          return (2/sigma)*normal((x-mu)/sigma)*normal_cdf(alpha*((x-mu)/sigma))

     def combined(x,mu1,sigma1,alpha1,mu2,sigma2,q):
          return q*normal((x-mu2)/sigma2)/sigma2+(1-q)*skew_normal(x,mu1,sigma1,alpha1)

     def skew_normal_cdf(x,mu,sigma,alpha):
          return normal_cdf((x-mu)/sigma)-2*owens_t((x-mu)/sigma,alpha)

     def stretched_expo(x,T2,alpha):
          return np.exp(-(x/T2)**alpha)

     #T-Matrix assembly (see Supplementary Information)
     def T_matrix(T,N_pi,A_vec,N_vec,w_vec,w_e,enh_vec,K,K_ga): #wvec-vector of transition frequencies; per single nuc group: (0,w3/2-w1/2,w1/2-w-1/2,w-1/2-w-3/2)
          
          w_kl=np.tile(w_vec,(len(w_vec),1))
          w_kl=w_kl.T-w_kl
          
          A_kl=np.tile(A_vec,(len(A_vec),1))
          A_kl=A_kl.T-A_kl
          
          w_kl_non_div=w_kl
          w_kl_non_div[w_kl_non_div == 0]=1#protection from divergence
          
          prefactors=np.outer(enh_vec,enh_vec)*np.outer(A_vec,A_vec)*np.sqrt(np.outer(N_vec,N_vec))/w_e
          
          T_kl_mat=prefactors*np.divide(w_kl,w_kl_non_div**2-A_kl**2)*(np.ones((4*(K+2*K_ga),4*(K+2*K_ga)))-np.divide(np.cos(A_kl*T/(2*N_pi)),np.cos(w_kl*T/(2*N_pi))))*np.sin(0.5*(w_kl*T+N_pi*math.pi))*np.exp(0.5j*(w_kl*T+N_pi*math.pi))

          for i in range(0,int(4*(K+2*K_ga)/4)): #removing diagonal blocks (self-interactions)
               S=[4*i,4*i+1,4*i+2,4*i+3]
               T_kl_mat[np.ix_(S,S)]=0
     
          return T_kl_mat
     
     ##Free Model Parameters##
     kappa=1.45
     beta=0.35
     #########################
     
     N_pi_sim=[1,3,9] ##considered range of N_pi to simulate

     #Pulsed laser parameters:
     prescale=1 
     Trep=13.13910582 #ns
     
     #time axis limits for simulation. Total sequence times considered are: times=2*N_pi*np.arange(0,up_lim,step)*prescale*Trep
     step=10
     up_lim=500
     
     fig,ax=plt.subplots(len(N_pi_sim))
           
     B_ext=6.5 #Magnetic field (Tesla)
     N=65000 #Number of nuclei
     K=200 #number of nuclear groups per species - see Methods

     angular_conv=(2*math.pi)

     #Hyperfine Constants
     A_Ga69=angular_conv*54.7/(2*math.pi) #GHz 
     A_Ga71=angular_conv*69.9/(2*math.pi) #GHz
     A_As75=angular_conv*65.3/(2*math.pi) #GHz
     
     #concentrations
     c_Ga69=0.604
     c_Ga71=0.396 
     c_As75=1
     
     ###simulated electron zeeman:
     bohr_magneton=5.788381801*0.00001 #eV/T
     h_planck=4.135667696*0.000001 #eV/GHz
     g_eff=0.04895# Electron g factor

     w_e=angular_conv*B_ext*g_eff*bohr_magneton/h_planck #Electron spin splitting


     #Zeeman frequencies
     w_Ga69=angular_conv*0.001*10.22*B_ext #GHz, 
     w_Ga71=angular_conv*0.001*12.98*B_ext #GHz,
     w_As75=angular_conv*0.001*7.22*B_ext #GHz,

     
     #######Scaling factors relative to NMR data (See methods)
     quad_inh_Ga69=-0.5
     quad_inh_Ga71=-0.5
     quad_inh_As75=1
     

     #######Reconstructed Distribution of Quadrupolar Shifts (Voigt Geometry):

     #Arsenic: sub-ensemble (A)
     mu1=angular_conv*0.000001*(252.20809688-246.150313)*kappa*0.5 #Voigt
     sigma1=angular_conv*0.000001*9.73615603*kappa*0.5 #Voigt
     alpha1=angular_conv*0.000001*3.22438213 #Voigt

     #Gallium: sub-ensemble (A)
     mu1_Ga=-0.5*angular_conv*0.000001*(252.20809688-246.150313)*kappa*0.5#Voigt
     sigma1_Ga=0.5*angular_conv*0.000001*9.73615603*kappa*0.5 #Voigt
     alpha1_Ga=-angular_conv*0.000001*3.22438213 #Voigt
     
     #Arsenic: sub-ensemble (B)
     mu2=0
     sigma2=angular_conv*0.000001*73.1236142*kappa #Valid for Voigt and Faraday

     #Considered range of shifts:
     offsets=np.linspace(-5*sigma2,5*sigma2,K)
     bin_width=offsets[1]-offsets[0]
     
     N_groups=[]
     N_ga_groups=[]
     offsets_ga=[]

     tag_index=0
     for offset in offsets:
          xp=offset+0.5*bin_width
          xm=offset-0.5*bin_width

          N_ga_bin=(skew_normal_cdf(xp,mu1_Ga,sigma1_Ga,alpha1_Ga)-skew_normal_cdf(xm,mu1_Ga,sigma1_Ga,alpha1_Ga))
          N_bin=beta*(normal_cdf((xp-mu2)/sigma2)-normal_cdf((xm-mu2)/sigma2))+(1-beta)*(skew_normal_cdf(xp,mu1,sigma1,alpha1)-skew_normal_cdf(xm,mu1,sigma1,alpha1))

          
          if tag_index==0:
               min_N=N_bin

          if N_ga_bin >= min_N:
               N_ga_groups.append(N_ga_bin)
               offsets_ga.append(offset)
          tag_index=tag_index+1
          N_groups.append(N_bin)
          
     
     offsets_ga=np.array(offsets_ga)
     N_groups=np.array(N_groups)
     N_ga_groups=np.array(N_ga_groups)
     K_ga=len(offsets_ga)
        
    
     N_groups=N_groups/np.sum(N_groups)
     N_ga_groups=(N_ga_groups)/np.sum(N_ga_groups)
     
     
     #Assembling broadened transition frequencies:
     def trans_freq_constructor(w,offsets): #Voigt
          ws=[]
          for offs in offsets:
               ws.append(0) 
               ws.append(w-offs) #3/2 to 1/2
               ws.append(w)      #1/2 to -1/2
               ws.append(w+offs) #-1/2 to -3/2
               
          return np.array(ws)
          
     
     w_Ga69_vec=trans_freq_constructor(w_Ga69,offsets_ga)#w_Ga69+offsets
     w_Ga71_vec=trans_freq_constructor(w_Ga71,offsets_ga)#w_Ga71+offsets
     w_As75_vec=trans_freq_constructor(w_As75,offsets)#w_As75+offsets

     w_vec=np.concatenate((w_Ga69_vec,w_Ga71_vec,w_As75_vec))

     ###Numbers of nuclei
     N_Ga69=np.repeat((c_Ga69*N/2)*N_ga_groups,4)
     N_Ga71=np.repeat((c_Ga71*N/2)*N_ga_groups,4)
     N_As75=np.repeat((c_As75*N/2)*N_groups,4)
     
     N_vec=np.concatenate((N_Ga69,N_Ga71,N_As75))

     

     #Hyperfine constant inhomogeneities:
     relative_hyperfine=0.0 #% error

     ###Hyperfine constants
     
     seed=2 #fixing randomization
     np.random.seed(seed)
     A_Ga69_vec=np.random.normal(A_Ga69,relative_hyperfine*A_Ga69,4*K_ga)
     np.random.seed(seed)
     A_Ga71_vec=np.random.normal(A_Ga71,relative_hyperfine*A_Ga71,4*K_ga)
     np.random.seed(seed)
     A_As75_vec=np.random.normal(A_As75,relative_hyperfine*A_As75,4*K)

     A_vec=np.concatenate((A_Ga69_vec,A_Ga71_vec,A_As75_vec))/N
     
     ##enhancement factors:
     enh_vec_single=np.tile([0,np.sqrt(1.5*(1.5+1)-0.5*(0.5+1)),np.sqrt(1.5*(1.5+1)+0.5*(-0.5+1)),np.sqrt(1.5*(1.5+1)+1.5*(-1.5+1))],K)/np.sqrt((2*1.5+1))#(2*1.5+1)
     enh_vec_single_Ga=np.tile([0,np.sqrt(1.5*(1.5+1)-0.5*(0.5+1)),np.sqrt(1.5*(1.5+1)+0.5*(-0.5+1)),np.sqrt(1.5*(1.5+1)+1.5*(-1.5+1))],K_ga)/np.sqrt((2*1.5+1))
     enh_vec=np.concatenate((enh_vec_single_Ga,enh_vec_single_Ga,enh_vec_single))

     ##########
     col_ind=0 #colour index
     for N_pi in N_pi_sim:
          visibilities=[]
          times=2*N_pi*np.arange(0,up_lim,step)*prescale*Trep 
          
          for T in times:
               T_kl=T_matrix(T,N_pi,A_vec,N_vec,w_vec,w_e,enh_vec,K,K_ga) #T-matrix evaluation

               sign, ldet=np.linalg.slogdet(np.eye(len(T_kl))+1j*T_kl)
               det=sign*np.exp(ldet)
               visibilities.append(np.real(1./det))
              
          
          ax[col_ind].set_xlim(0.2,500)#microseconds
          times=2*N_pi*np.arange(0,up_lim,step)*prescale*Trep #total sequence times
          ax[col_ind].plot(times/1000,visibilities,linestyle='-',marker='',c=plt.cm.viridis(col_ind/(len(N_pi_sim)+1)),label=r'$N_\pi=$'+str(N_pi),zorder=2)
          ax[col_ind].axhline(1/math.e,c='gray',linestyle='--',linewidth=0.7)
          ax[col_ind].axhline(0,c='gray',linestyle='-',linewidth=0.7)
          ax[col_ind].set_ylabel('Visibility')
          ax[col_ind].set_yticks([0,1])
          if N_pi !=N_pi_sim[-1]:
               ax[col_ind].set_xticklabels([],color='white')
          else:
               ax[col_ind].set_xlabel('Time ('+r'$\mu$'+'s)')
          ax[col_ind].legend()
          ax[col_ind].set_xscale('log')
          
          col_ind+=1
          np.savetxt('Simulation_CPMG'+str(N_pi)+'.txt', np.array([times/1000,visibilities]).T) #Columns: total sequence times, visibilities
          
     plt.show()
     

if __name__=="__main__":
     main()
