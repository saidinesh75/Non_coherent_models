import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # BE QUIET!!!! (info and warnings are not printed)
import numpy as np 
import matplotlib.pyplot as plt
dir_name = "/home/dinesh/Research_work/Non_coherent_models/POMP/POMP_figures/"
plt.rcParams["savefig.directory"] = os.chdir(os.path.dirname(dir_name))

import math
import sys
import numpy.linalg as la
import numpy.matlib
from scipy.io import loadmat
from sklearn.preprocessing import PolynomialFeatures 

rng = np.random.RandomState(seed=None)
import pickle
import torch
from gen_msg_mod_torch import gen_msg_mod_torch
# from amp_demod_torch2 import amp_demod_torch2
from ParallelOMP_demod import ParallelOMP_demod
from ParallelOMP_demodnc import ParallelOMP_demodnc

import pandas as pd

def is_power_of_2(x):
        return (x > 0) and ((x & (x - 1)) == 0)  # '&' id bitwise AND operation

def awgn_channel(in_array, awgn_var, cols,K,rand_seed=None):
    '''
    Adds Gaussian noise to input array

    Real input_array:
        Add Gaussian noise of mean 0 variance awgn_var.

    Complex input_array:
        Add complex Gaussian noise. Indenpendent Gaussian noise of mean 0
        variance awgn_var/2 to each dimension.
    '''
    # y = torch.zeros(list(in_array.size()), dtype=torch.cdouble).to(device) if K>2 else torch.zeros(list(in_array.size())).to(device)
    
    '''
    for c in range(cols):
        input_array = in_array[:,c]
        assert input_array.dim() == 1, 'input array must be one-dimensional'
        assert awgn_var >= 0

        # rng = np.random.RandomState(rand_seed)
        n   = list(input_array.size())[0]

        if K<=2:
            y[:,c] =  input_array + (torch.sqrt(awgn_var/2)*torch.randn(n)).to(device)

        elif K>2:
            noise = (torch.sqrt(awgn_var/2)*(torch.randn(n)+1j* torch.randn(n))).to(device)
            y[:,c] =  input_array + noise

        else:
            raise Exception("Unknown input type '{}'".format(input_array.dtype))
    '''

    assert awgn_var >= 0
    
    if K<=2:
        y =  in_array + (torch.sqrt(awgn_var/2)*torch.randn(list(in_array.size())[0])).to(device)

    elif K>2:
        noise = (torch.sqrt(awgn_var/2)*(torch.randn(list(in_array.size()))+1j* torch.randn(list(in_array.size())))).to(device)

        # For testing purpose
        # noise_csv = pd.read_csv("/home/saidinesh/Research_work/Modulated_SPARCs/debug_csv_files/noise_4.csv", sep=",", header=None)
        # noise = noise_csv.applymap(lambda s: complex(s.replace('i', 'j'))).values
        # noise = torch.from_numpy(noise).type(torch.cdouble).to(device)

        y =  in_array + noise

    else:
        raise Exception("Unknown input type '{}'".format(in_array.dtype))
    return y   

'''def pilot(beta,delim):
    beta_c = beta.clone()
    beta_1 = beta_c[int(delim[0,0]):int(delim[1,0]+1),:]
    beta_1[beta_1.nonzero()]=1
    beta_c[int(delim[0,0]):int(delim[1,0]+1),:] = beta_1
    return beta
'''

device = torch.device('cuda:0') # choose 'cpu' or 'cuda'

data=loadmat("/home/dinesh/Research_work/Modulated_SPARC_codes/MUB_2_6.mat")

A_ = np.array(data['B'])
n,_ = np.shape(A_)  # (64*4160)
N = int(n**2)
A_ = A_[:,:N]
B = torch.from_numpy(A_).type(torch.cdouble).to(device)

A = torch.zeros(n,N).type(torch.cdouble)
for i in range(n):
    A[:,(i*n):(i+1)*n ] = B[:,torch.arange(i,N,n)]

A = A.to(device)

sections = torch.Tensor([8])               # Number of Sections
EbN0_dB = torch.arange(0,11,2)
paths=2
number_of_seeds=math.floor(math.sqrt(n))
randomPhase = 0

cols = 1000
itr = 10

sec_err_ebno = torch.zeros([torch.numel(sections),torch.numel(EbN0_dB)])
block_err_ebno = torch.zeros([torch.numel(sections),torch.numel(EbN0_dB)])
sec_err_ebno1 = torch.zeros([torch.numel(sections),torch.numel(EbN0_dB)])
block_err_ebno1 = torch.zeros([torch.numel(sections),torch.numel(EbN0_dB)])

for l in range(torch.numel(sections)):
    L = int(sections[l].item())
    M = int(N/L)
    P = L/n
    K = 4
    if L==1:
        number_of_paths=L
    else:
        number_of_paths=number_of_seeds

    code_params   = {'P': P,    # Average codeword symbol power constraint
                    'n': n,     # Rate
                    'L': L,    # Number of sections
                    'M': M,      # Columns per section
                    'dist':0,
                    'modulated':True,
                    'power_allocated':True,
                    'spatially_coupled':False,
                    'dist':0,
                    'K':K,      # Modulation
                    }
    
    delim = torch.zeros([2,L]).to(device)
    delim[0,0] = 0
    delim[1,0] = M-1

    for i in range(1,L):
        delim[0,i] = delim[1,i-1]+1
        delim[1,i] = delim[1,i-1]+M

    # if randomPhase == 2 and L>1:
    #     for ii in range(1,L):
    #         phase = torch.tensor((ii-1)*torch.pi/(2*L))
    #         A[:,int(delim[0,ii]):int(delim[1,ii])] = torch.mul(A[:,int(delim[0,ii]):int(delim[1,ii])],torch.exp(1j*phase))
    BTB = (A.T.conj()).mm(A)

    for e in range(torch.numel(EbN0_dB)):
        code_params.update({'EbNo_dB':EbN0_dB[e]})
        print("Running for L = {l} and Eb/N0 = {e}".format(l=sections[l], e=EbN0_dB[e]))
        K = code_params['K'] if code_params['modulated'] else 1

        decode_params = {'t_max':20 ,'rtol':1e-5}
        Eb_No_linear = torch.pow(10, torch.divide(EbN0_dB[e],10)) 

        bit_len = int(round(L*math.log2(K*M)))
        logM = int(round(math.log2(M)))
        logK = int(round(math.log2(K)))
        sec_size = logM + logK

        R = bit_len/n  # Rate
        Eb = n*P/bit_len
        # Eb = L/bit_len
        awgn_var = Eb/Eb_No_linear
        sigma = math.sqrt(awgn_var)
        code_params.update({'awgn_var':awgn_var})
        snr_rx = P/awgn_var
        capacity = 0.5 * math.log2(1 + snr_rx)

        R_actual = bit_len / n      # Actual rate
        code_params.update({'n':n, 'R_actual':R_actual})
        
        # initialisation
        sec_err_sum = 0 
        blk_err_sum = 0 
        sec_err_sum1 = 0 
        blk_err_sum1 = 0 

        code_params.update({'no_paths':number_of_paths,
                            'no_seeds':number_of_seeds,
                            })
        
        for p in range(itr):
            beta,beta1,c = gen_msg_mod_torch(code_params,cols)
            # beta_c = pilot(beta,delim)
            beta = beta.to(device)
            beta1 = beta1.to(device)
            c = c.to(device)

            # x = torch.mm(A,beta)
            # y = awgn_channel(x,awgn_var,cols,K,rand_seed=None)

            x1 = torch.mm(A,beta1)
            # y1 = awgn_channel(x1,awgn_var,cols,K,rand_seed=None)

            # sec_err_rate, blk_err_rate = ParallelOMP_demod(beta, y, A, code_params, c, delim, cols, BTB)
            # sec_err_sum = sec_err_sum + sec_err_rate
            # blk_err_sum = blk_err_sum + blk_err_rate

            sec_err_rate1, blk_err_rate1 = ParallelOMP_demodnc(beta1, x1, A, code_params, c, delim, cols, BTB)
            # sec_err_sum1 = sec_err_sum1 + sec_err_rate1
            # blk_err_sum1 = blk_err_sum1 + blk_err_rate1

        sec_err_ebno[l,e] = sec_err_sum/itr
        block_err_ebno[l,e] = blk_err_sum/itr

        # sec_err_ebno1[l,e] = sec_err_sum/itr
        # block_err_ebno1[l,e] = blk_err_sum/itr

np.savetxt("/home/dinesh/Research_work/Non_coherent_models/POMP/POMP_results_csv/SER_1e6.csv", sec_err_ebno.cpu().numpy(), delimiter=",", fmt="%.5e")

fig, ax = plt.subplots()
ax.plot(EbN0_dB, sec_err_ebno[0,:],label='L=8')
# ax.plot(EbN0_dB, sec_err_ebno1[0,:],label='L=8')
plt.legend(loc="upper right")
ax.set_yscale('log')
ax.set_title('Avg_Section_error_rate vs Eb/N0 for POMP')
ax.set_xlabel('Eb/N0 (in dB)')
ax.set_ylabel('Section error rate (SER)') 
plt.grid(True, which="both") 
plt.savefig("POMP_SER_1e3.png")   

print("Done")