import torch
device = torch.device('cuda:0')
from scipy.optimize import minimize
from torch import nn
from torch import optim
from scipy.io import loadmat
import numpy as np
import math

# y_data = loadmat("/home/dinesh/Research_work/Non_coherent_models/POMP/rx_signal.mat")

def psdemod(x,c):
    out = torch.zeros(torch.numel(x),dtype=torch.cdouble)
    for i in range(torch.numel(x)):
        out[i] = c(int(x[i]))
    return out 

def argmin(y, A):
    # Calculate the pseudoinverse of A
    A_pinv = torch.pinverse(A).type(torch.cdouble)

    # Compute the optimal x using the pseudoinverse
    x = torch.matmul(A_pinv, y).type(torch.cdouble)

    return x

def pskmod(modulated_signal, num_phases):
    # Compute the phase angles of the modulated signal
    phase_angles = torch.angle(modulated_signal)

    # Normalize the phase angles to [0, 2*pi]
    phase_angles = (phase_angles + 2 * torch.pi) % (2 * torch.pi)

    # Adjust the phase angles by pi/4
    phase_angles += torch.pi / 4

    # Wrap the adjusted phase angles to [0, 2*pi]
    phase_angles = (phase_angles + 2 * torch.pi) % (2 * torch.pi)

    # Compute the demodulated symbols
    symbol_indices = (phase_angles * num_phases / (2 * torch.pi)).long()

    # Perform Gray coding
    symbol_indices ^= (symbol_indices >> 1)

    return symbol_indices

def ParallelOMP_demod(beta, y_cols, A, code_params, c, delim, cols, BTB):
    P,L,M,n,K = map(code_params.get, ['P','L','M','n','K'] )
    number_of_paths, number_of_seeds= map(code_params.get, ['no_paths', 'no_seeds'] )

    # innerProducts = (A.T).mm(y_cols)
    # _,seed_loc1=torch.topk(torch.real(innerProducts),number_of_paths,dim=0)
    # _,seed_max_abs_col=torch.topk(torch.abs(innerProducts),number_of_paths,dim=0)

    rx_columns_final = torch.zeros(L,cols).to(device)
    rx_integers_final = torch.zeros(L,cols).to(device)

    # msg_final = torch.zeros(int(L*M),cols).to(device)
    sec_err = 0
    blk_err = 0
    for q in range(cols):
        y = y_cols[:,q].reshape(-1,1)

        # Using a pre-determined message just for testing
        # y_data = loadmat("/home/dinesh/Research_work/Non_coherent_models/POMP/rx_signal.mat")
        # y_ = np.array(y_data['rx_signal'])
        # y = torch.from_numpy(y_).reshape(-1,1).type(torch.cdouble).to(device)

        innerProducts = (A.T.conj()).matmul(y)
        innerProducts_withSymbols = innerProducts.matmul(c.reshape(1,-1).conj())
        innerProductsLong = (innerProducts_withSymbols.T).reshape(-1,1)
        
        _,seed_loc1=torch.topk(torch.real(innerProducts),number_of_paths,dim=0)
        _,seed_max_abs_col=torch.topk(torch.abs(innerProducts),number_of_paths,dim=0)        
        _,seed_loc2=torch.topk(torch.abs(innerProductsLong),number_of_paths,dim=0)

        selected_integer_matrix=torch.zeros(L,number_of_paths).to(device)
        selected_column_matrix=torch.zeros(L,number_of_paths).to(device)
        residue=y.repeat(1,number_of_paths)
        
        for p in range(number_of_paths):
            rx_columns=[]
            rx_integers=[]
            innerProducts_copy=innerProducts.to(device)
            mask=torch.ones(n**2,1).to(device)
            for jj in range(L):
                if K == 0:
                    assert "Code has been written only for Modulated SPARCs"
                else:
                    if jj==0:
                        sel_col=seed_max_abs_col[p]
                    else:
                        sel_col=torch.argmax(torch.abs(innerProducts_copy))

                    rx_columns.append(sel_col.reshape(1))
                    rx_symbols = argmin(y,A[:,rx_columns])
                    residue[:,p]= (y- A[:,rx_columns].matmul(rx_symbols)).reshape([-1,])
                    innerProducts_copy= (A.T.conj()).matmul(residue[:,p].reshape([-1,1]))
                    block_loc=torch.nonzero(sel_col<=delim[1,:])[0]
                    mask[int(delim[0,block_loc]):int(delim[1,block_loc])]=0
                    innerProducts_copy=torch.mul(innerProducts_copy,mask)

            selected_column_matrix[:,p]=torch.cat(rx_columns)
            selected_integer_matrix[:,p]=pskmod(rx_symbols,K).reshape([-1,])   
        min_residue_loc=torch.argmin(torch.norm(residue,dim=0))
        rx_columns=selected_column_matrix[:,min_residue_loc]
        rx_integers=selected_integer_matrix[:,min_residue_loc]

        [rx_columns_final[:,q],ordr]=torch.sort(rx_columns)
        rx_integers_final[:,q] = rx_integers[ordr]
        
        error = torch.numel((~torch.eq(rx_columns_final[:,q],beta[:,q].nonzero().reshape(-1,))).nonzero())
        sec_err = sec_err + error
        blk_err = blk_err + math.ceil(error/L)
        
    sec_err_rate = sec_err/(L*cols)
    blk_err_rate = blk_err/(cols)    
    # print("Done")

    return sec_err_rate, blk_err_rate

    '''
def tsolve(A,b):
    # Split the complex tensors into real and imaginary components
    A_real = torch.real(A)
    A_imag = torch.imag(A)
    b_real = torch.real(b)
    b_imag = torch.imag(b)

    # Solve the equation for the real and imaginary components separately
    x_real = torch.linalg.lstsq(A_real, b_real).solution
    x_imag = torch.linalg.lstsq(A_imag, b_imag).solution

    x = torch.complex(x_real, x_imag)

    return x

def objective(x_np):
    x_tensor = torch.tensor(x_np, dtype=torch.cdouble)
    residual = torch.matmul(A, x_tensor) - y
    return torch.norm(residual).item()

def csolve(A,y):
    x = torch.ones(list(A.size())[1], dtype=torch.cdouble).mul(0.5 + 0.5j)

    criterion = nn.MSELoss()

    # Define the parameters that will be optimized (x_real and x_imag concatenated)
    parameters = torch.tensor(x, requires_grad=True)

    # Define the optimization algorithm
    optimizer = optim.LBFGS([parameters])

    # Function to calculate the loss and gradients
    def closure():
        optimizer.zero_grad()
        output = torch.matmul(A, parameters)
        loss = criterion(output, y)
        loss.backward()
        return loss

    # Optimize the parameters
    optimizer.step(closure)

    # Obtain the optimized values of x
    x_optimized = parameters.detach()
    # Convert the PyTorch tensors to numpy arrays
    return x_optimized

'''