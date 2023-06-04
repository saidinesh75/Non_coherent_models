import torch
device = torch.device('cuda:0')
from scipy.optimize import minimize
from torch import nn
from torch import optim

def pskmodfn(x,c):
    out = torch.zeros(torch.numel(x),dtype=torch.cdouble)
    for i in range(torch.numel(x)):
        out[i] = c(int(x[i]))
    return out 



def argmin(y, A):
    # Calculate the pseudoinverse of A
    A_pinv = torch.pinverse(A)

    # Compute the optimal x using the pseudoinverse
    x = torch.matmul(A_pinv, y)

    return x

def ParallelOMP_demod(beta, y_cols, A, code_params, c, delim, cols, BTB):
    P,L,M,n,K = map(code_params.get, ['P','L','M','n','K'] )
    number_of_paths, number_of_seeds= map(code_params.get, ['no_paths', 'no_seeds'] )

    # innerProducts = (A.T).mm(y_cols)
    # _,seed_loc1=torch.topk(torch.real(innerProducts),number_of_paths,dim=0)
    # _,seed_max_abs_col=torch.topk(torch.abs(innerProducts),number_of_paths,dim=0)

    for q in range(cols):
        y = y_cols[:,q].reshape(-1,1)
        innerProducts = (A.T).matmul(y)
        innerProducts_withSymbols = innerProducts.matmul(c.reshape(1,-1))
        innerProductsLong = (innerProducts_withSymbols.T).reshape(-1,1)

        _,seed_loc1=torch.topk(torch.real(innerProducts),number_of_paths,dim=0)
        _,seed_max_abs_col=torch.topk(torch.abs(innerProducts),number_of_paths,dim=0)        
        _,seed_loc2=torch.topk(torch.abs(innerProductsLong),number_of_paths,dim=0)

        selected_integer_matrix=torch.zeros(L,number_of_paths,dtype=torch.cdouble)
        selected_column_matrix=torch.zeros(L,number_of_paths,dtype=torch.cdouble)
        residue=y.repeat(1,number_of_paths)

        for p in range(number_of_paths):
            rx_columns=[]
            rx_integers=[]
            innerProducts_copy=innerProducts.to(device)
            mask=torch.ones(n**2,1).to(device)
            for jj in range(L):
                if K == 1:
                    break
                else:
                    if jj==0:
                        sel_col=seed_max_abs_col[p]
                    else:
                        sel_col=torch.argmax(torch.abs(innerProducts_copy))

                    rx_columns.append(sel_col)
                    # rx_symbols=torch.solve(A[:,rx_columns],y)
                    rx_symbols = argmin(A[:,rx_columns],y).T
                    residue[:,p]= (y- A[:,rx_columns].matmul(rx_symbols)).reshape([-1,])
                    innerProducts_copy= (A.T).matmul(residue[:,p])
                    block_loc=torch.nonzero(sel_col<=delim[1,:])[0]
                    mask[int(delim[0,block_loc]):int(delim[1,block_loc])]=0
                    innerProducts_copy=torch.mul(innerProducts_copy,mask)

            selected_column_matrix[:,p]=rx_columns
            selected_integer_matrix[:,p]=pskmodfn(rx_symbols,c);        
        min_residue_loc=torch.argmin(torch.norm(residue))
        rx_columns=selected_column_matrix[:,min_residue_loc]
        rx_integers=selected_integer_matrix[:,min_residue_loc]

        [rx_columns,ordr]=torch.sort(rx_columns)
        rx_integers=rx_integers[ordr]

    # return 





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