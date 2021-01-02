import numpy as np
from scipy.stats import ortho_group
import argparse
import os

# == tools ===
np.set_printoptions(precision=4, suppress=True)
np.random.seed(40)

def vec_norm2(x):
    return np.sqrt(x.dot(x))

# 利用正交矩阵生成固定秩的矩阵
def genfixedRankMatrix(M=5, N=3, r=3):
    if (r>min(M,N)):
        print('input error')
        return 
    A = np.zeros((M,N))
    A[:r,:N] = np.random.rand(r, N)
    Q3 = np.float32(ortho_group.rvs(dim=M))
    Q4 = np.float32(ortho_group.rvs(dim=N))
    A = Q3.dot(A).dot(Q4)
    return A

# ===== main functions  ==========
def LU(A):
    m, n = A.shape
    if m!=n:
        print('=================================================================')
        print('           LU is mainly used for square matrix!')
        print('=================================================================')
    U = np.zeros((n,n)).astype(np.float); U[:m,:n] = A
    L = np.eye(m,n).astype(np.float)
    P = np.identity(m).astype(np.float)
    for j in range(0, n-1): # col to be ruducted
        # select max |Uij| (i>=j) 
        maxInd = np.argmax(np.abs(U[j:,j])) + j
        if(not(maxInd==j)):
            U[[j, maxInd],j:n] = U[[maxInd,j],j:n]
            L[[j, maxInd],0:j] = L[[maxInd,j],0:j]
            P[[j, maxInd],:] = P[[maxInd,j],:]
        for i in range(min(j+1,m-1), min(n,m)): #row
            L[i][j] = U[i][j] / U[j][j] if U[j][j] != 0 else 0
            for k in range(j, n):
                U[i][k] -= L[i][j]*U[j][k]
    return P, L, U


# 验证PLU分解
def LUVerification(params):
    A, P, L, U = params
    M, N = A.shape
    t_limit = 1e-8
    print('P: \n', P)
    print('L: \n', L)
    print('U: \n', U)
    if M!=N:
        return
    print("P*A:\n", P.dot(A))
    print("L*U:\n", L.dot(U))
    print("norm of (P*A-L*U): ", end='')
    test1 = np.linalg.norm(P.dot(A) - L.dot(U)); print('%0.2e' % test1)
    if (np.abs(test1) < t_limit):
        print('LU succeed!')

# === 正交约化 ====
def Householder(A):
    M, N = A.shape
    count = min(M, N)
    T = np.copy(A).astype(np.float)
    P = np.identity(M).astype(np.float)
    for n in range(count):
        if M-1 <= n:
            break
        x = T[n:,n] # (M-n,)
        e1 = np.zeros_like(x)
        e1[0] += 1
        u = (x - e1*vec_norm2(x)).reshape(-1,1)
        Rk_head = np.identity(M-n) - 2*(u.dot(u.T)) / (u.T.dot(u)) # (M-n, M-n)
        T[n:,n:] = Rk_head.dot(T[n:,n:])
        R = np.identity(M)
        R[n:,n:] = Rk_head #(M, M)
        P = R.dot(P)    
    return P, T

def Given(A):
    M, N = A.shape
    count = min(M,N)
    P = np.identity(M).astype(np.float)
    T = np.copy(A).astype(np.float)
    for n in range(count):
        for m in range(n+1, M):
            if(T[m,n]==0):
                continue
            Pt = np.identity(M)
            factor = 1.0/np.sqrt(T[n,n]*T[n,n] + T[m,n]*T[m,n])
            Pt[n,n] = T[n,n]*factor
            Pt[n,m] = T[m,n]*factor
            Pt[m,n] = -T[m,n]*factor
            Pt[m,m] = T[n,n]*factor
            T = Pt.dot(T)
            P = Pt.dot(P)    
    return P, T



# 验证正交约化
def orthoReductionVerification(params):
    A, P, T = params
    t_limit = 1e-8
    print('P')
    print(P)
    print('T')
    print(T)
    print('norm of (P.T*P-I): ', end='')
    test1 = np.linalg.norm(P.T.dot(P)-np.identity(P.shape[1])); print('%.2e' % test1)
    print('norm of (P*A-T): ', end='')
    test2 = np.linalg.norm(P.dot(A)-T); print('%.2e' % test2)
    if (np.abs(test1) < t_limit) and (np.abs(test2) < t_limit):
        print('reduction succeed!')
    else:
        print('reduction failed!')

# ========= QR 分解 ===============
def QR(A):
    M, N = A.shape
    Q = np.zeros((M,N),dtype=np.float)
    R = np.zeros((N,N), dtype=np.float)
    for c in range(N):
        tc = min(M, c)
        R[:tc, c] = Q[:,:tc].T.dot(A[:, c]) # tc*1 = tc*M * M*1
        v = A[:,c] - np.sum(Q[:,:tc]*R[:tc,c], axis=1) # M*1 = M*1 - M*tc * tc*1
        R[c,c] = np.sqrt(v.dot(v))
        if R[c,c]==0:
            Q[:,c] = A[:,c]/vec_norm2(A[:,c])
        else:
            Q[:,c] = v / R[c,c]
    return Q, R


def QRVerification(params):
    A, Q, R = params
    m, n = A.shape
    t_limit = 1e-8
    print("Q")
    print(Q)
    print("R")
    print(R)
    print("norm of (A - Q*R): ", end='')
    test1 = np.linalg.norm(A - Q.dot(R)); print('%0.2e' % test1)
    test2 = 0
    if (m==n) and (np.linalg.matrix_rank(A)==m):
        print("norm of (Q.T*Q-I): ", end='')
        test2 = np.linalg.norm((Q.T.dot(Q))-np.identity(n)); print('%0.2e' % test2)
    if (test1 < t_limit) and (test2 < t_limit):
        print('QR succeed!')
    else:
        print('QR failed!')

# === URV =====
def URV(A):
    M, N = A.shape
    P, Be = Householder(A)
    
    # get the rank
    rs = np.where(np.linalg.norm(Be,axis=1)<1e-14)[0]
    r = rs[0] if rs.size>0 else M

    B = Be[:r, :] # r N
    Q, T = Householder(B.T) # Q N*N  T: N*r
    U = P.T
    R = np.zeros_like(A)
    R[:r, :] = T.T
    V = Q.T
    return U, R, V

def URVVerification(params):
    A, U, R, V = params
    t_limit = 1e-10
    print('U: \n', U)
    print('R: \n', R)
    print('V: \n', V)
    print('norm of (U*R*V.T - A): ', end='')
    test1 = np.linalg.norm(U.dot(R).dot(V.T)-A); print('%0.2e' % test1)
    print('norm of (U.T*U -I ): ', end='')
    test2 = np.linalg.norm( U.T.dot(U) - np.identity(U.shape[1]) ); print('%0.2e' % test2)
    print('norm of (V.T*V -I ): ', end='')
    test3 = np.linalg.norm( V.dot(V.T) - np.identity(V.shape[0])); print('%0.2e' % test3)
    if (test1 < t_limit) and (test2 < t_limit) and (test3 < t_limit):
        print('URV succeed!')
    else:
        print('URV failed!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Helps for set the params at shell")
    parser.add_argument('--data',
                        type=str, 
                        help="decide how to get the matrix", 
                        default="data.txt",
                        choices=["random", "data.txt"])    
    parser.add_argument('--method',
                        type=str,
                        help='the method used for the matrix',
                        default='URV',
                        choices=['LU','Householder', 'Given','QR','URV'])
    parser.add_argument('--rows',
                        type=int,
                        help="the row num of the matrix that generated by random",
                        default=5)
    parser.add_argument('--cols',
                        type=int,
                        help='the col num of the matrix that generated by random',
                        default=4)
    parser.add_argument('--rank',
                        type=int,
                        help='the rank of the matrix that generated by random (can\'t more than cols or rows)',
                        default=None)
    args = parser.parse_args()
    if(args.data == 'data.txt'):
        A = np.loadtxt('data.txt').astype(np.float)
    elif(args.data == 'random'):
        cols = max(2, args.cols)
        rows = max(2, args.rows)
        
        if args.rank is None:
            A = np.random.rand(rows, cols)
        else:
            rank = min([args.cols, args.rows, args.rank])
            A = genfixedRankMatrix(rows, cols, rank)
    else:
        Exception('error! get the matrix wrongly')


    methods = {'LU':LU, 'Householder':Householder, 'Given':Given, 'QR':QR, 'URV':URV}
    verifications = {'LU':LUVerification, 
                        'Householder':orthoReductionVerification, 
                        'Given':orthoReductionVerification, 
                        'QR':QRVerification, 
                        'URV':URVVerification}

    if args.method not in methods.keys():
        Exception('error! get method wrongly')

    print('\n\n')
    print('=====================================   BEGIN ================================================')
    print('test method: ', args.method)
    print('shape of matrix A', A.shape)
    print('matrix A:')
    print(A)
    rst = methods[args.method](A)
    rst = (A,) + rst
    verifications[args.method](rst)
    print('=====================================    END =================================================')
