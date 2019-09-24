import numpy as np
from math import pi
from numba import jit,njit

lastN=None
lastE=None

@jit(parallel=True)
def make_E(N,p):
    d2 = np.array([1,-2,1])
    d_p = 1
    s = 0
    st = np.zeros(N)
    for k in range(1,int(p/2)+1):
        d_p = np.convolve(d2,d_p)
        st[N-k:N]=d_p[0:k]
        st[0:k+1]=d_p[k:]
        st[0]=0
        temp=np.matrix([range(1,k+1),range(1,k+1)])
        temp=np.ravel(temp.T)/range(1,2*k+1)
        s = s + np.power(-1,k-1)*np.prod(temp)*2*st
    
    col=np.matrix(range(0,N)).T
    row=np.matrix(range(N,0,-1))
    idx=col[:,np.zeros(N).astype(int)]+row[np.zeros(N).astype(int),:]-1
    st=np.zeros(2*N-1)
    st[0:N-1] = s[N-1:0:-1]
    st[N-1:]=s
    H=st[idx]+np.diag(np.real(np.fft.fft(s)))

    r=N//2
    V1=(np.eye(N-1)+np.flipud(np.eye(N-1)))/np.sqrt(2)
    V1[N-r-1:,N-r-1:]=-V1[N-r-1:,N-r-1:]
    if N%2==0:
        V1[r-1,r-1]=1
    V=np.eye(N)
    V[1:,1:]=V1  

    VHV=V.dot(H).dot(V.T)
    E=np.zeros((N,N))
    Ev=VHV[:r+1,:r+1]
    Od=VHV[r+1:,r+1:]
    ee,ve=np.linalg.eig(Ev)
    idx=ee.argsort()
    ee=ee[idx]
    ve=ve[:,idx]
    eo,vo=np.linalg.eig(Od)
    idx=eo.argsort()
    eo=eo[idx]
    vo=vo[:,idx]
    
    E[:r+1,:r+1]=np.fliplr(ve)
    E[r+1:N,r+1:N]=np.fliplr(vo)
    E=V.dot(E)

    ind=np.matrix([range(0,r+1),range(r+1,2*r+2)]).T.ravel()
    if N%2==0:
        ind=np.delete(ind,[N-1,N+1])
    else:
        ind=np.delete(ind,N)

    E=E[:,np.ravel(ind)]
    return E

def get_E(N,p):
    global lastN,lastE
    if lastN==None:
        lastN=N
        lastE=make_E(N,p)

    return lastE

@jit(parallel=True)
#f:离散信号 a:阶数 p:近似阶数，默认len(f)/2
def disfrft(f,a,p=-1):
    N=len(f)
    if p==-1:
        p=len(f)/2
    even=0
    if N%2==0:
        even = 1
    shft = (np.array(range(0,N))+int(N/2))%N
    f = np.matrix(f).T
    p = min(max(2,p),N-1)
    E = get_E(N,p)
    #print(E)
    ret = np.zeros(N).astype(complex)
    alist=np.append(range(0,N-1),N-1+even)
    pt1=np.exp(-1j*pi/2*a*alist)
    pt2=np.ravel(E.T.dot(f[shft].astype(complex)))
    ret[shft]=E.dot(pt1*pt2)
    return ret

@jit(parallel=True)
def frft2d(mat,ax,ay):
    m,n=mat.shape
    xspec=np.zeros((m,n)).astype(complex)
    ret=np.zeros((m,n)).astype(complex)
    for i in range(0,m):
        xspec[i,:]=disfrft(mat[i,:],ax)
    for j in range(0,n):
        ret[:,j]=disfrft(xspec[:,j],ay)
    return ret
