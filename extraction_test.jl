import Pkg
include("ridge_extraction.jl")
using PyCall
using LinearAlgebra
using SciPy
np=pyimport.("numpy")
pywt=pyimport("pywt")

#test 1
sig_lgth = 500
t_vec=np.linspace(0,10,sig_lgth)

nbr_scales=279

sign_chirp_1 = SciPy.signal.chirp(t_vec, f0=2, f1=10, t1=20, method="linear")
sign_chirp_2 = SciPy.signal.chirp(t_vec, f0=.4, f1=7, t1=20, method="quadratic")

sign_chirp=sign_chirp_1+sign_chirp_2

plot(sign_chirp)

scales=np.exp(np.linspace(np.log(1.51),np.log(622.207),nbr_scales))
cwtmatr, freqs = pywt.cwt(sign_chirp,scales,"cmor2.0-1.0")

penalty=2.0

Energy,ridge_idx,a = extract_fridges(cwtmatr,scales,penalty,2,25)

np.savetxt("Energy_julia.txt", Energy)
np.savetxt("ridge_idx_julia.txt", ridge_idx)
np.savetxt("a_julia.txt", a)