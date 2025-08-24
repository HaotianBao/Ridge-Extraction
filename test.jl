include("ridge_extraction.jl")
using LinearAlgebra
using PyCall
using SciPy
using Test

# Import Python library
push!(PyVector(pyimport("sys")."path"), "")
rt_py = pyimport("ridge_extraction")

np=pyimport.("numpy")
pywt=pyimport("pywt")

sig_lgth = 500
t_vec=np.linspace(0,10,sig_lgth)
f1=0.5
f2=2.0
nbrVoices=32
nbr_scales=279

@testset "test1" begin
    signal_1=np.sin(f1*2*np.pi*t_vec)
    signal_2=np.cos(f2*2*np.pi*t_vec)

    signal=signal_1+signal_2
    scales=np.exp(np.linspace(np.log(1.51),np.log(622.207),nbr_scales))
    cwtmatr, freqs = pywt.cwt(signal,scales,"cmor2.0-1.0")
    penalty=2.0

    tf_transf = cwtmatr
    frequency_scales = scales
    num_ridges=2
    BW=25

    Energy_py=np.square(np.abs(tf_transf))
    dim_py= np.shape(Energy_py)
    ridge_idx_py = np.zeros((dim_py[2],num_ridges), dtype=np.int64)
    max_Energy_py = np.zeros((dim_py[2],num_ridges), dtype=np.float64)
    fridge_py = np.zeros((dim_py[2],num_ridges), dtype=np.float64)

    Energy_jl=abs2.(tf_transf)
    dim_jl=size(Energy_jl)
    ridge_idx_jl = zeros(Int, dim_jl[2], num_ridges)
    max_Energy_jl = zeros(Float64, dim_jl[2], num_ridges)
    fridge_jl = zeros(Float64, dim_jl[2], num_ridges)
    @testset "Initialize" begin
        @test Energy_jl ≈ convert(Array{Float64}, Energy_py)
        @test dim_jl == dim_py
        @info "dim_jl: $dim_jl, dim_py: $dim_py"
        @test ridge_idx_jl ≈ convert(Array{Int64}, ridge_idx_py)
        @test max_Energy_jl ≈ convert(Array{Float64}, max_Energy_py)
        @test fridge_jl ≈ convert(Array{Float64}, fridge_py)
    end

    penalty_matrix_py = np.squeeze(rt_py.generate_penalty_matrix(frequency_scales,penalty))
    penalty_matrix_jl = generate_penalty_matrix(frequency_scales,penalty)
    @testset "penalty_matrix" begin
        @test penalty_matrix_jl ≈ convert(Array{Float64}, penalty_matrix_py)
    end

    eps_py= np.finfo(np.float64).eps
    eps_jl = eps(Float64)

    penalised_energy_frwd_py, ridge_idx_frwd_py = rt_py.calculate_accumulated_penalty_energy_forwards(Energy_py, penalty_matrix_py)
    penalised_energy_frwd_jl, ridge_idx_frwd_jl = calculate_accumulated_penalty_energy_forwards(Energy_jl, penalty_matrix_jl)

    @testset "calculate_accumulated_penalty_energy_forwards" begin
        @test penalised_energy_frwd_jl ≈ convert(Array{Float64}, penalised_energy_frwd_py)
        @test ridge_idx_frwd_jl ≈ convert(Array{Int64}, ridge_idx_frwd_py)

        @test Energy_jl ≈ convert(Array{Float64}, Energy_py)
        @test penalty_matrix_jl ≈ convert(Array{Float64}, penalty_matrix_py)
        @test penalised_energy_frwd_jl ≈ convert(Array{Float64}, penalised_energy_frwd_py)
        @test ridge_idx_frwd_jl ≈ convert(Array{Int64}, ridge_idx_frwd_py)
    end

    ridge_idx_bkwd_jl = calculate_accumulated_penalty_energy_backwards(Energy_jl, penalty_matrix_jl, penalised_energy_frwd_jl, ridge_idx_frwd_jl)
    ridge_idx_bkwd_py = np.array(rt_py.calculate_accumulated_penalty_energy_backwards(Energy_py, penalty_matrix_py, penalised_energy_frwd_py, ridge_idx_frwd_py))

    @testset "calculate_accumulated_penalty_energy_backwards" begin
        @test Energy_jl ≈ convert(Array{Float64}, Energy_py)
        @test penalty_matrix_jl ≈ convert(Array{Float64}, penalty_matrix_py)
        @test penalised_energy_frwd_jl ≈ convert(Array{Float64}, penalised_energy_frwd_py)
        @test penalised_energy_frwd_jl[195, 141] ≈ penalised_energy_frwd_py[195, 141]
        @test penalised_energy_frwd_jl[198, 26] ≈ penalised_energy_frwd_py[198, 26]
        @test ridge_idx_frwd_jl ≈ convert(Array{Int64}, ridge_idx_frwd_py)

        @test ridge_idx_bkwd_jl == convert(Array{Int64}, ridge_idx_bkwd_py)
    end

    @testset "current_ridge_index == 1" begin
        current_ridge_index = 1
        @info "eps_jl: $eps_jl, eps_py: $eps_py"
        @test eps_jl ≈ eps_py
        energy_max_py= np.max(Energy_py,axis=0) 
        energy_max_jl = maximum(Energy_jl, dims=1)
        @test dropdims(energy_max_jl, dims=1) ≈ convert(Array{Float64}, energy_max_py)

        Energy_neg_log_norm_py = -np.log((Energy_jl ./energy_max_jl) .+ eps_py)
        Energy_neg_log_norm_jl = -log.((Energy_jl ./ energy_max_jl) .+ eps_jl)

        @info "shape of Energy_neg_log_norm_jl: $(size(Energy_neg_log_norm_jl)), shape of Energy_neg_log_norm_py: $(size(Energy_neg_log_norm_py))"
        @test Energy_neg_log_norm_jl ≈ convert(Array{Float64}, Energy_neg_log_norm_py)

        fwrd_results_py = np.array(rt_py.frwd_bckwd_ridge_tracking(Energy_neg_log_norm_py,penalty_matrix_py))
        fwrd_results_jl = frwd_bckwd_ridge_tracking(Energy_neg_log_norm_jl,penalty_matrix_jl)
        @info "shape of fwrd_results_jl: $(size(fwrd_results_jl)), shape of fwrd_results_py: $(size(fwrd_results_py))"
        @info "typeof fwrd_results_jl: $(typeof(fwrd_results_jl)), typeof fwrd_results_py: $(typeof(fwrd_results_py))"
        @test fwrd_results_jl == fwrd_results_py

        ridge_idx_jl[:, current_ridge_index] .= fwrd_results_jl
        ridge_idx_jl = convert(Array{Int64}, ridge_idx_jl)

        ridge_idx_py[:,current_ridge_index]=fwrd_results_py
        ridge_idx_py=ridge_idx_py
        @test ridge_idx_jl ≈ convert(Array{Int64}, ridge_idx_py)

    end

    @testset "Final Results 1" begin
        final_Energy_jl, final_ridge_idx_jl, final_a_jl, scale_results_jl = extract_fridges(cwtmatr,scales,penalty,2,25)
        final_Energy_py, final_ridge_idx_py, final_a_py, scale_results_py = rt_py.extract_fridges(cwtmatr,scales,penalty,2,25)

        @test final_Energy_jl ≈ final_Energy_py
        @test final_ridge_idx_jl ≈ final_ridge_idx_py
        @test final_a_jl ≈ final_a_py
        @test scale_results_jl ≈ scale_results_py
    end
end

@testset "Test 2" begin
    sign_chirp_1 = SciPy.signal.chirp(t_vec, f0=2, f1=10, t1=20, method="linear")
    sign_chirp_2 = SciPy.signal.chirp(t_vec, f0=.4, f1=7, t1=20, method="quadratic")

    sign_chirp=sign_chirp_1+sign_chirp_2


    scales=np.exp(np.linspace(np.log(1.51),np.log(622.207),nbr_scales))
    cwtmatr, freqs = pywt.cwt(sign_chirp,scales,"cmor2.0-1.0")
    penalty=.3

    tf_transf = cwtmatr
    frequency_scales = scales
    num_ridges=2
    BW=25

    Energy_py=np.square(np.abs(tf_transf))
    dim_py= np.shape(Energy_py)
    ridge_idx_py = np.zeros((dim_py[2],num_ridges), dtype=np.int64)
    max_Energy_py = np.zeros((dim_py[2],num_ridges), dtype=np.float64)
    fridge_py = np.zeros((dim_py[2],num_ridges), dtype=np.float64)

    Energy_jl=abs2.(tf_transf)
    dim_jl=size(Energy_jl)
    ridge_idx_jl = zeros(Int, dim_jl[2], num_ridges)
    max_Energy_jl = zeros(Float64, dim_jl[2], num_ridges)
    fridge_jl = zeros(Float64, dim_jl[2], num_ridges)
    @testset "Initialize" begin
        @test Energy_jl ≈ convert(Array{Float64}, Energy_py)
        @test dim_jl == dim_py
        @info "dim_jl: $dim_jl, dim_py: $dim_py"
        @test ridge_idx_jl ≈ convert(Array{Int64}, ridge_idx_py)
        @test max_Energy_jl ≈ convert(Array{Float64}, max_Energy_py)
        @test fridge_jl ≈ convert(Array{Float64}, fridge_py)
    end

    penalty_matrix_py = np.squeeze(rt_py.generate_penalty_matrix(frequency_scales,penalty))
    penalty_matrix_jl = generate_penalty_matrix(frequency_scales,penalty)
    @testset "penalty_matrix" begin
        @test penalty_matrix_jl ≈ convert(Array{Float64}, penalty_matrix_py)
    end

    eps_py= np.finfo(np.float64).eps
    eps_jl = eps(Float64)

    penalised_energy_frwd_py, ridge_idx_frwd_py = rt_py.calculate_accumulated_penalty_energy_forwards(Energy_py, penalty_matrix_py)
    penalised_energy_frwd_jl, ridge_idx_frwd_jl = calculate_accumulated_penalty_energy_forwards(Energy_jl, penalty_matrix_jl)

    @testset "calculate_accumulated_penalty_energy_forwards" begin
        @test penalised_energy_frwd_jl ≈ convert(Array{Float64}, penalised_energy_frwd_py)
        @test ridge_idx_frwd_jl ≈ convert(Array{Int64}, ridge_idx_frwd_py)

        @test Energy_jl ≈ convert(Array{Float64}, Energy_py)
        @test penalty_matrix_jl ≈ convert(Array{Float64}, penalty_matrix_py)
        @test penalised_energy_frwd_jl ≈ convert(Array{Float64}, penalised_energy_frwd_py)
        @test ridge_idx_frwd_jl ≈ convert(Array{Int64}, ridge_idx_frwd_py)
    end

    ridge_idx_bkwd_jl = calculate_accumulated_penalty_energy_backwards(Energy_jl, penalty_matrix_jl, penalised_energy_frwd_jl, ridge_idx_frwd_jl)
    ridge_idx_bkwd_py = np.array(rt_py.calculate_accumulated_penalty_energy_backwards(Energy_py, penalty_matrix_py, penalised_energy_frwd_py, ridge_idx_frwd_py))

    @testset "calculate_accumulated_penalty_energy_backwards" begin
        @test Energy_jl ≈ convert(Array{Float64}, Energy_py)
        @test penalty_matrix_jl ≈ convert(Array{Float64}, penalty_matrix_py)
        @test penalised_energy_frwd_jl ≈ convert(Array{Float64}, penalised_energy_frwd_py)
        @test penalised_energy_frwd_jl[195, 141] ≈ penalised_energy_frwd_py[195, 141]
        @test penalised_energy_frwd_jl[198, 26] ≈ penalised_energy_frwd_py[198, 26]
        @test ridge_idx_frwd_jl ≈ convert(Array{Int64}, ridge_idx_frwd_py)

        @test ridge_idx_bkwd_jl == convert(Array{Int64}, ridge_idx_bkwd_py)
    end

    @testset "current_ridge_index == 1" begin
        current_ridge_index = 1
        @info "eps_jl: $eps_jl, eps_py: $eps_py"
        @test eps_jl ≈ eps_py
        energy_max_py= np.max(Energy_py,axis=0) 
        energy_max_jl = maximum(Energy_jl, dims=1)
        @test dropdims(energy_max_jl, dims=1) ≈ convert(Array{Float64}, energy_max_py)

        Energy_neg_log_norm_py = -np.log((Energy_jl ./energy_max_jl) .+ eps_py)
        Energy_neg_log_norm_jl = -log.((Energy_jl ./ energy_max_jl) .+ eps_jl)

        @info "shape of Energy_neg_log_norm_jl: $(size(Energy_neg_log_norm_jl)), shape of Energy_neg_log_norm_py: $(size(Energy_neg_log_norm_py))"
        @test Energy_neg_log_norm_jl ≈ convert(Array{Float64}, Energy_neg_log_norm_py)

        fwrd_results_py = np.array(rt_py.frwd_bckwd_ridge_tracking(Energy_neg_log_norm_py,penalty_matrix_py))
        fwrd_results_jl = frwd_bckwd_ridge_tracking(Energy_neg_log_norm_jl,penalty_matrix_jl)
        @info "shape of fwrd_results_jl: $(size(fwrd_results_jl)), shape of fwrd_results_py: $(size(fwrd_results_py))"
        @info "typeof fwrd_results_jl: $(typeof(fwrd_results_jl)), typeof fwrd_results_py: $(typeof(fwrd_results_py))"
        @test fwrd_results_jl == fwrd_results_py

        ridge_idx_jl[:, current_ridge_index] .= fwrd_results_jl
        ridge_idx_jl = convert(Array{Int64}, ridge_idx_jl)

        ridge_idx_py[:,current_ridge_index]=fwrd_results_py
        ridge_idx_py=ridge_idx_py
        @test ridge_idx_jl ≈ convert(Array{Int64}, ridge_idx_py)

    end

    Energy_jl,ridge_idx_jl,a_jl, scale_results_jl = extract_fridges(cwtmatr,scales,penalty,2,25)
    Energy_py,ridge_idx_py,a_py, scale_results_py = rt_py.extract_fridges(cwtmatr,scales,penalty,2,25)

    @testset "Final Results 2" begin
        @test Energy_jl ≈ convert(Array{Float64}, Energy_py)
        @test maximum(abs.(Energy_jl - convert(Array{Float64}, Energy_py))) < 1e-10
        @test size(Energy_jl) == size(Energy_py)
        @test ridge_idx_jl ≈ convert(Array{Int64}, ridge_idx_py)
        @test maximum(abs.(ridge_idx_jl - convert(Array{Int64}, ridge_idx_py))) < 1e-10
        @test size(ridge_idx_jl) == size(ridge_idx_py)
        @test a_jl ≈ convert(Array{Float64}, a_py)
        @test maximum(abs.(a_jl - convert(Array{Float64}, a_py))) < 1e-10
        @test size(a_jl) == size(a_py)
        @test scale_results_jl ≈ convert(Array{Float64}, scale_results_py)
        @test maximum(abs.(scale_results_jl - convert(Array{Float64}, scale_results_py))) < 1e-10
    end
end
