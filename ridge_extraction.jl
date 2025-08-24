using LinearAlgebra
using Statistics

# Using absolute difference to calculate the penalty for freq. diff.
function generate_penalty_matrix(frequency_scales, penalty)
    freq_scale = copy(frequency_scales)
    dist_matrix = abs.(freq_scale .- freq_scale') * penalty
    return dist_matrix
end

# Forward penalty
function calculate_accumulated_penalty_energy_forwards(Energy_to_track, penalty_matrix)
    penalised_energy = copy(Energy_to_track)
    for idx_time in 2:size(penalised_energy, 2)
        for idx_freq in 1:size(penalised_energy, 1)
            penalised_energy[idx_freq, idx_time] += minimum(
                penalised_energy[:, idx_time - 1] .+ penalty_matrix[idx_freq, :]
            )
        end
    end
    # Find the smallest index for each col.
    ridge_idx = [argmin(penalised_energy[:, idx_time]) for idx_time in 1:size(penalised_energy, 2)]
    return penalised_energy, ridge_idx
end

# Backward penalty
function calculate_accumulated_penalty_energy_backwards(Energy_to_track, penalty_matrix, penalised_energy_frwd, ridge_idx_frwd)
    pen_e = copy(penalised_energy_frwd)
    e = copy(Energy_to_track)
    ridge_idx_frwd = copy(ridge_idx_frwd)
    for idx_time in size(e, 2)-1:-1:1
        val = pen_e[ridge_idx_frwd[idx_time + 1], idx_time + 1] - e[ridge_idx_frwd[idx_time + 1], idx_time + 1]
        for idx_freq in 1:size(e, 1)
            new_penalty = penalty_matrix[ridge_idx_frwd[idx_time + 1], idx_freq]
            if abs(val - (pen_e[idx_freq, idx_time] + new_penalty)) < eps(Float64)
                ridge_idx_frwd[idx_time] = idx_freq
            end
        end
    end
    return ridge_idx_frwd
end

# Tracking the ridge from both directions
function frwd_bckwd_ridge_tracking(Energy_to_track, penalty_matrix)
    penalised_energy_frwd, ridge_idx_frwd = calculate_accumulated_penalty_energy_forwards(Energy_to_track, penalty_matrix)
    ridge_idx_frwd_bck = calculate_accumulated_penalty_energy_backwards(Energy_to_track, penalty_matrix, penalised_energy_frwd, ridge_idx_frwd)
    return ridge_idx_frwd_bck
end

function smooth_ridge(ridge_idx, window_size=5)
    smoothed_ridge = copy(ridge_idx)
    for i in 1:length(ridge_idx)
        # Set the window to avoid exceeding boundaries
        start = max(1, i - div(window_size, 2))
        stop = min(length(ridge_idx), i + div(window_size, 2))
        # Turning "mean" into int type
        smoothed_ridge[i] = Int(round(mean(ridge_idx[start:stop])))
    end
    return smoothed_ridge
end

# extrac ridges
function extract_fridges(tf_transf, frequency_scales, penalty=2.0, num_ridges=1, BW=25)
    Energy = abs2.(tf_transf)
    dim = size(Energy)
    ridge_idx = zeros(Int, dim[2], num_ridges)
    max_Energy = zeros(dim[2], num_ridges)
    fridge = zeros(dim[2], num_ridges)

    penalty_matrix = generate_penalty_matrix(frequency_scales, penalty)
    eps_val = eps(Float64)
    scale_results = zeros(Float64, dim[2])  # initialize scale_results

    for current_ridge_index in 1:num_ridges
        energy_max = maximum(Energy, dims=1)
        Energy_neg_log_norm = -log.(Energy ./ energy_max .+ eps_val)

        fwrd_results = frwd_bckwd_ridge_tracking(Energy_neg_log_norm, penalty_matrix)

        # smoothing the ridge
        smoothed_fwrd_results = smooth_ridge(fwrd_results, 50)

        ridge_idx[:, current_ridge_index] = Int.(smoothed_fwrd_results)

        for time_idx in 1:dim[2]
            max_Energy[time_idx, current_ridge_index] = Energy[ridge_idx[time_idx, current_ridge_index], time_idx]
        end

        scale_results = frequency_scales[ridge_idx[:, current_ridge_index]]
        fridge[:, current_ridge_index] = scale_results

        for time_idx in 1:dim[2]
            for freq_idx in max(1, ridge_idx[time_idx, current_ridge_index] - BW):min(dim[1], ridge_idx[time_idx, current_ridge_index] + BW)
                Energy[freq_idx, time_idx] = 0
            end
        end
    end
    return max_Energy, ridge_idx, fridge, scale_results
end

using TickTock
using FFTW
using LinearAlgebra
using Statistics
using JLD
using SpecialFunctions
using Plots
using PyCall
include("hermf.jl")
include("nround.jl")

const fft = FFTW.fft

function sqSTFT(x, t, N, h, Dh, trace = false)
	# If x is a matrix, it assigns the number of rows to xrow and the number of 	    columns to xcol.

	# If x is a vector, it assigns the length of x to xrow and sets xcol to 1.
	if ndims(x) != 1 	
		xrow, xcol = size(x) 	
	else	
		xrow = length(x); xcol = 1	
	end

	# Finding the number of rows in the time instant variable
	trow=length(size(t))
	# Checking validity of the input signal and time instant
	if xcol !=1	
		error("X must have only one column!")	
	elseif trow !=1	
		error("T must only have one row!")
	elseif nextpow(2,N) != N && trace	
		println("For a faster computation, N should be a power of two.")	
	end
	

	# Calculating half-window size for FFT 
	N₂ = Int(floor(N/2))
    tcol = length(t)
	# Initiating the size of the smoothing window
	hrow, hcol = size(h)
	# Checking validity of the smoothing window
	# Ensures it only has 1 column and odd number of rows
	if hcol!=1 || rem(hrow,2)==0	
        error("H must be a smoothing window with odd length!")	
    end
	
	# Calculating half-length of the smoothing window
	Lh = Int((hrow-1)/2) 
	# Defining the time step based on the time instant
	if tcol==1	
    	Dt=1	
	else
		Δt=diff(t)
		Mini = minimum(Δt); Maxi = maximum(Δt)
	# Limiting the time step within an epsilon distance
		if Maxi-Mini > eps()	
			error("The time instants must be regularly sampled!")
		else	
			Dt=Mini	
		end
	end
	# tfr and tf3 will be used to store intermediate results 
	# while calculating SFTF and FFT
	tfr = zeros(N,tcol) 
	tf3 = zeros(N,tcol)	
	if trace
		println("Spectrogram and its differentiation.")
	end
	# Measuring the elapsed time
	if trace	
		tick()	
	end
	# Computing STFT and its differentiation
	#Iterate over each column 
	for icol = 1:tcol
	    tᵢ = t[icol]
	# For each time instant, calculate the corresponding indice
	# with the in window size 'N' 		
    	τ = -minimum([nround(N/2)-1,Lh,tᵢ-1]):minimum([nround(N/2)-1,Lh,xrow-tᵢ])	
    	indices= rem.((N.+τ),N).+1
		norm_h=norm(h[Lh.+τ.+1])
	# Calculating the normalized STFT and normalized differentiation value
	# by multipying the input signal with the conjugate of the smoothing window and 
	# differentiation respectively
	# and dividing by the normalization factor
    	tfr[indices,icol] = x[tᵢ.+τ] .* conj(h[Lh.+τ.+1]) / norm_h
    	tf3[indices,icol] = x[tᵢ.+τ] .* conj(Dh[Lh.+τ.+1]) / norm_h
	end

	# Converting the time-domain representation of the signal 
	# to the frequency-domain representation by applying FFT to tfr and tf3
	tfr = fft(tfr,1)
	tf3 = fft(tf3,1)
	tfr_copy=deepcopy(tfr)
	tf3_copy=deepcopy(tf3)
	tfr_vec=vec(tfr_copy)
	tf3_vec=vec(tf3_copy)
	# Finding thelocations in the frequency spectrum
	# where the STFT values are non-zero
	avoid_warn = findall(!iszero,tfr_vec)
	tf3[avoid_warn] = round.(imag(N*tf3[avoid_warn]./tfr[avoid_warn]/(2.0*pi)),RoundNearestTiesUp)
	# Display the elapsed time
	if trace 
		QQ = tock(); println("Elapsed time = " * string(QQ)) 
	end
	if trace 
		println("Synchrosqueezing: ") 
	end

	# Creating a complex matrix to store the result of the Synchrosqueezing transform
	rtfr = zeros(Complex{Float64},N₂,tcol)

	# Calculating the mean energy of the signal
	Ex = mean(abs.(x[minimum(t):maximum(t)]).^2)

	# Defining a threshold for the SST based on the mean energy of the signal
	Threshold = (1.0e-8)*Ex

	if trace
		
		tick()
		
	end
	
	# Calculating the SST
	for icol = 1:tcol
		
		for jcol = 1:N
	# For each element in tfr that exceeds the threshold value
	# calculate the shift index
			if abs(tfr[jcol,icol]) > Threshold
				
				jcolhat = jcol - Int64(real(tf3[jcol,icol]))
	# Ensuring that jcolhat wraps around to the valid range
				jcolhat = rem(rem(jcolhat-1,N)+N,N)+1
	# If the shift index is within a valid range		
	# the corresponding element in rtfr is incremented by the value in tfr
				if jcolhat < N₂+1
					
					rtfr[jcolhat,icol] = rtfr[jcolhat,icol] + tfr[jcol,icol]
					
				end
			end
		end
	end

	if trace
		RR=tok(); println("Elapsed time = " * string(RR))
	end

	return tfr, rtfr
end

np=pyimport.("numpy")
scipy = pyimport("scipy")

H, DH = hermf(301,1,6);
lpfs=load("lpfs.jld","lpfs")
tmp=lpfs[:,34];

tmp2=zeros(1024);
tmp2[1:1000]=tmp
tmp2[1001:end].=tmp[end]
tmp3=(tmp2.-mean(tmp2));
tmp3=tmp3./norm(tmp3)

tfr, rtfr=sqSTFT(tmp3, 1:1024, 2048, H', DH', true)

nbrVoices = 64  
nbr_scales = 1024  
scales = collect(1:size(rtfr, 1))  
# scales = np.exp(np.linspace(np.log(1.51), np.log(622.207), size(rtfr, 1)))


penalty = 0.5
Energy, ridge_idx, fridge, _ = extract_fridges(rtfr, scales, 0.5, 1, 25)

f1=heatmap(abs.(rtfr[1:40,:]),c=:viridis)
plot!(ridge_idx,linestyle=:dash,color=:red,linewidth=3)

f2=plot(lpfs[:,34],legend=false)

f3=heatmap(abs.(tfr[1:40,:]),c=:viridis)

output=plot(f1,f2,f3,layout=grid(3,1,heights=[0.35,0.35,0.3]))
savefig(output,"test.png")