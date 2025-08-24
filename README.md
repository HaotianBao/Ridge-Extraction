# ridge_extraction.jl — Ridge Extraction from Synchrosqueezed STFT

## Overview
This Julia implementation extracts **time–frequency ridges** from the **synchrosqueezed Short-Time Fourier Transform (SST)**. A ridge represents a dominant frequency trajectory over time, capturing the instantaneous frequency of key oscillatory modes in a signal. Extracting and tracking these ridges is crucial in applications such as **speech analysis, music processing, biomedical signal interpretation, and mode decomposition of nonstationary signals**.

The algorithm uses **dynamic programming** to minimize cumulative energy penalties and smooth trajectories, producing stable ridges that align with the strongest structures in the SST.

## Features
- Constructs **penalty matrices** based on frequency differences.
- Performs **forward accumulated energy tracking** (dynamic programming).
- Refines ridge indices using **backward consistency checks**.
- Tracks ridges in **both directions** for robustness.
- Applies **ridge smoothing** via local averaging.
- Extracts **multiple ridges sequentially**, zeroing out energy around previous ridges to avoid duplication.
- Flexible **penalty strength** (`penalty`) and **bandwidth suppression** (`BW`) parameters.
- End-to-end demo included: STFT → SST → ridge extraction → visualization.

## Function Signatures

### Penalty Construction
```julia
generate_penalty_matrix(frequency_scales, penalty) -> Matrix
```
- Builds a matrix of frequency penalties proportional to absolute differences in `frequency_scales`.

### Forward Energy Tracking
```julia
calculate_accumulated_penalty_energy_forwards(Energy_to_track, penalty_matrix) 
    -> penalised_energy, ridge_idx
```
- Dynamic programming (forward pass).  
- Minimizes penalized cumulative energy across time.  
- Returns:
  - `penalised_energy`: accumulated energy surface.  
  - `ridge_idx`: initial ridge path.

### Backward Refinement
```julia
calculate_accumulated_penalty_energy_backwards(Energy_to_track, penalty_matrix,
                                               penalised_energy_frwd, ridge_idx_frwd) 
    -> ridge_idx_refined
```
- Corrects ridge indices backwards in time.  
- Ensures ridge consistency between successive frames.

### Bidirectional Tracking
```julia
frwd_bckwd_ridge_tracking(Energy_to_track, penalty_matrix) -> ridge_idx
```
- Combines forward and backward passes.  
- Produces refined ridge indices across all time steps.

### Ridge Smoothing
```julia
smooth_ridge(ridge_idx, window_size=5) -> smoothed_ridge
```
- Local moving-average smoothing.  
- Window size adjustable.  
- Reduces jitter in ridge trajectories.

### Main Ridge Extraction
```julia
extract_fridges(tf_transf, frequency_scales; penalty=2.0, num_ridges=1, BW=25) 
    -> max_Energy, ridge_idx, fridge, scale_results
```
- Inputs:  
  - `tf_transf`: STFT or SST matrix.  
  - `frequency_scales`: vector of frequency bin values.  
  - `penalty`: cost for frequency jumps (default 2.0).  
  - `num_ridges`: number of ridges to extract (default 1).  
  - `BW`: bandwidth for energy suppression around ridges.  
- Outputs:  
  - `max_Energy`: ridge energy across time.  
  - `ridge_idx`: ridge indices (matrix if multiple ridges).  
  - `fridge`: ridge frequency trajectories.  
  - `scale_results`: scale (frequency) vector for extracted ridge.

### Synchrosqueezed STFT (Included)
```julia
sqSTFT(x, t, N, h, Dh; trace=false) -> tfr, rtfr
```
- Same implementation as `sqSTFT.jl`.  
- Provides the SST needed as input to `extract_fridges`.

## How It Works (Step by Step)
1. **Compute SST**  
   - Use `sqSTFT` to compute sharpened spectrogram `rtfr`.

2. **Form Energy Matrix**  
   - Take squared magnitude of SST:  
     \[
     E = |rtfr|^2
     \]

3. **Penalty Matrix**  
   - Construct matrix where entry `(i,j)` is proportional to  
     \[
     |f_i - f_j| \times penalty
     \]  
     discouraging large frequency jumps.

4. **Forward Pass**  
   - Dynamic programming accumulates minimal energy path across frames, penalized by jumps.

5. **Backward Pass**  
   - Ensures temporal consistency by refining forward ridge indices.

6. **Bidirectional Tracking**  
   - Produces a stable ridge index per frame.

7. **Smoothing**  
   - Applies local averaging (`window_size`) to reduce jitter.

8. **Multiple Ridge Extraction**  
   - After extracting one ridge, suppresses energy within bandwidth `BW` around ridge to allow subsequent ridges to be extracted.

9. **Output**  
   - Returns energy and frequency trajectories (`fridge`) for analysis or plotting.

## Parameter Tips
- **penalty**:  
  - Small penalty → ridge can jump rapidly.  
  - Large penalty → enforces smooth ridge.
- **num_ridges**:  
  - Extract more than one trajectory by setting `num_ridges > 1`.  
- **BW (bandwidth)**:  
  - Larger values suppress more surrounding energy; useful when ridges are close.  
- **window_size (smoothing)**:  
  - Default 5; increase for smoother trajectories.  

## Common Pitfalls & Troubleshooting
- **Empty ridges / noisy results**: Check SST quality and ensure correct Hermite window length.  
- **Jittery ridges**: Increase `window_size` in `smooth_ridge`.  
- **Too many crossings**: Adjust `penalty` upward to enforce smoother trajectories.  
- **Overlap of ridges**: Increase `BW` when extracting multiple ridges.  

## Performance Notes
- Algorithm complexity is proportional to `O(N × T)` where `N` = frequency bins and `T` = frames.  
- Use power-of-two `N` in SST for FFT speed.  
- Ridge extraction itself is lightweight compared to SST computation.  

## Applications
- **Speech analysis**: Extracting formants.  
- **Music**: Tracking harmonics and vibrato.  
- **Biomedical**: Instantaneous frequency of ECG, EEG.  
- **General TF analysis**: Extracting AM–FM components.

## Acknowledgments
- Original synchrosqueezing concepts: Hau-tieng Wu, F. Auger, et al.  
- Ridge extraction dynamic programming ideas adapted for Julia implementation.  
- Julia adaptation and integration: Naoki Saito.  

## License
The original code header mentions confidentiality for SST parts. Verify usage rights before redistribution. For academic work, please cite the original authors.
