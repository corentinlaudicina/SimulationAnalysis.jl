#=
# OverlapFluctuationFunction — calling guide

All public functions accept an `all_mobilities_fn(t1, t2) -> Vector{Float64}` closure
so that any mobility indicator can be plugged in without changing the analysis code.

### Available mobility indicators

```julia
# Gaussian  (dim-agnostic)
(t1, t2) -> _gaussian_mobilities(s, t1, t2; a=0.4)

# Heaviside  (dim-agnostic)
(t1, t2) -> _heaviside_mobilities(s, t1, t2; a=0.4)

# Bond-breaking  (requires a neighbour list indexed as nl[t][particle])
(t1, t2) -> _bond_breaking_mobilities(s, nl, t1, t2)

# Cage-corrected Gaussian  (2D only)
(t1, t2) -> _corrected_gaussian_mobilities(s, nl, t1, t2; a=0.4)
```

### Prerequisite: precomputing `C_avg`

Both `chi4_per_simulation` and `compute_G4_per_simulation` require `C_avg`, a
`Vector{Float64}` of length `Ndt` holding the mean mobility ⟨μ(t)⟩ at each lag.
It must be computed **before** the fluctuation functions are called, typically in a
separate averaging pass over all simulations at a given temperature.

Use `average_mobility_per_simulation` from `OverlapFunction.jl` with the same
mobility closure:

```julia
# Gaussian
C_avg, t_arr = average_mobility_per_simulation(s,
    i -> _find_gaussian_overlap_per_particle(s, i; a=factor))

# Heaviside
C_avg, t_arr = average_mobility_per_simulation(s,
    i -> _find_overlap_per_particle(s, i; a=factor))
```

If averaging over multiple simulations at the same temperature, accumulate and
normalise across simulations before passing `C_avg` to the fluctuation functions.
`C_avg[idt]` is subtracted from each per-particle mobility `μᵢ` before computing
fluctuations, so an accurate mean is important.

### Typical cluster script

```julia
import Pkg; Pkg.activate(".")
using SimulationAnalysis
using DelimitedFiles

factor = 0.4

d                = parse(Int,     ARGS[1])
N                = parse(Int,     ARGS[2])
dir_data         = ARGS[3]
simulation_fname = ARGS[4]
dir_overlap_data = ARGS[5]
dir_to_save      = ARGS[6]
dir_meta_data    = ARGS[7]

println("Processing: $simulation_fname")

T       = get_temperature(simulation_fname)
T_float = round(parse(Float64, T); digits=3)

fname_C_avg = "self_overlap_a_$(factor)_T_$(T_float).dat"
t_arr, C_avg = read_C_avg_file(dir_overlap_data, fname_C_avg)

s = SimulationAnalysis.read_continuously_hard_sphere_simulation(
        joinpath(dir_data, simulation_fname);
        original=false, velocities=false, forcestype=false, time_origins="quasilog")

mobility_fn = (t1, t2) -> _gaussian_mobilities(s, t1, t2; a=factor)
# mobility_fn = (t1, t2) -> _heaviside_mobilities(s, t1, t2; a=factor)
# mobility_fn = (t1, t2) -> _bond_breaking_mobilities(s, nl, t1, t2)

# --- scalar χ₄(t) ---
fname_chi4 = "chi4_a_$(factor)_" * simulation_fname[1:end-3] * ".dat"
chi4 = chi4_per_simulation(s, mobility_fn, C_avg; time_avg=true)
open(joinpath(dir_to_save, fname_chi4), "w") do io
    writedlm(io, hcat(t_arr, chi4), ',')
end

# --- spatial G4(r,t) ---
r_arr       = 0:0.1:s.box_sizes[1]/2
fname_G4    = "G4_a_$(factor)_" * simulation_fname[1:end-3] * ".dat"
compute_G4_per_simulation(s, mobility_fn, C_avg, r_arr, fname_G4,
                          dir_meta_data, dir_to_save;
                          time_avg=true, definition=:traditional)
```

### Metadata file written by `compute_G4_per_simulation`

`compute_G4_per_simulation` writes one metadata file per simulation to `dir_meta_data`:

```
G4_r_vals_t_vals_<simulation_fname_without_extension>.dat
```

It is a two-column CSV whose rows are padded with `missing` so both columns reach
the same length:

| column 1 | column 2 |
|----------|----------|
| t[1]     | r[1]     |
| t[2]     | r[2]     |
| …        | …        |
| t[Ndt]   | r[Nr+1]  |
| missing  | r[Nr+1]  | ← if Nr+1 > Ndt, or vice-versa

To read it back and recover the axes:

```julia
meta = readdlm(joinpath(dir_meta_data, "G4_r_vals_t_vals_<stem>.dat"), ',', Any)
t_vals = Float64.(filter(!ismissing, meta[:, 1]))   # length Ndt
r_vals = Float64.(filter(!ismissing, meta[:, 2]))   # length Nr+1  (bin edges)
```

The G4 and r²G4 output files (`[trad_]fname` and `[trad_]r2_fname`) are matrices
of shape `(Ndt, Nr)` — index as `G4[idt, ir]` to get G4 at lag `t_vals[idt]` and
radial bin centred at `(r_vals[ir] + r_vals[ir+1]) / 2`.
=#

"""
    find_pair_distance(r1_vec, r2_vec, box_sizes)

Compute the minimum-image distance between `r1_vec` and `r2_vec` under periodic
boundary conditions defined by `box_sizes`.
"""
function find_pair_distance(r1_vec::SVector{d,Float64}, r2_vec::SVector{d,Float64}, box_sizes) where d
    dr_vec = r1_vec - r2_vec
    dr_vec -= box_sizes[1] * round.(dr_vec / box_sizes[1])
    return sqrt(sum(dr_vec .^ 2))
end

"""
    _gaussian_mobilities(s, t1, t2; a=0.4, wrapped=false) -> Vector{Float64}

Return the Gaussian mobility `exp(-dr2_i / (2a²))` for every particle at time
pair `(t1, t2)`. Dim-agnostic.
"""
function _gaussian_mobilities(s::SimulationAnalysis.SingleComponentSimulation, t1::Int, t2::Int; a=0.4, wrapped=false)
    dims = s.Ndims
    mu = zeros(Float64, s.N)
    for particle in 1:s.N
        dr2 = 0.0
        for dim in 1:dims
            drdim = s.r_array[dim, particle, t2] - s.r_array[dim, particle, t1]
            if wrapped
                drdim -= s.box_sizes[dim] * round(drdim / s.box_sizes[dim])
            end
            dr2 += drdim^2
        end
        mu[particle] = exp(-dr2 / (2 * a^2))
    end
    return mu
end

"""
    _heaviside_mobilities(s, t1, t2; a=0.4, wrapped=false) -> Vector{Float64}

Return the Heaviside mobility `θ(a - |Δr_i|)` for every particle at time pair
`(t1, t2)`. Dim-agnostic.
"""
function _heaviside_mobilities(s::SimulationAnalysis.SingleComponentSimulation, t1::Int, t2::Int; a=0.4, wrapped=false)
    dims = s.Ndims
    mu = zeros(Float64, s.N)
    for particle in 1:s.N
        dr2 = 0.0
        for dim in 1:dims
            drdim = s.r_array[dim, particle, t2] - s.r_array[dim, particle, t1]
            if wrapped
                drdim -= s.box_sizes[dim] * round(drdim / s.box_sizes[dim])
            end
            dr2 += drdim^2
        end
        mu[particle] = sqrt(dr2) < a ? 1.0 : 0.0
    end
    return mu
end

"""
    _bond_breaking_mobilities(s, neighbourlists, t1, t2) -> Vector{Float64}

Return the bond-breaking mobility `C_B` for every particle at time pair `(t1, t2)`,
using `CB_microkernel` from BondBreakingParameter.jl. Particles with an empty initial
neighbor list (sentinel value 1000.0) are assigned 1.0 (all bonds trivially retained).
"""
function _bond_breaking_mobilities(s::SimulationAnalysis.SingleComponentSimulation, neighbourlists, t1::Int, t2::Int)
    mu = zeros(Float64, s.N)
    for particle in 1:s.N
        cb = CB_microkernel(neighbourlists[t1][particle], neighbourlists[t2][particle])
        mu[particle] = cb < 100.0 ? cb : 1.0
    end
    return mu
end

"""
    _corrected_gaussian_mobilities(s, neighborlist, t1, t2; a=0.4) -> Vector{Float64}

Return cage-corrected Gaussian mobilities for every particle at time pair
`(t1, t2)`. Displacements are relative to the neighbor center-of-mass motion.
Currently 2D only.
"""
function _corrected_gaussian_mobilities(s::SimulationAnalysis.SingleComponentSimulation, neighborlist, t1::Int, t2::Int; a=0.4)
    ex, ey = 1, 2
    mu = zeros(Float64, s.N)

    for particle in 1:s.N
        r1x = s.r_array[ex, particle, t1]
        r1y = s.r_array[ey, particle, t1]
        r2x = s.r_array[ex, particle, t2]
        r2y = s.r_array[ey, particle, t2]

        nb_cm_x1 = 0.0
        nb_cm_y1 = 0.0
        nb_cm_x2 = 0.0
        nb_cm_y2 = 0.0

        nl = neighborlist[t1][particle]
        if !isempty(nl)
            @turbo for i in eachindex(nl)
                nb = nl[i]
                nb_cm_x1 += s.r_array[ex, nb, t1]
                nb_cm_y1 += s.r_array[ey, nb, t1]
                nb_cm_x2 += s.r_array[ex, nb, t2]
                nb_cm_y2 += s.r_array[ey, nb, t2]
            end
            nb_cm_x1 /= length(nl)
            nb_cm_y1 /= length(nl)
            nb_cm_x2 /= length(nl)
            nb_cm_y2 /= length(nl)
        end

        drx = (r2x - nb_cm_x2) - (r1x - nb_cm_x1)
        dry = (r2y - nb_cm_y2) - (r1y - nb_cm_y1)
        dr2 = drx^2 + dry^2
        mu[particle] = exp(-dr2 / (2 * a^2))
    end
    return mu
end

"""
    _find_G4_t_per_simulation(s, all_mobilities_fn, C_avg, idt, r_arr; time_avg=true, definition=:standard)

Compute G4(r,t) and r²G4(r,t) at a single lag index `idt`.

`all_mobilities_fn(t1, t2) -> Vector{Float64}` returns one mobility value per
particle for the pair `(t1, t2)`. Provide it as a closure that captures `s` and
any indicator-specific parameters:

```julia
# Gaussian
_find_G4_t_per_simulation(s, (t1, t2) -> _gaussian_mobilities(s, t1, t2; a=0.4), C_avg, idt, r_arr)

# Heaviside
_find_G4_t_per_simulation(s, (t1, t2) -> _heaviside_mobilities(s, t1, t2; a=0.4), C_avg, idt, r_arr)

# Bond-breaking
_find_G4_t_per_simulation(s, (t1, t2) -> _bond_breaking_mobilities(s, neighbourlists, t1, t2), C_avg, idt, r_arr)

# Cage-corrected
_find_G4_t_per_simulation(s, (t1, t2) -> _corrected_gaussian_mobilities(s, nl, t1, t2; a=0.4), C_avg, idt, r_arr)
```

`definition`:
- `:standard`    — bins `δμ_i * δμ_j` where `δμ_i = μ_i - C_avg[idt]`
- `:traditional` — bins `μ_i * μ_j`, then subtracts `C_avg[idt]²` after normalization

Returns `(G4_bins, r2G4_bins)` each of length `length(r_arr) - 1`.
"""
function _find_G4_t_per_simulation(s::SimulationAnalysis.SingleComponentSimulation, all_mobilities_fn, C_avg, idt::Int, r_arr; time_avg=true, definition=:standard)

    dims = s.Ndims
    pairs_idt = s.t1_t2_pair_array[idt]

    Nr = length(r_arr) - 1
    G4_bins = zeros(Float64, Nr)
    r2G4_bins = zeros(Float64, Nr)

    dr_bin = r_arr[2] - r_arr[1]
    pair_count = 0

    for ipair in axes(pairs_idt, 1)
        t1, t2 = pairs_idt[ipair, 1], pairs_idt[ipair, 2]

        mu = all_mobilities_fn(t1, t2)

        dmu = if definition == :traditional
            mu
        else
            mu .- C_avg[idt]
        end

        for particle_id1 in 1:s.N
            dmu1 = dmu[particle_id1]
            r1_vec = SVector(ntuple(d -> s.r_array[d, particle_id1, t2], dims))

            for particle_id2 in particle_id1:s.N
                dmu2 = dmu[particle_id2]
                r2_vec = SVector(ntuple(d -> s.r_array[d, particle_id2, t2], dims))

                pair_dist = find_pair_distance(r1_vec, r2_vec, s.box_sizes)
                r_index = ceil(Int, (pair_dist - r_arr[1]) / dr_bin)

                if 0 < r_index <= Nr
                    r_center = (r_arr[r_index] + r_arr[r_index + 1]) / 2
                    G4_bins[r_index] += 2 * dmu1 * dmu2
                    r2G4_bins[r_index] += 2 * r_center^2 * dmu1 * dmu2
                end
            end
        end

        pair_count += 1
        if !time_avg
            break
        end
    end

    pair_norm = time_avg ? pair_count : 1

    for k in 1:Nr
        vol_shell = shell_volume(r_arr[k], r_arr[k + 1], dims)
        norm = pair_norm * vol_shell * s.N
        G4_bins[k] /= norm
        r2G4_bins[k] /= norm
    end

    if definition == :traditional
        c2 = C_avg[idt]^2
        for k in 1:Nr
            r_center = (r_arr[k] + r_arr[k + 1]) / 2
            G4_bins[k] -= c2
            r2G4_bins[k] -= r_center^2 * c2
        end
    end

    return G4_bins, r2G4_bins
end

"""
    compute_G4_per_simulation(s, all_mobilities_fn, C_avg, r_arr, fname, dir_meta_data, dir_chi; time_avg=true, definition=:standard)

Compute G4(r,t) and r²G4(r,t) for every lag in `s`, write bin metadata to
`dir_meta_data`, and save results to `dir_chi`.

`all_mobilities_fn(t1, t2) -> Vector{Float64}` — see `_find_G4_t_per_simulation`.

Output files: `fname` for G4, `r2_fname` for r²G4. Prefixed with `"trad_"` when
`definition=:traditional`.

# Examples
```julia
# Gaussian self-overlap
compute_G4_per_simulation(s,
    (t1, t2) -> _gaussian_mobilities(s, t1, t2; a=0.4),
    C_avg, r_arr, fname, dir_meta, dir_chi)

# Heaviside self-overlap
compute_G4_per_simulation(s,
    (t1, t2) -> _heaviside_mobilities(s, t1, t2; a=0.4),
    C_avg, r_arr, fname, dir_meta, dir_chi)

# Bond-breaking
compute_G4_per_simulation(s,
    (t1, t2) -> _bond_breaking_mobilities(s, neighbourlists, t1, t2),
    C_avg, r_arr, fname, dir_meta, dir_chi)

# Cage-corrected Gaussian
compute_G4_per_simulation(s,
    (t1, t2) -> _corrected_gaussian_mobilities(s, neighborlist, t1, t2; a=0.4),
    C_avg, r_arr, fname, dir_meta, dir_chi)
```
"""
function compute_G4_per_simulation(s::SimulationAnalysis.SingleComponentSimulation, all_mobilities_fn, C_avg, r_arr, fname, dir_meta_data, dir_chi; time_avg=true, definition=:standard)

    Ndt = length(s.dt_array)
    Nr = length(r_arr) - 1

    dt_step = s.t_array[2] - s.t_array[1]
    t_arr = s.dt_array .* dt_step

    max_length = max(Ndt, length(r_arr))
    fname_base = split(s.filepath, "/")[end]
    fname_meta = "G4_r_vals_t_vals_" * fname_base[1:end-3] * ".dat"

    r_arr_padded = vcat(r_arr, fill(missing, max_length - length(r_arr)))
    t_arr_padded = vcat(t_arr, fill(missing, max_length - Ndt))
    r_t_data = hcat(t_arr_padded, r_arr_padded)

    println("Writing $fname_meta to $dir_meta_data")
    open(joinpath(dir_meta_data, fname_meta), "w") do io
        writedlm(io, r_t_data, ',')
    end

    G4 = zeros(Ndt, Nr)
    r2G4 = zeros(Ndt, Nr)

    println("STARTING COMPUTATION FOR ALL idt")
    @threads for idt in eachindex(s.dt_array)
        @time G4_t, r2G4_t = _find_G4_t_per_simulation(s, all_mobilities_fn, C_avg, idt, r_arr; time_avg=time_avg, definition=definition)
        G4[idt, :] = G4_t
        r2G4[idt, :] = r2G4_t
        flush(stdout)
    end

    G4 = stack(G4)
    r2G4 = stack(r2G4)

    prefix = definition == :traditional ? "trad_" : ""

    println("Writing G4(r,t) to $dir_chi/$(prefix)$fname")
    open(joinpath(dir_chi, prefix * fname), "w") do io
        writedlm(io, G4, ',')
    end

    fname_r2 = prefix * "r2_" * fname
    println("Writing r2G4(r,t) to $dir_chi/$fname_r2")
    open(joinpath(dir_chi, fname_r2), "w") do io
        writedlm(io, r2G4, ',')
    end
end

"""
    chi4_per_simulation(s, all_mobilities_fn, C_avg; time_avg=true) -> Vector{Float64}

Compute χ₄(t) = (1/N) ⟨Σᵢ (μᵢ(t) - C_avg[t])²⟩ for every lag in `s`.

`all_mobilities_fn(t1, t2) -> Vector{Float64}` — see `_find_G4_t_per_simulation`.

# Examples
```julia
# Gaussian
chi4_per_simulation(s, (t1, t2) -> _gaussian_mobilities(s, t1, t2; a=0.4), C_avg)

# Heaviside
chi4_per_simulation(s, (t1, t2) -> _heaviside_mobilities(s, t1, t2; a=0.4), C_avg)

# Bond-breaking
chi4_per_simulation(s, (t1, t2) -> _bond_breaking_mobilities(s, neighbourlists, t1, t2), C_avg)

# Cage-corrected Gaussian
chi4_per_simulation(s, (t1, t2) -> _corrected_gaussian_mobilities(s, neighbourlists, t1, t2; a=0.4), C_avg)
```
"""
function chi4_per_simulation(s::SimulationAnalysis.SingleComponentSimulation, all_mobilities_fn, C_avg; time_avg=true)
    Ndt = length(s.dt_array)
    chi4 = zeros(Float64, Ndt)

    @threads for idt in eachindex(s.dt_array)
        pairs_idt = s.t1_t2_pair_array[idt]
        pair_count = 0

        for ipair in axes(pairs_idt, 1)
            t1, t2 = pairs_idt[ipair, 1], pairs_idt[ipair, 2]
            mu = all_mobilities_fn(t1, t2)
            c = C_avg[idt]
            for particle in 1:s.N
                chi4[idt] += (mu[particle] - c)^2
            end
            pair_count += 1
            if !time_avg
                break
            end
        end

        if pair_count > 0
            chi4[idt] /= (pair_count * s.N)
        end
    end

    return chi4, s.dt_array
end
