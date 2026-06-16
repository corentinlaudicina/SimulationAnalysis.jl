"""
    _find_overlap_per_particle(s, particle_id; a=0.4, wrapped=false, time_averaging=true)

Compute the Heaviside overlap function for `particle_id` across all `t1,t2` pairs.
If `time_averaging` is false, only the first pair per lag is used.
Returns `(corr_particle, t_array)`.
"""
function _find_overlap_per_particle(s::Union{SingleComponentSimulation, SelfPropelledVoronoiSimulation}, particle_id::Int; a=0.4, wrapped=false, time_averaging::Bool=true)
    if particle_id < 1 || particle_id > s.N
        throw(ArgumentError("particle_id must be between 1 and $(s.N) (got $particle_id)"))
    end

    dims = s.Ndims
    Ndt = length(s.dt_array)
    corr_particle = zeros(Float64, Ndt)

    for idt in eachindex(s.dt_array)
        pairs_idt = s.t1_t2_pair_array[idt]
        corr_sum = 0.0
        pair_count = 0

        for ipair in axes(pairs_idt, 1)
            t1, t2 = pairs_idt[ipair, 1], pairs_idt[ipair, 2]
            dr2 = 0.0
            for dim in 1:dims
                drdim = s.r_array[dim, particle_id, t1] - s.r_array[dim, particle_id, t2]
                if wrapped
                    drdim -= s.box_sizes[dim] * round(drdim / s.box_sizes[dim])
                end
                dr2 += drdim^2
            end
            corr_sum += sqrt(dr2) < a ? 1.0 : 0.0
            pair_count += 1

            if !time_averaging
                break
            end
        end

        if pair_count > 0
            corr_particle[idt] = corr_sum / pair_count
        else
            @warn "No pairs for particle $particle_id at dt index $idt. Returning 0.0"
        end
    end

    dt_step = s.t_array[2] - s.t_array[1]
    t_array = s.dt_array .* dt_step
    return corr_particle, t_array
end

"""
    find_overlap_function(s::Union{SingleComponentSimulation, SelfPropelledVoronoiSimulation}; a=0.4, wrapped=false)

Calculates the overlap function `Q(t)`.

The overlap function is a measure of the similarity between the particle configurations at time `0` and time `t`. It is defined as:
`Q(t) = (1/N) * sum_i theta(a - |r_i(t) - r_i(0)|)`,
where `theta` is the Heaviside step function, `a` is a cutoff distance, and the average is taken over all particles `i` and time origins.
A value of 1 means the configurations are identical (within the cutoff), and a value of 0 means they are completely different.

# Arguments
- `s::Union{SingleComponentSimulation, SelfPropelledVoronoiSimulation}`: The simulation data.
- `a::Float64=0.4`: The cutoff distance for calculating the overlap. Typically this is a fraction of the particle diameter.
- `wrapped::Bool=false`: Whether to apply periodic boundary conditions.

# Returns
- `Fs::Vector{Float64}`: A vector containing `Q(t)` for each time delay in `s.dt_array`.
- `Fs_pp::Matrix{Float64}`: A `(Ndt, N)` matrix containing the overlap function for each particle.
"""
function find_overlap_function(s::Union{SingleComponentSimulation, SelfPropelledVoronoiSimulation}; a=0.4, wrapped=false)
    N = s.N
    Ndt = length(s.dt_array)
    Fs_pp = zeros(Ndt, N)
    for particle in 1:N
        corr, _ = _find_overlap_per_particle(s, particle; a=a, wrapped=wrapped)
        Fs_pp[:, particle] .= corr
    end
    Fs = sum(Fs_pp; dims=2)[:] / N
    return Fs, Fs_pp
end

"""
    _find_gaussian_overlap_per_particle(s, particle_id; a=0.4, wrapped=false, time_averaging=true)

Compute the Gaussian self-overlap for `particle_id` across all `t1,t2` pairs
using a Gaussian window of width `a`. If `time_averaging` is false, only
the first pair per lag is used. Returns `(corr_particle, t_array)`.
"""
function _find_gaussian_overlap_per_particle(s::SimulationAnalysis.SingleComponentSimulation, particle_id::Int; a=0.4, wrapped=false, time_averaging::Bool=true)
    if particle_id < 1 || particle_id > s.N
        throw(ArgumentError("particle_id must be between 1 and $(s.N) (got $particle_id)"))
    end

    dims = s.Ndims
    Ndt = length(s.dt_array)
    corr_particle = zeros(Float64, Ndt)

    for idt in eachindex(s.dt_array)
        pairs_idt = s.t1_t2_pair_array[idt]
        corr_sum = 0.0
        pair_count = 0

        for ipair in axes(pairs_idt, 1)
            t1, t2 = pairs_idt[ipair, 1], pairs_idt[ipair, 2]
            dr2 = 0.0
            for dim in 1:dims
                drdim = s.r_array[dim, particle_id, t2] - s.r_array[dim, particle_id, t1]
                if wrapped
                    drdim -= s.box_sizes[dim] * round(drdim / s.box_sizes[dim])
                end
                dr2 += drdim^2
            end
            corr_sum += exp(-dr2 / (2 * a^2))
            pair_count += 1

            if !time_averaging
                break
            end
        end

        if pair_count > 0
            corr_particle[idt] = corr_sum / pair_count
        else
            @warn "No pairs for particle $particle_id at dt index $idt. Returning 0.0"
        end
    end

    dt_step = s.t_array[2] - s.t_array[1]
    t_array = s.dt_array .* dt_step
    return corr_particle, t_array
end

"""
    _find_corrected_gaussian_overlap_per_particle(s, neighborlist, particle_id; a=0.4, wrapped=false, time_averaging=true)

Compute Gaussian self-overlap for `particle_id` after subtracting the local neighbor
center-of-mass motion defined by `neighborlist`. Returns `(corr_particle, t_array)`.
"""
function _find_corrected_gaussian_overlap_per_particle(s::SimulationAnalysis.SingleComponentSimulation, neighborlist, particle_id::Int; a=0.4, wrapped=false, time_averaging::Bool=true)
    if particle_id < 1 || particle_id > s.N
        throw(ArgumentError("particle_id must be between 1 and $(s.N) (got $particle_id)"))
    end

    ex, ey = 1, 2
    Ndt = length(s.dt_array)
    corr_particle = zeros(Float64, Ndt)

    for idt in eachindex(s.dt_array)
        pairs_idt = s.t1_t2_pair_array[idt]
        corr_sum = 0.0
        pair_count = 0

        for ipair in axes(pairs_idt, 1)
            t1, t2 = pairs_idt[ipair, 1], pairs_idt[ipair, 2]

            r1x = s.r_array[ex, particle_id, t1]
            r1y = s.r_array[ey, particle_id, t1]
            r2x = s.r_array[ex, particle_id, t2]
            r2y = s.r_array[ey, particle_id, t2]

            nb_cm_x1 = 0.0
            nb_cm_y1 = 0.0
            nb_cm_x2 = 0.0
            nb_cm_y2 = 0.0

            nl = neighborlist[t1][particle_id]
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

            if wrapped
                drx -= s.box_sizes[ex] * round(drx / s.box_sizes[ex])
                dry -= s.box_sizes[ey] * round(dry / s.box_sizes[ey])
            end

            dr2 = drx^2 + dry^2
            corr_sum += exp(-dr2 / (2 * a^2))
            pair_count += 1

            if !time_averaging
                break
            end
        end

        if pair_count > 0
            corr_particle[idt] = corr_sum / pair_count
        else
            @warn "No pairs for particle $particle_id at dt index $idt. Returning 0.0"
        end
    end

    dt_step = s.t_array[2] - s.t_array[1]
    t_array = s.dt_array .* dt_step
    return corr_particle, t_array
end

"""
    average_mobility_per_simulation(s, per_particle_fn)

Average a per-particle mobility measure over all particles in simulation `s`.
`per_particle_fn(particle_id)` must return `(corr::Vector, t_array::Vector)`.

# Examples
```julia
# Gaussian self-overlap
average_mobility_per_simulation(s, i -> _find_gaussian_overlap_per_particle(s, i; a=a, time_averaging=true))

# Neighbor-corrected self-overlap
average_mobility_per_simulation(s, i -> _find_corrected_gaussian_overlap_per_particle(s, neighborlist, i; a=a))

# Heaviside overlap
average_mobility_per_simulation(s, i -> _find_overlap_per_particle(s, i; a=0.4, wrapped=wrapped))
```
"""
function average_mobility_per_simulation(s::SimulationAnalysis.SingleComponentSimulation, per_particle_fn)
    Ndt = length(s.dt_array)
    corr_per_sim = zeros(Ndt)
    t_array = zeros(Ndt)

    for i in 1:s.N
        corr_temp, t_array_temp = per_particle_fn(i)
        corr_per_sim .+= corr_temp
        t_array .+= t_array_temp
    end
    corr_per_sim ./= s.N
    t_array ./= s.N

    return corr_per_sim, t_array
end
