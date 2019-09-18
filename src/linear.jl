function linear_step(hmm::HMM{Univariate,Float64}, observations::Matrix{Int64}, obs_lengths::Vector{Int64})
    O,T = size(observations);
    a = log.(hmm.π); π0 = transpose(log.(hmm.π0))
    N = length(hmm.D); D = length(hmm.D[1].support);
    mask=observations.!=0
    #INITIALIZATION
    βoi_T = zeros(O,N); βoi_t = zeros(O,N) #log betas at T initialised as zeros
    Eoimd_T = fill(-Inf,O,N,N,D); Eoimd_t = fill(-Inf,O,N,N,D)
    @inbounds for m in 1:N, i in 1:N, γ in 1:D, o in 1:O
        observations[o, obs_lengths[o]] == γ && m == i && (Eoimd_T[o, i, m, γ] = 0)
    end
    Tijm_T = fill(-Inf,O,N,N,N); Tijm_t = fill(-Inf,O,N,N,N) #Ti,j(T,m) = 0 for all m; in logspace
        
    #RECURRENCE
    βoi_T,Tijm_T,Eoimd_T=backwards_sweep!(a,N,D,βoi_T,βoi_t,Tijm_T,Tijm_t,Eoimd_T,Eoimd_t,mask,obs_lengths)

    #TERMINATION
    lls = llhs(hmm,observations[:,1])
    α1om = lls .+ π0 #first position forward msgs
    Toij = [logsumexp([lps(view(Tijm_T,o,i,j,m), view(α1om,o,m)) for m in 1:N]) for o in 1:O, i in 1:N, j in 1:N] #terminate Tijs with forward messages
    Eoid=[logsumexp([lps(view(Eoimd_T,o,i,m,γ), view(α1om,o,m)) for m in 1:N]) for o in 1:O, i in 1:N, γ in 1:D] #terminate Eids with forward messages

    #INTEGRATE ACROSS OBSERVATIONS AND SOLVE FOR NEW HMM PARAMS
    obs_penalty=log(O) #broadcast subtraction to normalise log prob vals by obs number
    #INITIAL STATE DIST
    π0_o=α1om.+βoi_T.-logsumexp.(eachrow(α1om.+βoi_T)) #estimate π0 for each o
    new_π0=logsumexp.(eachcol(π0_o)).-obs_penalty #sum over obs and normalise by number of obs
    #TRANSITION MATRIX
    a_int = Toij.-logsumexp.([view(Toij,o,i,:) for o in 1:O, i in 1:N])
    new_a = logsumexp.([a_int[:,i,j] for i in 1:N, j in 1:N]).-obs_penalty
    #EMISSION MATRIX
    e_int=Eoid.-logsumexp.([view(Eoid,o,j,:) for o in 1:O, j in 1:N])
    new_b=logsumexp.([view(e_int,:,j,d) for d in 1:D, j in 1:N]).-obs_penalty
    new_D::Vector{Categorical}=[Categorical(exp.(new_b[:,i])) for i in 1:N]

    return typeof(hmm)(exp.(new_π0), exp.(new_a), new_D), lps([logsumexp(lps.(α1om[o,:], βoi_T[o,:])) for o in 1:O])
end
                #LINEAR_STEP SUBFUNCS
                function backwards_sweep!(a::Matrix{Float64},N::Int64,D::Int64,βoi_T::Matrix{Float64},βoi_t::Matrix{Float64},Tijm_T::Array{Float64},Tijm_t::Array{Float64},Eoimd_T::Array{Float64},Eoimd_t::Array{Float64}, mask::BitMatrix, obs_lengths::Vector{Int64})
                    @inbounds for t in maximum(obs_lengths)-1:-1:1
                        lls = llhs(hmm,observations[:,t+1])
                        last_β=copy(βoi_T)
                        omask = mask[:,t+1]
                        βoi_T[omask,:] .+= view(lls,omask,:)
                        for m in 1:N
                            trans_view = transpose(view(a,m,:))
                            βoi_t[omask,m] = logsumexp.(eachrow(view(βoi_T,omask,:).+trans_view))
                            for i in 1:N, j in 1:N
                                Tijm_t[omask, i, j, m] .= logsumexp.(eachrow(lps.(view(Tijm_T,omask,i,j,:), view(lls,omask,:), trans_view)))
                                i==m && (Tijm_t[omask, i, j, m] .= logaddexp.(Tijm_t[omask, i, j, m], (last_β[omask,j].+ a[m,j].+ lls[omask,j])))
                            end
                            for i in 1:N, γ in 1:D
                                Eoimd_t[omask, i, m, γ] .= logsumexp.(eachrow(lps.(view(Eoimd_T,omask,i,:,γ),view(lls,omask,:),trans_view)))
                                if i==m
                                    symmask = observations[:,t].==γ
                                    Eoimd_t[symmask, i, m, γ] .= logaddexp.(Eoimd_t[symmask,i, m, γ], βoi_t[symmask,m])
                                end
                            end
                        end
                        βoi_T=copy(βoi_t); Tijm_T=copy(Tijm_t); Eoimd_T = copy(Eoimd_t);
                    end
                    return βoi_T, Tijm_T, Eoimd_T
                end

                function llhs(hmm::AbstractHMM{Univariate}, observation::Vector{Int64})
                    lls = zeros(length(observation),length(hmm.D))
                    for d in 1:length(hmm.D)
                        lls[:,d] = logpdf.(hmm.D[d], observation)
                    end
                    return lls
                end

                #subfuncs to handle sums of log probabilities that may include -Inf (ie p=0), returning -Inf in this case rather than NaNs
                function lps(adjuvants::AbstractArray)
                    prob = sum(adjuvants) ; isnan(prob) ? - Inf : prob
                end
                
                function lps(base, adjuvants...)::Float64
                    prob = base+sum(adjuvants) ; isnan(prob) ? -Inf : prob
                end


function linear_hmm_converger!(hmm_jobs::RemoteChannel, output_hmms::RemoteChannel, no_models::Int64, ; delta_thresh=1e-3, max_iterations=5000, verbose=false)
    while isready(hmm_jobs)
        workerid = myid()
        jobid::Tuple{String, Int64, Int64, Int64}, start_iterate::Int64, hmm::HMM, job_norm::Float64, observations::Matrix{Int64} = take!(hmm_jobs)
        jobid == 0 && break #no valid job for this worker according to load_table entry

        @assert start_iterate < max_iterations - 1
        curr_iterate = start_iterate

        #build array of observation lengths
        obs_lengths = [findfirst(iszero,observations[:,o])-1 for o in 1:size(observations)[2]] #mask calculations here rather than mle_step to prevent recalculation every iterate
        start_iterate == 1 && put!(output_hmms, (workerid, jobid, curr_iterate, hmm, 0, 0, false)); #on the first iterate return the initial HMM for the chain right away
        verbose && @info "Fitting HMM, start iterate $start_iterate, job ID $jobid with $(size(hmm.π)[1]) states and $(length(hmm.D[1].support)) symbols..."

        curr_iterate += 1
        if curr_iterate == 2 #no delta value is available
            new_hmm, last_norm = linear_step(hmm, observations, obs_lengths)
            put!(output_hmms, (workerid, jobid, curr_iterate, new_hmm, last_norm, 0, false))
        else #get the delta value from the channel-supplied job value to resume a chain properly
            new_hmm, last_norm = linear_step(hmm, observations, obs_lengths)
            delta = lps(job_norm, -last_norm)
            put!(output_hmms, (workerid, jobid, curr_iterate, new_hmm, last_norm, delta, false))
        end

        for i in curr_iterate:max_iterations
            new_hmm, norm = linear_step(new_hmm, observations, obs_lengths)
            curr_iterate += 1
            delta = lps(norm, -last_norm)
            if delta < delta_thresh
                put!(output_hmms, (workerid, jobid, curr_iterate, new_hmm, norm, delta, true))
                verbose && @info "Job ID $jobid converged after $(curr_iterate-1) EM steps"
                break
            else
                put!(output_hmms, (workerid, jobid, curr_iterate, new_hmm, norm, delta, false))
                verbose && @info "Job ID $jobid EM step $(curr_iterate-1) delta $delta"
                last_norm = norm
            end
        end
    end
end

function lin_obs_set_lh(hmm::HMM{Univariate,Float64}, observations::Matrix{Int64})
    O = size(observations)[2]; obs_lengths = [findfirst(iszero,observations[:,o])-1 for o in 1:size(observations)[2]]
    a = log.(hmm.π); π0 = log.(hmm.π0)
    N = length(hmm.D); D = length(hmm.D[1].support); b = [log(hmm.D[m].p[γ]) for m in 1:N, γ in 1:D]
    α1oi = zeros(O,N); β1oi = zeros(O,N); Eoi = zeros(O,D,N); Toij = zeros(O,N,N); πoi = zeros(O,N); log_pobs=zeros(O); γt=0
    
    Threads.@threads for o in 1:O
        #INITIALIZATION
        T = obs_lengths[o]; βT = zeros(N) #log betas at T initialised as zeros
        #RECURRENCE
        for t in T-1:-1:1
            βt = similar(βT); Γ = observations[t+1,o]
            for m in 1:N
                βt[m] = logsumexp([lps(a[m,j], b[j,Γ], βT[j]) for j in 1:N])
            end
            βT = βt
        end
        #TERMINATION
        Γ = observations[1,o]
        α1oi[o,:] = [lps(π0[i], b[i, Γ]) for i in 1:N]
        log_pobs[o] = logsumexp(lps.(α1oi[o,:], βT[:]))
     end

    return logsumexp(log_pobs)
end