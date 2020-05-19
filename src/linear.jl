function linear_step(hmm::HMM{Univariate,AbstractFloat}, observations::Matrix{Integer}, obs_lengths::Vector{Integer})
    O,T = size(observations);
    a = log.(hmm.π); π0 = transpose(log.(hmm.π0))
    N = length(hmm.D); Γ = length(hmm.D[1].support);
    mask=observations.!=0
    #INITIALIZATION
    βoi_T = zeros(O,N); βoi_t = zeros(O,N) #log betas at T initialised as zeros
    Eoγim_T = fill(-Inf,O,Γ,N,N); Eoγim_t = fill(-Inf,O,Γ,N,N)
    @inbounds for m in 1:N, i in 1:N, γ in 1:Γ, o in 1:O
        observations[o, obs_lengths[o]] == γ && m == i && (Eoγim_T[o, γ, i, m] = 0)
    end
    Tijm_T = fill(-Inf,O,N,N,N); Tijm_t = fill(-Inf,O,N,N,N) #Ti,j(T,m) = 0 for all m; in logspace
        
    #RECURRENCE
    βoi_T,Tijm_T,Eoγim_T=backwards_sweep!(hmm,a,N,Γ,βoi_T,βoi_t,Tijm_T,Tijm_t,Eoγim_T,Eoγim_t,observations,mask,obs_lengths)
        
    #TERMINATION
    lls = llhs(hmm,observations[:,1])
    α1om = lls .+ π0 #first position forward msgs
    println(view(Tijm_T,o,i,j,m))
    println(view(α1om,o,m))

    Toij = [logsumexp([lps(view(Tijm_T,o,i,j,m), view(α1om,o,m)) for m in 1:N]) for o in 1:O, i in 1:N, j in 1:N] #terminate Tijs with forward messages
    Eoiγ=[logsumexp([lps(view(Eoγim_T,o,γ,i,m), view(α1om,o,m)) for m in 1:N]) for o in 1:O, i in 1:N, γ in 1:Γ] #terminate Eids with forward messages

    #INTEGRATE ACROSS OBSERVATIONS AND SOLVE FOR NEW HMM PARAMS
    obs_penalty=log(O) #broadcast subtraction to normalise log prob vals by obs number
    #INITIAL STATE DIST
    π0_o=α1om.+βoi_T.-logsumexp.(eachrow(α1om.+βoi_T)) #estimate π0 for each o
    new_π0=logsumexp.(eachcol(π0_o)).-obs_penalty #sum over obs and normalise by number of obs
    #TRANSITION MATRIX
    a_int = Toij.-logsumexp.([view(Toij,o,i,:) for o in 1:O, i in 1:N])
    new_a = logsumexp.([a_int[:,i,j] for i in 1:N, j in 1:N]).-obs_penalty
    #EMISSION MATRIX
    e_int=Eoiγ.-logsumexp.([view(Eoiγ,o,j,:) for o in 1:O, j in 1:N])
    new_b=logsumexp.([view(e_int,:,j,γ) for γ in 1:Γ, j in 1:N]).-obs_penalty
    new_D::Vector{Categorical}=[Categorical(exp.(new_b[:,i])) for i in 1:N]

    return typeof(hmm)(exp.(new_π0), exp.(new_a), new_D), lps([logsumexp(lps.(α1om[o,:], βoi_T[o,:])) for o in 1:O])
end
                #LINEAR_STEP SUBFUNCS
                function backwards_sweep!(hmm::HMM{Univariate,AbstractFloat}, a::Matrix{AbstractFloat},N::Integer,Γ::Integer,βoi_T::Matrix{AbstractFloat},βoi_t::Matrix{AbstractFloat},Tijm_T::Array{AbstractFloat},Tijm_t::Array{AbstractFloat},Eoγim_T::Array{AbstractFloat},Eoγim_t::Array{AbstractFloat}, observations::Matrix{Integer}, mask::BitMatrix, obs_lengths::Vector{Integer})
                    @inbounds for t in maximum(obs_lengths)-1:-1:1
                        last_β=copy(βoi_T)
                        lls = llhs(hmm,observations[:,t+1])
                        omask = findall(mask[:,t+1])
                        βoi_T[omask,:] .+= view(lls,omask,:)
                        for m in 1:N
                            βoi_t[omask,m] = logsumexp.(eachrow(view(βoi_T,omask,:).+transpose(view(a,m,:))))
                            for j in 1:N, i in 1:N
                                Tijm_t[omask, i, j, m] .= logsumexp.(eachrow(lps.(view(Tijm_T,omask,i,j,:), view(lls,omask,:), transpose(view(a,m,:)))))
                                i==m && (Tijm_t[omask, i, j, m] .= logaddexp.(Tijm_t[omask, i, j, m], (last_β[omask,j].+ a[m,j].+ lls[omask,j])))
                            end
                            for i in 1:N, γ in 1:Γ
                                Eoγim_t[omask, γ, i, m] .= logsumexp.(eachrow(lps.(view(Eoγim_T,omask,γ,i,:),view(lls,omask,:),transpose(view(a,m,:)))))
                                if i==m
                                    symmask = findall(observations[:,t].==γ)
                                    Eoγim_t[symmask, γ, i, m] .= logaddexp.(Eoγim_t[symmask, γ, i, m], βoi_t[symmask,m])
                                end
                            end
                        end
                        βoi_T=copy(βoi_t); Tijm_T=copy(Tijm_t); Eoγim_T = copy(Eoγim_t);
                    end
                    return βoi_T, Tijm_T, Eoγim_T
                end

                function llhs(hmm::AbstractHMM{Univariate}, observation::Vector{Integer})
                    lls = zeros(length(observation),length(hmm.D))
                    for d in 1:length(hmm.D)
                        lls[:,d] = logpdf.(hmm.D[d], observation)
                    end
                    return lls
                end

                #subfuncs to handle sums of log probabilities that may include -Inf (ie p=0), returning -Inf in this case rather than NaNs
                function lps(adjuvants::Array{AbstractFloat})
                    prob = sum(adjuvants) ; isnan(prob) ? - Inf : prob
                end
                
                function lps(base::AbstractFloat, adjuvants ::AbstractFloat ...)
                    prob = base+sum(adjuvants) ; isnan(prob) ? -Inf : prob
                end


function linear_hmm_converger!(hmm_jobs::RemoteChannel, output_hmms::RemoteChannel, no_models::Integer, ; delta_thresh=1e-3, max_iterations=5000, verbose=false)
    while isready(hmm_jobs)
        workerid = myid()
        jobid::Tuple{String, Integer, Integer, Integer}, start_iterate::Integer, hmm::HMM, job_norm::AbstractFloat, observations::Matrix{Integer} = take!(hmm_jobs)
        jobid == 0 && break #no valid job for this worker according to load_table entry

        @assert start_iterate < max_iterations - 1
        curr_iterate = start_iterate

        #build array of observation lengths
        obs_lengths = [findfirst(iszero,observations[o,:])-1 for o in 1:size(observations)[1]] #mask calculations here rather than mle_step to prevent recalculation every iterate
        start_iterate == 1 && put!(output_hmms, (workerid, jobid, curr_iterate, hmm, 0, 0, false)); #on the first iterate return the initial HMM for the chain right away
        verbose && @info "Fitting HMM, start iterate $start_iterate, job ID $jobid with $(size(hmm.π)[1]) states and $(length(hmm.D[1].support)) symbols..."

        curr_iterate += 1
        if curr_iterate == 2 #no delta value is available
            new_hmm, last_norm = linear_step(hmm, observations, obs_lengths)
            put!(output_hmms, (workerid, jobid, curr_iterate, new_hmm, last_norm, 0, false))
        else #get the delta value from the channel-supplied job value to resume a chain properly
            new_hmm, last_norm = linear_step(hmm, observations, obs_lengths)
            delta = abs(lps(job_norm, -last_norm))
            put!(output_hmms, (workerid, jobid, curr_iterate, new_hmm, last_norm, delta, false))
        end

        for i in curr_iterate:max_iterations
            new_hmm, norm = linear_step(new_hmm, observations, obs_lengths)
            curr_iterate += 1
            delta = abs(lps(norm, -last_norm))
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

function lin_obs_set_lh(hmm::HMM{Univariate,AbstractFloat}, observations::Matrix{Integer})
    O = size(observations)[1]; obs_lengths = [findfirst(iszero,observations[o,:])-1 for o in 1:O]
    a = log.(hmm.π); π0 = log.(hmm.π0)
    N = length(hmm.D); D = length(hmm.D[1].support); b = [log(hmm.D[m].p[γ]) for m in 1:N, γ in 1:D]
    α1oi = zeros(O,N); β1oi = zeros(O,N); Eoi = zeros(O,D,N); Toij = zeros(O,N,N); πoi = zeros(O,N); log_pobs=zeros(O); γt=0
    
    Threads.@threads for o in 1:O
        #INITIALIZATION
        T = obs_lengths[o]; βT = zeros(N) #log betas at T initialised as zeros
        #RECURRENCE
        for t in T-1:-1:1
            βt = similar(βT); Γ = observations[o,t+1]
            for m in 1:N
                βt[m] = logsumexp([lps(a[m,j], b[j,Γ], βT[j]) for j in 1:N])
            end
            βT = βt
        end
        #TERMINATION
        Γ = observations[o,1]
        α1oi[o,:] = [lps(π0[i], b[i, Γ]) for i in 1:N]
        log_pobs[o] = logsumexp(lps.(α1oi[o,:], βT[:]))
     end

    return logsumexp(log_pobs)
end