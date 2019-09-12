function linear_step(hmm::AbstractHMM{F}, observations::Matrix{Int64}, obs_lengths::Vector{Int64}) where F
    O = size(observations)[2]
    a = log.(hmm.π); π0 = log.(hmm.π0)
    N = length(hmm.D); D = length(hmm.D[1].support); b = [log(hmm.D[m].p[γ]) for m in 1:N, γ in 1:D]
    α1oi = zeros(O,N); β1oi = zeros(O,N); Eoi = zeros(O,D,N); Toij = zeros(O,N,N); πoi = zeros(O,N); log_pobs=zeros(O); γt=0
    
    for o in 1:O
        #INITIALIZATION
        T = obs_lengths[o]
        βT = zeros(N) #log betas at T initialised as zeros
        TijT = fill(-Inf,N,N,N) #Ti,j(T,m) = 0 for all m; in logspace
        EiT = fill(-Inf,D,N,N)
        for m in 1:N, i in 1:N, γ in 1:D
            observations[T, o] == γ && m == i ? (EiT[γ, i, m] = 0) :
                (EiT[γ, i, m] = -Inf) #log Ei initialisation
        end

        #RECURRENCE
        for t in T-1:-1:1
            βt = similar(βT); Tijt = similar(TijT); Eit = similar(EiT)
            Γ = observations[t+1,o]; γt = observations[t,o] 
            for m in 1:N
                βt[m] = logsumexp([log_prob_sum(a[m,j], b[j,Γ], βT[j]) for j in 1:N])
                for i in 1:N
                    for j in 1:N
                        Tijt[i, j, m] = logsumexp([log_prob_sum(a[m,n], TijT[i, j, n], b[n,Γ]) for n in 1:N])
                        i==m && (Tijt[i, j, m] = logaddexp(Tijt[i, j, m], log_prob_sum(βT[j], a[m,j], b[j, Γ])))
                    end
                    for γ in 1:D
                        Eit[γ, i, m] = logsumexp([log_prob_sum(b[n,Γ], a[m,n], EiT[γ, i, n]) for n in 1:N])
                        i==m && γ==γt && (Eit[γ, i, m] = logaddexp(Eit[γ, i, m], βt[m]))
                    end
                end
            end
            βT = βt; TijT = Tijt; EiT = Eit
        end

        #TERMINATION
        Γ = observations[1,o]
        α1oi[o,:] = [log_prob_sum(π0[i], b[i, Γ]) for i in 1:N]
        β1oi[o,:] = βT
        log_pobs[o] = logsumexp(MS_HMMBase.log_prob_sum.(α1oi[o,:], βT[:]))
        Eoi[o,:,:] = [logsumexp([log_prob_sum(EiT[γ,i,m], π0[m], b[m,γt]) for m in 1:N]) for γ in 1:D, i in 1:N]
        Toij[o,:,:] = [logsumexp([log_prob_sum(TijT[i,j,m], π0[m], b[m,γt]) for m in 1:N]) for i in 1:N, j in 1:N]
    end

    #INTEGRATE ACROSS OBSERVATIONS AND SOLVE FOR NEW HMM PARAMS
    new_π0 = zeros(N); new_a = zeros(N,N); new_b = zeros(N,D)
    #SUM ACROSS OBS
    α1i = [logsumexp(α1oi[:,i]) for i in 1:N]; β1i = [logsumexp(β1oi[:,i]) for i in 1:N]
    Ei = [logsumexp(Eoi[:,γ,i]) for γ in 1:D, i in 1:N]; Tij = [logsumexp(Toij[:,i,j]) for i in 1:N, j in 1:N]
    π0_norm = logsumexp([log_prob_sum(α1i[i],β1i[i]) for i in 1:N])

    for i in 1:N
        new_π0[i] = log_prob_sum(α1i[i], β1i[i], -π0_norm)         
        new_a[i,:] = [log_prob_sum(Tij[i,j], -logsumexp(Tij[i,:])) for j in 1:N]
        new_b[i,:] = [log_prob_sum(Ei[γ,i], -logsumexp(Ei[:,i])) for γ in 1:D]
    end

    new_D::Vector{Distribution{F}}=[Categorical(exp.(new_b[i,:])) for i in 1:N]

    return typeof(hmm)(exp.(new_π0), exp.(new_a), new_D), log_prob_sum(log_pobs)
end

function fit_mle_linear!(hmm_jobs::RemoteChannel, output_hmms::RemoteChannel, no_models::Int64, load_table::Vector{Tuple{Int64,Int64,Vector{String},Int64}}; eps_thresh=1e-3, max_iterations=5000, verbose=false)
    while isready(hmm_jobs)
        workerid = myid()
        jobid, start_iterate, hmm, job_norm, observations, wait_secs = MS_HMMBase.load_balancer(no_models, hmm_jobs, load_table[workerid])
        jobid == 0 && break #no valid job for this worker according to load_table entry

        @assert start_iterate < max_iterations - 1
        curr_iterate = start_iterate

        #mask calculations here rather than mle_step to prevent recalculation every iterate
        #build array of observation lengths
        obs_lengths = [findfirst(iszero,observations[:,o])-1 for o in 1:size(observations)[2]]

        start_iterate == 1 && put!(output_hmms, (workerid, jobid, curr_iterate, hmm, 0, 0, false)); #on the first iterate return the initial HMM for the chain right away
        verbose && @info "Fitting HMM, start iterate $start_iterate, job ID $jobid with $(size(hmm.π)[1]) states and $(length(hmm.D[1].support)) symbols..."

        sleep(wait_secs)

        curr_iterate += 1
        if curr_iterate == 2 #no eps value is available
            new_hmm, last_norm = linear_step(hmm, observations, obs_lengths)
            put!(output_hmms, (workerid, jobid, curr_iterate, new_hmm, last_norm, 0, false))
        else #get the eps value from the channel-supplied job value to resume a chain properly
            new_hmm, last_norm = linear_step(hmm, observations, obs_lengths)
            eps = abs(job_norm - last_norm)
            put!(output_hmms, (workerid, jobid, curr_iterate, new_hmm, last_norm, eps, false))
        end

        for i in curr_iterate:max_iterations
            new_hmm, norm = linear_step(new_hmm, observations, obs_lengths)
            curr_iterate += 1
            eps = abs(norm - last_norm)
            if eps < eps_thresh
                put!(output_hmms, (workerid, jobid, curr_iterate, new_hmm, norm, eps, true))
                verbose && @info "Job ID $jobid converged after $(curr_iterate-1) EM steps"
                break
            else
                put!(output_hmms, (workerid, jobid, curr_iterate, new_hmm, norm, eps, false))
                verbose && @info "Job ID $jobid EM step $(curr_iterate-1) eps $eps"
                last_norm = norm
            end
        end
    end
end