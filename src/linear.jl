function linear_step(hmm::AbstractHMM{F}, observations::Matrix{Int64}, obs_lengths::Vector{Int64}) where F
    O = size(observations)[2]
    a = log.(hmm.π); π0 = log.(hmm.π0)
    N = length(hmm.D); D = length(hmm.D[1].support); b = [log(hmm.D[m].p[γ]) for m in 1:N, γ in 1:D]
    α1oi = zeros(O,N); β1oi = zeros(O,N); Eoi = zeros(O,D,N); Toij = zeros(O,N,N); πoi = zeros(O,N); log_pobs=zeros(O); γt=0
    
    for o in 1:O
        #INITIALIZATION
        T = obs_lengths[o]; βT = zeros(N) #log betas at T initialised as zeros
        EiT = fill(-Inf,D,N,N); TijT = fill(-Inf,N,N,N) #Ti,j(T,m) = 0 for all m; in logspace
        
        for m in 1:N, i in 1:N, γ in 1:D
            observations[T, o] == γ && m == i ? (EiT[γ, i, m] = 0) :
                (EiT[γ, i, m] = -Inf) #log Ei initialisation
        end

        #RECURRENCE
        for t in T-1:-1:1
            βt = similar(βT); Tijt = similar(TijT); Eit = similar(EiT)
            Γ = observations[t+1,o]; γt = observations[t,o] 
            for m in 1:N
                βt[m] = logsumexp([lps(a[m,j], b[j,Γ], βT[j]) for j in 1:N])
                for i in 1:N
                    for j in 1:N
                        Tijt[i, j, m] = logsumexp([lps(a[m,n], TijT[i, j, n], b[n,Γ]) for n in 1:N])
                        i==m && (Tijt[i, j, m] = logaddexp(Tijt[i, j, m], lps(βT[j], a[m,j], b[j, Γ])))
                    end
                    for γ in 1:D
                        Eit[γ, i, m] = logsumexp([lps(b[n,Γ], a[m,n], EiT[γ, i, n]) for n in 1:N])
                        i==m && γ==γt && (Eit[γ, i, m] = logaddexp(Eit[γ, i, m], βt[m]))
                    end
                end
            end
            βT = βt; TijT = Tijt; EiT = Eit
        end

        #TERMINATION
        Γ = observations[1,o]
        α1oi[o,:] = [lps(π0[i], b[i, Γ]) for i in 1:N]
        β1oi[o,:] = βT
        log_pobs[o] = logsumexp(lps.(α1oi[o,:], βT[:]))
        Eoi[o,:,:] = [logsumexp([lps(EiT[γ,i,m], π0[m], b[m,γt]) for m in 1:N]) for γ in 1:D, i in 1:N]
        Toij[o,:,:] = [logsumexp([lps(TijT[i,j,m], π0[m], b[m,γt]) for m in 1:N]) for i in 1:N, j in 1:N]
    end

    #INTEGRATE ACROSS OBSERVATIONS AND SOLVE FOR NEW HMM PARAMS
    new_π0 = zeros(N); new_a = zeros(N,N); new_b = zeros(N,D)
    #SUM ACROSS OBS
    α1i = [logsumexp(α1oi[:,i]) for i in 1:N]; β1i = [logsumexp(β1oi[:,i]) for i in 1:N]
    Ei = [logsumexp(Eoi[:,γ,i]) for γ in 1:D, i in 1:N]; Tij = [logsumexp(Toij[:,i,j]) for i in 1:N, j in 1:N]
    π0_norm = logsumexp([lps(α1i[i],β1i[i]) for i in 1:N])

    for i in 1:N
        new_π0[i] = lps(α1i[i], β1i[i], -π0_norm)         
        new_a[i,:] = [lps(Tij[i,j], -logsumexp(Tij[i,:])) for j in 1:N]
        new_b[i,:] = [lps(Ei[γ,i], -logsumexp(Ei[:,i])) for γ in 1:D]
    end

    new_D::Vector{Distribution{F}}=[Categorical(exp.(new_b[i,:])) for i in 1:N]

    return typeof(hmm)(exp.(new_π0), exp.(new_a), new_D), lps(log_pobs)
end
                #subfuncs to handle sums of log probabilities that may include -Inf (ie p=0), returning -Inf in this case rather than NaNs
                function lps(adjuvants::AbstractArray)
                    prob = sum(adjuvants) ; isnan(prob) ? - Inf : prob
                end
                
                function lps(base, adjuvants...)::Float64
                    prob = base+sum(adjuvants) ; isnan(prob) ? -Inf : prob
                end


function linear_hmm_converger!(hmm_jobs::RemoteChannel, output_hmms::RemoteChannel, no_models::Int64, load_table::Vector{Tuple{Int64,Int64,Vector{String},Int64}}; eps_thresh=1e-3, max_iterations=5000, verbose=false)
    while isready(hmm_jobs)
        workerid = myid()
        jobid, start_iterate, hmm, job_norm, observations, wait_secs = load_balancer(no_models, hmm_jobs, load_table[workerid])
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
                #subfunc to handle balancing memory load on dissimilar machines in cluster
                function load_balancer(no_models::Int64, hmm_jobs::RemoteChannel, load_params::Tuple{Int64,Int64,Vector{String}, Int64})
                    lb_kmin::Int64,lb_kmax::Int64,blacklist::Vector{String},wait_secs::Int64 = load_params
                    lb_job_counter = 1
                    
                    jobid::Tuple{String, Int64, Int64, Int64}, start_iterate::Int64, hmm::HMM, job_norm::Float64, observations::Matrix{Int64} = take!(hmm_jobs)

                    while (jobid[2] < lb_kmin || jobid[2] > lb_kmax || jobid[1] in blacklist) && lb_job_counter <= no_models #while a job prohibited by load table, keep putting the job back and drawing a new one
                        put!(hmm_jobs, (jobid, start_iterate, hmm, observations))
                        jobid, start_iterate, hmm, job_norm, observations = take!(hmm_jobs)
                        lb_job_counter += 1
                    end
                    lb_job_counter > no_models ? (return 0,0,0,0,0) : (return jobid, start_iterate, hmm, job_norm, observations, wait_secs)
                end

function lin_obs_set_lh(hmm::HMM{Univariate,Float64}, observations::Matrix{Int64})
    O = size(observations)[2]; obs_lengths = [findfirst(iszero,observations[:,o])-1 for o in 1:size(observations)[2]]
    a = log.(hmm.π); π0 = log.(hmm.π0)
    N = length(hmm.D); D = length(hmm.D[1].support); b = [log(hmm.D[m].p[γ]) for m in 1:N, γ in 1:D]
    α1oi = zeros(O,N); β1oi = zeros(O,N); Eoi = zeros(O,D,N); Toij = zeros(O,N,N); πoi = zeros(O,N); log_pobs=zeros(O); γt=0
    
    for o in 1:O
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