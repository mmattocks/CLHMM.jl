using CLHMM,HMMBase,Serialization,Distributions, BenchmarkTools, Test
import StatsFuns:logsumexp, logaddexp

dict=deserialize("/bench/PhD/NGS_binaries/BGHMM/hmmchains.sat.bak")
hmm=dict[("periexonic", 6, 1, 1)][1][2]
no_obs=10
observations=zeros(Int64,no_obs,1001)
obs_lengths=Vector{Int64}()
for o in 1:size(observations)[1]
    obsl=rand(no_obs:1000)
    push!(obs_lengths,obsl)
    observations[o,1:obsl]=rand(1:4,obsl)
end

function old_linear(hmm::AbstractHMM{F}, observations::Matrix{Int64}, obs_lengths::Vector{Int64}) where F
    O = size(observations)[2]
    a = log.(hmm.π); π0 = log.(hmm.π0)
    N = length(hmm.D); D = length(hmm.D[1].support); b = [log(hmm.D[m].p[γ]) for m in 1:N, γ in 1:D]
    α1oi = zeros(O,N); β1oi = zeros(O,N); Eoi = zeros(O,D,N); Toij = zeros(O,N,N); πoi = zeros(O,N); log_pobs=zeros(O); γt=0
    track=[]
    
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
            βT = copy(βt); TijT = copy(Tijt); EiT = copy(Eit); track=copy(TijT)
        end

        #TERMINATION
        Γ = observations[1,o]
        α1oi[o,:] = [lps(π0[i], b[i, Γ]) for i in 1:N]
        β1oi[o,:] = βT
        log_pobs[o] = logsumexp(lps.(α1oi[o,:], βT[:]))
        Eoi[o,:,:] = [logsumexp([lps(EiT[γ,i,m], π0[m], b[m,γt]) for m in 1:N]) for γ in 1:D, i in 1:N]
        Toij[o,:,:] = [logsumexp([lps(TijT[i,j,m], π0[m], b[m,γt]) for m in 1:N]) for i in 1:N, j in 1:N]
    end

    return β1oi, track, Toij, Eoi
end

function old_tij_method(hmm::AbstractHMM{F}, observations::Matrix{Int64}, obs_lengths::Vector{Int64}) where F
    O = size(observations)[2]
    a = log.(hmm.π); π0 = log.(hmm.π0)
    N = length(hmm.D); D = length(hmm.D[1].support); b = [log(hmm.D[m].p[γ]) for m in 1:N, γ in 1:D]
    Toij = zeros(O,N,N,N); πoi = zeros(O,N); γt=0
    βo=zeros(O,N)
    track=[]
    
    for o in 1:O
        #INITIALIZATION
        T = obs_lengths[o]
        TijT = fill(-Inf,N,N,N) #Ti,j(T,m) = 0 for all m; in logspace
        βT = zeros(N) #log betas at T initialised as zeros

        #RECURRENCE
        for t in T-1:-1:1
            βt = similar(βT);Tijt = similar(TijT);
            Γ = observations[t+1,o]; γt = observations[t,o] 
            for m in 1:N
                βt[m] = logsumexp([lps(a[m,j], b[j,Γ], βT[j]) for j in 1:N])
                for i in 1:N
                    for j in 1:N
                        Tijt[i, j, m] = logsumexp([lps(a[m,n], TijT[i, j, n], b[n,Γ]) for n in 1:N])
                        i==m && (Tijt[i, j, m] = logaddexp(Tijt[i, j, m], lps(βT[j], a[m,j], b[j, Γ])))
                    end
                end
            end
            βT=copy(βt);TijT=copy(Tijt)
        end
        #TERMINATION
        Toij[o,:,:,:] = [lps(TijT[i,j,m], π0[m], b[m,γt]) for i in 1:N, j in 1:N, m in 1:N]
        βo[o,:]=βT
        
    end
    return Toij,βo
   
end

function new_tij_method(hmm::HMM{Univariate,Float64}, observations::Matrix{Int64}, obs_lengths::Vector{Int64})
    D = length(hmm.D[1].support);O = size(observations)[1];
    a = log.(hmm.π); π0 = transpose(log.(hmm.π0))
    N = length(hmm.D); 
    mask=observations.!=0
    
    #INITIALIZATION
    βT = zeros(O,N); βt = zeros(O,N) #log betas at T initialised as zeros
    TijT = fill(-Inf,O,N,N,N); Tijt = fill(-Inf,O,N,N,N) #Ti,j(T,m) = 0 for all m; in logspace
        

    #RECURRENCE
    for t in maximum(obs_lengths)-1:-1:1
        omask = mask[:,t+1]
        lls = llhs(hmm,observations[:,t+1])
        zo=copy(βT)
        βT[omask,:] .+= view(lls,omask,:)
        TijT[omask,:,:,:] .+= view(lls,omask,:)
        for m in 1:N
            trans_view = transpose(view(a,m,:))
            βt[omask,m] = logsumexp.(eachrow(βT[omask,:].+trans_view))
            for i in 1:N
                for j in 1:N
                    Tijt[omask, i, j, m] .= logsumexp.(eachrow(view(TijT, omask, i, j, :).+trans_view))
                    i==m && (Tijt[omask, i, j, m] .= logaddexp.(view(Tijt,omask,i, j, m), ((zo[omask,j].+a[m,j]).+lls[omask,j])))
                end
            end
        end
        βT=copy(βt); TijT=copy(Tijt)
    end

    #To = [logsumexp.([lps.(TijT[o,i,j,m], αo[o,m]) for m in 1:N]) for o in 1:O, i in 1:N, j in 1:N]

    return TijT,βT
end



function test_bw(hmm::HMM{Univariate,Float64}, observations::Matrix{Int64}, obs_lengths::Vector{Int64})
    D = length(hmm.D[1].support);O = size(observations)[1];
    a = log.(hmm.π); π0 = transpose(log.(hmm.π0))
    N = length(hmm.D); b = [log(hmm.D[m].p[γ]) for m in 1:N, γ in 1:D]
    mask=observations.!=0
    
    #INITIALIZATION
    βT = zeros(O,N); βt = zeros(O,N) #log betas at T initialised as zeros
    EiT = fill(-Inf,O,N,N,D); Eit = fill(-Inf,O,N,N,D)
    TijT = fill(-Inf,O,N,N,N); Tijt = fill(-Inf,O,N,N,N) #Ti,j(T,m) = 0 for all m; in logspace
        
    for m in 1:N, i in 1:N, γ in 1:D, o in 1:O
        observations[o, obs_lengths[o]] == γ && m == i && (EiT[o, i, m, γ] = 0)
    end

    #RECURRENCE
    for t in maximum(obs_lengths)-1:-1:1
        omask = mask[:,t+1]
        lls = llhs(hmm,observations[:,t+1])
        zo=copy(βT)
        βT[omask,:] .+= view(lls,omask,:)
        TijT[omask,:,:,:] = lps.(TijT[omask,:,:,:], view(lls,omask,:))
        EiT[omask,:,:,:] = lps.(EiT[omask,:,:,:], view(lls,omask,:))
        for m in 1:N
            trans_view = transpose(view(a,m,:))
            βt[omask,m] = logsumexp.(eachrow(view(βT,omask,:).+trans_view))
            for i in 1:N
                for j in 1:N
                    Tijt[omask, i, j, m] .= logsumexp.(eachrow(lps.(view(TijT, omask, i, j, m),trans_view)))
                    i==m && (Tijt[omask, i, j, m] .= logaddexp.(view(Tijt,omask,i, j, m), lps.(zo[j],a[m,j],lls[omask,j])))
                end
                for γ in 1:D
                    Eit[omask, i, m, γ] .= logsumexp.(eachrow(lps.(view(EiT,omask,i,:,γ),trans_view)))
                    if i==m
                        symmask = observations[:,t].==γ
                        Eit[symmask, i, m, γ] .= logaddexp.(view(Eit, symmask,i, m, γ), view(βt,symmask,m))
                    end
                end
            end
        end
        βT = copy(βt); TijT = copy(Tijt); EiT = copy(Eit)
    end

    # #TERMINATION
    lls = llhs(hmm,observations[:,1])
    αo = lls .+ π0
    # log_pobs = logsumexp.(eachcol(lps.(αo, βT)))
    # Eo = [logsumexp([lps(EiT[o,i,m,γ], αo[o,m]) for m in 1:N]) for γ in 1:D, i in 1:N, o in 1:O]
    To = [logsumexp.([lps.(TijT[o,i,j,m], αo[o,m]) for m in 1:N]) for o in 1:O, i in 1:N, j in 1:N]
    return βT, TijT, To, EiT
    # #INTEGRATE ACROSS OBSERVATIONS AND SOLVE FOR NEW HMM PARAMS
    # new_π0 = zeros(N); new_a = zeros(N,N); new_b = zeros(N,D)
    # #SUM ACROSS OBS
    # α = [logsumexp(αo[:,i]) for i in 1:N]; β = [logsumexp(βT[:,i]) for i in 1:N]
    # E = [logsumexp(Eo[:,i,γ]) for γ in 1:D, i in 1:N]; Tij = [logsumexp(To[:,i,j]) for i in 1:N, j in 1:N]
    # π0_norm = logsumexp([lps(α[i],β[i]) for i in 1:N])

    # for i in 1:N
    #     new_π0[i] = lps(α[i], β[i], -π0_norm)         
    #     new_a[i,:] = [lps(Tij[i,j], -logsumexp(Tij[i,:])) for j in 1:N]
    #     new_b[i,:] = [lps(E[γ,i], -logsumexp(E[:,i])) for γ in 1:D]
    # end

    # new_D::Vector{Categorical}=[Categorical(exp.(new_b[i,:])) for i in 1:N]

    # println(new_π0)
    # println(new_a)

    # return typeof(hmm)(exp.(new_π0), exp.(new_a), new_D), lps(log_pobs)
end

function llhs(hmm::AbstractHMM{Univariate}, observation::Vector{Int64})
    lls = zeros(length(observation),length(hmm.D))
    for d in 1:length(hmm.D)
        lls[:,d] = logpdf.(hmm.D[d], observation)
    end
    return lls
end
 new_tij, new_bo = new_tij_method(hmm, observations, obs_lengths)
old_tij, old_bo = old_tij_method(hmm, Array(transpose(observations)), obs_lengths)

 @test new_bo == old_bo
 @test new_tij == old_tij


# new_b, new_tij, new_t, new_e = test_bw(hmm, observations, obs_lengths)

# old_b, old_tij, old_t, old_e = old_linear(hmm, Array(transpose(observations)),obs_lengths)

# println("NEW")
# println(new_t)
# println("OLD")
# println(old_t)

#@test new_b == old_b == new_bo == old_bo

# @test new_tij == old_tij
# @test new_t == old_t
# @test new_e == old_e

# println(old_betas)
# for n in fieldnames(typeof(dev_hmm[1]))
#     if n == :D
#         for (d, dist) in enumerate(D)
#         @test isapprox(dev_hmm[1].D[d].support,linear_hmm[1].D[d].support)
#         @test isapprox(dev_hmm[1].D[d].p,linear_hmm[1].D[d].p)
#         end
#     else
#         @test isapprox(getfield(dev_hmm[1],n), getfield(linear_hmm[1],n))
#     end
# end

# @test dev_hmm[2] == linear_hmm[2]
