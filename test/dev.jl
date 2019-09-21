using CLHMM,HMMBase,Serialization,Distributions, BenchmarkTools, Test,Random,MS_HMMBase,Profile,ProfileView
import StatsFuns:logsumexp, logaddexp
include("ref_fns.jl")
Random.seed!(1)
π = fill((1/6),6,6)
D = [Categorical(ones(4)/4), Categorical([.7,.05,.15,.1]),Categorical([.15,.35,.4,.1]), Categorical([.6,.15,.15,.1]),Categorical([.1,.4,.4,.1]), Categorical([.2,.2,.3,.3])]
hmm = HMM(π, D)
log_π = log.(hmm.π)

#function modstep(hmm::HMM{Univariate,Float64}, observations::Matrix{Int64}, obs_lengths::Vector{Int64})
    # O,T = size(observations);
    # a = log.(hmm.π); π0 = transpose(log.(hmm.π0))
    # N = length(hmm.D); Γ = length(hmm.D[1].support);
    # mask=observations.!=0
    # #INITIALIZATION
    # βoi_T = zeros(O,N); βoi_t = zeros(O,N) #log betas at T initialised as zeros
    # Eoγim_T = fill(-Inf,O,Γ,N,N); Eoγim_t = fill(-Inf,O,Γ,N,N)
    # @inbounds for m in 1:N, i in 1:N, γ in 1:Γ, o in 1:O
    #     observations[o, obs_lengths[o]] == γ && m == i && (Eoγim_T[o, γ, i, m] = 0)
    # end
    # Tijm_T = fill(-Inf,O,N,N,N); Tijm_t = fill(-Inf,O,N,N,N) #Ti,j(T,m) = 0 for all m; in logspace
        
#     #RECURRENCE
#     modsweep!(hmm,a,N,Γ,βoi_T,βoi_t,Tijm_T,Tijm_t,Eoγim_T,Eoγim_t,observations,mask,obs_lengths)

#     #TERMINATION
#     lls = CLHMM.llhs(hmm,observations[:,1])
#     α1om = lls .+ π0 #first position forward msgs
#     Toij = [logsumexp([lps(view(Tijm_T,o,i,j,m), view(α1om,o,m)) for m in 1:N]) for o in 1:O, i in 1:N, j in 1:N] #terminate Tijs with forward messages
#     Eoiγ=[logsumexp([lps(view(Eoγim_T,o,γ,i,m), view(α1om,o,m)) for m in 1:N]) for o in 1:O, i in 1:N, γ in 1:Γ] #terminate Eids with forward messages

#     #INTEGRATE ACROSS OBSERVATIONS AND SOLVE FOR NEW HMM PARAMS
#     obs_penalty=log(O) #broadcast subtraction to normalise log prob vals by obs number
#     #INITIAL STATE DIST
#     π0_o=α1om.+βoi_T.-logsumexp.(eachrow(α1om.+βoi_T)) #estimate π0 for each o
#     new_π0=logsumexp.(eachcol(π0_o)).-obs_penalty #sum over obs and normalise by number of obs
#     #TRANSITION MATRIX
#     a_int = Toij.-logsumexp.([view(Toij,o,i,:) for o in 1:O, i in 1:N])
#     new_a = logsumexp.([a_int[:,i,j] for i in 1:N, j in 1:N]).-obs_penalty
#     #EMISSION MATRIX
#     e_int=Eoiγ.-logsumexp.([view(Eoiγ,o,j,:) for o in 1:O, j in 1:N])
#     new_b=logsumexp.([view(e_int,:,j,γ) for γ in 1:Γ, j in 1:N]).-obs_penalty
#     new_D::Vector{Categorical}=[Categorical(exp.(new_b[:,i])) for i in 1:N]

#     println(new_a)

#     return typeof(hmm)(exp.(new_π0), exp.(new_a), new_D), lps([logsumexp(lps.(α1om[o,:], βoi_T[o,:])) for o in 1:O])
# end


function modsweep!(hmm::HMM{Univariate,Float64}, a::Matrix{Float64},N::Int64,Γ::Int64,βoiT::Matrix{Float64},βoit::Matrix{Float64},TijmT::Array{Float64},Tijmt::Array{Float64},EoγimT::Array{Float64},Eoγimt::Array{Float64}, observations::Matrix{Int64}, mask::BitMatrix, obs_lengths::Vector{Int64})
    @inbounds for t in maximum(obs_lengths)-1:-1:1
        lls = CLHMM.llhs(hmm,observations[:,t+1])
        omask = findall(mask[:,t+1])
        for m in 1:N
            βoit[omask,m] .= logsumexp.(eachrow((view(βoiT,omask,:).+view(lls,omask,:)).+transpose(view(a,m,:))))
            for j in 1:N, i in 1:N
                Tijmt[omask, i, j, m] .= logsumexp.(eachrow(lps.(view(TijmT,omask,i,j,:), view(lls,omask,:), transpose(view(a,m,:)))))
                i==m && (Tijmt[omask, i, j, m] .= logaddexp.(Tijmt[omask, i, j, m], (βoiT[omask,j].+ a[m,j].+ lls[omask,j])))
            end
            for i in 1:N, γ in 1:Γ
                Eoγimt[omask, γ, i, m] .= logsumexp.(eachrow(lps.(view(EoγimT,omask,γ,i,:),view(lls,omask,:),transpose(view(a,m,:)))))
                if i==m
                    symmask = findall(observations[:,t].==γ)
                    Eoγimt[symmask, γ, i, m] .= logaddexp.(Eoγimt[symmask, γ, i, m], βoit[symmask,m])
                end
            end
        end
        βoiT=copy(βoit); TijmT=copy(Tijmt); EoγimT = copy(Eoγimt);
    end
    return βoiT, TijmT, EoγimT
end

no_obs=500
observations=zeros(Int64,no_obs,1001)
obs_lengths=Vector{Int64}()
for o in 1:size(observations)[1]
    obsl=rand(100:1000)
    push!(obs_lengths,obsl)
    observations[o,1:obsl]=rand(1:4,obsl)
end

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


# linear_step(hmm,observations,obs_lengths)

# Profile.clear()

# @profile linear_step(hmm,observations,obs_lengths)
# ProfileView.view()

@info "test sweeps"
ob, oa, od = CLHMM.backwards_sweep!(hmm,a,N,Γ,copy(βoi_T),copy(βoi_t),copy(Tijm_T),copy(Tijm_t),copy(Eoγim_T),copy(Eoγim_t),observations,mask,obs_lengths)
nb, na, nd = modsweep!(hmm,a,N,Γ,copy(βoi_T),copy(βoi_t),copy(Tijm_T),copy(Tijm_t),copy(Eoγim_T),copy(Eoγim_t),observations,mask,obs_lengths)

@test ob==nb
@test oa==na
@test od==nd

@info "bench sweeps"
modsweep = median(@benchmark (modsweep!(hmm,a,N,Γ,copy(βoi_T),copy(βoi_t),copy(Tijm_T),copy(Tijm_t),copy(Eoγim_T),copy(Eoγim_t),observations,mask,obs_lengths)))
sweep = median(@benchmark ( CLHMM.backwards_sweep!(hmm,a,N,Γ,copy(βoi_T),copy(βoi_t),copy(Tijm_T),copy(Tijm_t),copy(Eoγim_T),copy(Eoγim_t),observations,mask,obs_lengths)))

display(judge(modsweep,sweep))


# new_hmm, new_logpobs = linear_step(hmm, observations, obs_lengths)
# println(new_hmm.π)
# mod_hmm, mod_pobs = modstep(hmm, observations, obs_lengths)

# modsweep = median(@benchmark (modstep(hmm, observations, obs_lengths)))
# sweep = median(@benchmark (linear_step(hmm, observations, obs_lengths)))
# display(judge(modsweep,sweep))
# old_hmm, old_logpobs = old_linear(hmm, Array(transpose(observations)),obs_lengths)
# MS_hmm, MS_logpobs = MS_HMMBase.mle_step(hmm, Array(transpose(observations)),obs_lengths)

# for n in fieldnames(typeof(new_hmm))
#     if n == :D
#         for (d, dist) in enumerate(new_hmm.D)
#             @test new_hmm.D[d].support==mod_hmm.D[d].support #/==MS_hmm.D[d].support
#             #@test isapprox(new_hmm.D[d].support,old_hmm.D[d].support)=
#             @test isapprox(new_hmm.D[d].p,mod_hmm.D[d].p)
#             #@test isapprox(new_hmm.D[d].p,MS_hmm.D[d].p)
#         end
#     else
#         @test isapprox(getfield(new_hmm,n), getfield(mod_hmm,n))
#         #@test isapprox(getfield(new_hmm,n), getfield(MS_hmm,n))
#     end
# end

# @test isapprox(new_logpobs,mod_pobs)
# @test isapprox(new_logpobs,MS_logpobs)
# @info "Judging new linear vs old linear"

# @info "Judging new linear vs old B-W"
# new = median(@benchmark (modstep($hmm, $observations, $obs_lengths)))
# ms = median(@benchmark (MS_HMMBase.mle_step($hmm, $Array(transpose(observations)), $obs_lengths)))
# display(judge(new,ms))
# @info "Judging old linear vs old B-W"
# old = median(@benchmark (linear_step($hmm, $observations, $obs_lengths)))
# display(judge(old,ms))
# display(judge(new,old))

# obsl=1500
# sorted_obs = zeros(Int64, 1000,obsl+1)
# sorted_obsl = [i for i in obsl:-1:obsl-size(sorted_obs)[1]]
# unsorted_obs = zeros(Int64, 1000,obsl+1)
# unsorted_obsl = Vector{Int64}()
# for o in 1:size(sorted_obs)[1]
#     sorted_obs[o,1:obsl]=rand(1:4,obsl)
#     unsorted_obs[rand(findall(iszero,unsorted_obs[:,1])), 1:obsl]=rand(1:4,obsl)
#     push!(unsorted_obsl,obsl)
#     global obsl-=1
# end

# O,T = size(unsorted_obs);
# a = log.(hmm.π); π0 = transpose(log.(hmm.π0))
# N = length(hmm.D); D = length(hmm.D[1].support);
# umask=unsorted_obs.!=0
# smask=sorted_obs.!=0
# #INITIALIZATION
# βoi_T = zeros(O,N); βoi_t = zeros(O,N) #log betas at T initialised as zeros
# Eoγim_T = fill(-Inf,O,N,N,D); Eoγim_t = fill(-Inf,O,N,N,D)
# mEoimd_T = fill(-Inf,O,D,N,N); mEoimd_t = fill(-Inf,O,D,N,N)
# @inbounds for m in 1:N, i in 1:N, γ in 1:D, o in 1:O
#     unsorted_obs[o, unsorted_obsl[o]] == γ && m == i && (Eoγim_T[o, i, m, γ] = 0)
# end
# Tijm_T = fill(-Inf,O,N,N,N); Tijm_t = fill(-Inf,O,N,N,N) #Ti,j(T,m) = 0 for all m; in logspace

# @info "Judging modsweep vs sweep"
# modsweep = median(@benchmark (modsweep!(hmm,a,N,D,βoi_T,βoi_t,Tijm_T,Tijm_t,mEoimd_T,mEoimd_t,sorted_obs,smask,sorted_obsl)))
# sweep = median(@benchmark (CLHMM.backwards_sweep!(hmm,a,N,D,βoi_T,βoi_t,Tijm_T,Tijm_t,Eoγim_T,Eoγim_t,unsorted_obs,umask,unsorted_obsl)))
# display(judge(modsweep,sweep))
    
