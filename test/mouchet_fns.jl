function mouchet_mle_step(hmm::AbstractHMM{F}, observations) where F
    # NOTE: This function works but there is room for improvement.

    log_likelihoods = mouchet_log_likelihoods(hmm, observations)

    log_α = mouchet_messages_forwards_log(hmm.π0, hmm.π, log_likelihoods)
    log_β = mouchet_messages_backwards_log(hmm.π, log_likelihoods)
    log_π = log.(hmm.π)

    normalizer = logsumexp(log_α[1,:] + log_β[1,:])

    # E-step

    T, K = size(log_likelihoods)
    log_ξ = zeros(T, K, K)

    @inbounds for t = 1:T-1, i = 1:K, j = 1:K
        log_ξ[t,i,j] = log_α[t,i] + log_π[i,j] + log_β[t+1,j] + log_likelihoods[t+1,j] - normalizer
    end

    ξ = exp.(log_ξ)
    ξ ./= sum(ξ, dims=[2,3])

    # M-step
    new_π = sum(ξ[1:end-1,:,:], dims=1)[1,:,:] #index fix-MM
    new_π ./= sum(new_π, dims=2)

    new_π0 = exp.((log_α[1,:] + log_β[1,:]) .- normalizer)
    new_π0 ./= sum(new_π0)

    # TODO: Cleanup/optimize this part
    γ = exp.((log_α .+ log_β) .- normalizer)

    D = Distribution{F}[]
    for (i, d) in enumerate(hmm.D)
        # Super hacky...
        # https://github.com/JuliaStats/Distributions.jl/issues/809
        push!(D, fit_mle(eval(typeof(d).name.name), permutedims(observations), γ[:,i]))
    end

    typeof(hmm)(new_π0, new_π, D), normalizer
end


function mouchet_log_likelihoods(hmm::AbstractHMM{Univariate}, observations)
    hcat(map(d -> logpdf.(d, observations), hmm.D)...)
end



function mouchet_messages_forwards_log(init_distn::AbstractVector{Float64}, trans_matrix::AbstractMatrix{Float64}, log_likelihoods::AbstractMatrix{Float64})
    # OPTIMIZE
    log_alphas = zeros(size(log_likelihoods))
    log_trans_matrix = log.(trans_matrix)
    log_alphas[1,:] = log.(init_distn) .+ log_likelihoods[1,:]
    @inbounds for t = 2:size(log_alphas)[1]
        for i in 1:size(log_alphas)[2]
            log_alphas[t,i] = logsumexp(log_alphas[t-1,:] .+ log_trans_matrix[:,i]) + log_likelihoods[t,i]
        end
    end
    log_alphas
end

function mouchet_messages_backwards_log(trans_matrix::AbstractMatrix{Float64}, log_likelihoods::AbstractMatrix{Float64})
    # OPTIMIZE
    log_betas = zeros(size(log_likelihoods))
    log_trans_matrix = log.(trans_matrix)
    @inbounds for t = size(log_betas)[1]-1:-1:1
        tmp = view(log_betas, t+1, :) .+ view(log_likelihoods, t+1, :)
        @inbounds for i in 1:size(log_betas)[2]
            log_betas[t,i] = logsumexp(view(log_trans_matrix, i, :) .+ tmp)
        end
    end
    log_betas
end