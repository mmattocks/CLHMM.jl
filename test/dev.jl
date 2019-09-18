using CLHMM,HMMBase,Serialization,Distributions, BenchmarkTools, Test,Random,MS_HMMBase,Profile,ProfileView
import StatsFuns:logsumexp, logaddexp
include("ref_fns.jl")
Random.seed!(1)
dict=deserialize("/bench/PhD/NGS_binaries/BGHMM/hmmchains.sat.bak")
hmm=dict[("periexonic", 6, 1, 1)][1][2]
no_obs=200
observations=zeros(Int64,no_obs,1001)
obs_lengths=Vector{Int64}()
for o in 1:size(observations)[1]
    obsl=rand(100:1000)
    push!(obs_lengths,obsl)
    observations[o,1:obsl]=rand(1:16,obsl)
end

new_hmm, new_logpobs = linear_step(hmm, observations, obs_lengths)
old_hmm, old_logpobs = old_linear(hmm, Array(transpose(observations)),obs_lengths)
MS_hmm, MS_logpobs = MS_HMMBase.mle_step(hmm, Array(transpose(observations)),obs_lengths)

for n in fieldnames(typeof(new_hmm))
    if n == :D
        for (d, dist) in enumerate(new_hmm.D)
            @test new_hmm.D[d].support==old_hmm.D[d].support==MS_hmm.D[d].support
            #@test isapprox(new_hmm.D[d].support,old_hmm.D[d].support)
            @test isapprox(new_hmm.D[d].p,old_hmm.D[d].p)
            #@test isapprox(new_hmm.D[d].p,MS_hmm.D[d].p)
        end
    else
        @test isapprox(getfield(new_hmm,n), getfield(old_hmm,n))
        #@test isapprox(getfield(new_hmm,n), getfield(MS_hmm,n))
    end
end

@test isapprox(new_logpobs,old_logpobs)
@test isapprox(new_logpobs,MS_logpobs)

@info "Judging new linear vs old B-W"
new = median(@benchmark (linear_step($hmm, $observations, $obs_lengths)))
ms = median(@benchmark (MS_HMMBase.mle_step($hmm, $Array(transpose(observations)), $obs_lengths)))
display(judge(new,ms))
@info "Judging old linear vs old B-W"
old = median(@benchmark (old_linear($hmm, $Array(transpose(observations)), $obs_lengths)))
display(judge(old,ms))
@info "Judging new linear vs old linear"
display(judge(new,old))