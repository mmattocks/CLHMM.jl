using BenchmarkTools, CLHMM, HMMBase, Distributions
using StatsFuns: logsumexp
include("mouchet_fns.jl")
import MS_HMMBase:mle_step

no_samples=100
nevals=2000
obsl=1000

obs = zeros(Int64, obsl+1,1)
obs[1:obsl] = rand(1:4,obsl)
obs_lengths=[obsl]

π = [.5 .5
.5 .5]
D = [Categorical(ones(4)/4), Categorical([.7,.1,.1,.1])]
hmm = HMM(π, D)
log_π = log.(hmm.π)

@info "Judging CLHMM.linear_step vs HMMBase.mle_step"
old_step = median(@benchmark (mouchet_mle_step($hmm, $obs[1:obsl])))
lin_step = median(@benchmark (linear_step($hmm, $obs, $obs_lengths)))
display(judge(old_step,lin_step))

@info "Judging linear_step vs MS"
new_step = median(@benchmark (mle_step($hmm, $obs, $obs_lengths)))
lin_step = median(@benchmark (linear_step($hmm, $obs, $obs_lengths)))
display(judge(new_step,lin_step))