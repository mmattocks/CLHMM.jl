using BenchmarkTools, CLHMM, MS_HMMBase, Distributions
using StatsFuns: logsumexp
include("mouchet_fns.jl")
import HMMBase:HMM

no_samples=100
nevals=2000
obsl=1000

obs = zeros(Int64, obsl+1,1)
obs[1:obsl] = rand(1:4,obsl)
obs_lengths=[obsl]

π = [.5 .5
.5 .5]
D = [Categorical(ones(4)/4), Categorical([.7,.1,.1,.1])]
m_hmm = HMMBase.HMM(π, D)
ms_hmm = MS_HMMBase.HMM(π,D)

@info "Judging CLHMM.linear_step vs HMMBase.mle_step"
old_step = median(@benchmark (mouchet_mle_step($m_hmm, $obs[1:obsl])))
lin_step = median(@benchmark (linear_step($m_hmm, $obs, $obs_lengths)))
display(judge(old_step,lin_step))

@info "Judging linear_step vs MS"
new_step = median(@benchmark (MS_HMMBase.mle_step($ms_hmm, $obs, $obs_lengths)))
display(judge(new_step,lin_step))

@info "Judging linear_step vs threaded_linear_step"
threaded_step = median(@benchmark (CLHMM.threaded_linear_step($m_hmm, $obs, $obs_lengths)))
display(judge(threaded_step,lin_step))