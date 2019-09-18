module CLHMM
    import HMMBase:AbstractHMM, HMM
    import Distributions:Distribution,Univariate,Categorical,logpdf
    import Distributed: RemoteChannel, myid
    import StatsFuns: logsumexp, logaddexp

    export linear_step, linear_hmm_converger!, lin_obs_set_lh, lps

    include("linear.jl")
end # module
