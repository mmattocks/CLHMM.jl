module CLHMM
    import Distributions:Distribution,VariateForm,Univariate,Categorical,logpdf,isprobvec
    import Distributed: RemoteChannel, myid
    import StatsFuns: logsumexp, logaddexp

    export AbstractHMM, HMM, linear_step, linear_hmm_converger!, lin_obs_set_lh, lps

    include("hmm.jl")
    include("linear.jl")
end # module
