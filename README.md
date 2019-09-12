## CLHMM
[![Build Status](https://travis-ci.org/mmattocks/CLHMM.jl.svg?branch=master)](https://travis-ci.org/mmattocks/CLHMM.jl)
[![codecov](https://codecov.io/gh/mmattocks/CLHMM.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/mmattocks/CLHMM.jl)

Churbanov Linear Memory Backwards Pass Baum Welch HMM Optimisation
Intended for use with [BGHMM.jl](https://github.com/mmattocks/BGHMM.jl) in distributed training of large model "zoos" of background genomic hidden markov models. Allows observations arrays with multiple observations sequences (intended for categorical dists only- end of sequence is indexed by a trailing zero 0).

So long as tests are passing, mle_step is numerically equivalent to [HMMBase](https://github.com/max_mouchet/HMMBase.jl), except that the transition array in the HMMBase mle_step formula is calculated by Rabiner's formula as indicated here: https://github.com/maxmouchet/HMMBase.jl/issues/10

