## CLHMM
[![Build Status](https://travis-ci.org/mmattocks/CLHMM.jl.svg?branch=master)](https://travis-ci.org/mmattocks/CLHMM.jl)
[![codecov](https://codecov.io/gh/mmattocks/CLHMM.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/mmattocks/CLHMM.jl)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)

Churbanov Linear Memory Backwards Pass Baum Welch HMM Optimisation
Intended for use with [BGHMM.jl](https://github.com/mmattocks/BGHMM.jl) in distributed training of large model "zoos" of background genomic hidden markov models. Allows observations arrays with multiple observations sequences (intended for categorical dists only- end of sequence is indexed by a trailing zero 0).
