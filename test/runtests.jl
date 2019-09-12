using CLHMM, Distributed, Distributions, HMMBase, Test
import StatsFuns: logsumexp
include("mouchet_fns.jl")

@testset "mle_step functions" begin
    @info "Setting up for MLE function tests.."
    π = [.5 .5
         .5 .5]
    D = [Categorical(ones(4)/4), Categorical([.7,.1,.1,.1])]
    hmm = HMM(π, D)
    log_π = log.(hmm.π)

    obs = zeros(Int64, 22,1)
    obs[1:21] = [4,3,3,2,3,2,1,1,2,3,3,3,4,4,2,3,2,3,4,3,2]
    obs_lengths=[21]
    @info "Testing mle_step..."

    #verify that above methods independently produce equivalent output, and that this is true of multiple identical obs, but not true of different obs sets
    mouchet_hmm = mouchet_mle_step(hmm, obs[1:21])

    new_hmm = linear_step(hmm, obs, obs_lengths)

    dblobs = zeros(Int64, 22,2)
    dblobs[1:21,1] = [4,3,3,2,3,2,1,1,2,3,3,3,4,4,2,3,2,3,4,3,2]
    dblobs[1:21,2] = [4,3,3,2,3,2,1,1,2,3,3,3,4,4,2,3,2,3,4,3,2]
    dblobs_lengths=[21,21]
    dbl_hmm = linear_step(hmm, dblobs, dblobs_lengths)

    otherobs = dblobs
    otherobs[1:21,2] = [1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1]
    other_hmm = linear_step(hmm, otherobs, dblobs_lengths)

    for n in fieldnames(typeof(new_hmm[1]))
        if n == :D
            for (d, dist) in enumerate(D)
            @test isapprox(new_hmm[1].D[d].support,mouchet_hmm[1].D[d].support)
            @test isapprox(new_hmm[1].D[d].p,mouchet_hmm[1].D[d].p)
            @test new_hmm[1].D[d].support==dbl_hmm[1].D[d].support
            @test isapprox(new_hmm[1].D[d].p,dbl_hmm[1].D[d].p)
            @test new_hmm[1].D[d].support==other_hmm[1].D[d].support
            @test !isapprox(new_hmm[1].D[d].p, other_hmm[1].D[d].p)
            end
        else
            @test isapprox(getfield(new_hmm[1],n), getfield(mouchet_hmm[1],n))
            @test isapprox(getfield(new_hmm[1],n), getfield(dbl_hmm[1],n))
            @test !isapprox(getfield(new_hmm[1],n), getfield(other_hmm[1],n))
        end
    end

    @test new_hmm[2] == mouchet_hmm[2] != dbl_hmm[2] != other_hmm[2]

    @info "Testing fit_mle!..."

    #test fit_mle! function
    input_hmms= RemoteChannel(()->Channel{Tuple}(1))
    output_hmms = RemoteChannel(()->Channel{Tuple}(30))
    jobtuple=("Test",2,0,1)
    put!(input_hmms, (jobtuple, 2, hmm, 0.0, obs))
    linear_hmm_converger!(input_hmms, output_hmms, 1, [(1,3,[""],0)]; max_iterations=4, verbose=true)
    wait(output_hmms)
    workerid, jobid, iterate, hmm3, log_p, epsilon, converged = take!(output_hmms)
    @test jobid == jobtuple
    @test iterate == 3
    @test assert_hmm(hmm3.π0, hmm3.π, hmm3.D)
    @test size(hmm3) == size(hmm) == (2,1)
    @test log_p < 1
    @test log_p == lin_obs_set_lh(hmm, obs)
    @test converged == false
    wait(output_hmms)
    workerid, jobid, iterate, hmm4, log_p, epsilon, converged = take!(output_hmms)
    @test jobid == jobtuple
    @test iterate == 4
    @test assert_hmm(hmm4.π0, hmm4.π, hmm4.D)
    @test size(hmm4) == size(hmm) == (2,1)
    @test log_p < 1
    @test log_p == lin_obs_set_lh(hmm3, obs)
    @test converged == false

    @info "Test convergence.."
    obs=zeros(Int64, 101, 2)
    for i in 1:size(obs)[2]
        obs[1:100,i]=rand(1:4,100)
    end
    input_hmms= RemoteChannel(()->Channel{Tuple}(1))
    output_hmms = RemoteChannel(()->Channel{Tuple}(30))
    put!(input_hmms, (jobtuple, 2, hmm, 0.0, obs))
    linear_hmm_converger!(input_hmms, output_hmms, 1, [(1,3,[""],0)]; eps_thresh=.05, max_iterations=100, verbose=true)
    wait(output_hmms)
    workerid, jobid, iterate, hmm4, log_p, epsilon, converged = take!(output_hmms)
    while isready(output_hmms)
        workerid, jobid, iterate, hmm4, log_p, epsilon, converged = take!(output_hmms)
    end
    @test converged==1
end