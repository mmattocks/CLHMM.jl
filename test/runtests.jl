using CLHMM, Distributed, Distributions, Test, Random, MS_HMMBase
import StatsFuns: logsumexp
include("ref_fns.jl")

Random.seed!(1)

@testset "mle_step functions" begin
    @info "Setting up for MLE function tests.."
    π = fill((1/6),6,6)
    D = [Categorical(ones(4)/4), Categorical([.7,.05,.15,.1]),Categorical([.15,.35,.4,.1]), Categorical([.6,.15,.15,.1]),Categorical([.1,.4,.4,.1]), Categorical([.2,.2,.3,.3])]
    hmm = HMM(π, D)
    log_π = log.(hmm.π)

    obs = zeros(Int64,1,250)
    obs[1:249] = rand(1:4,249)
    obs_lengths=[249]
    @info "Testing mle_step..."

    #verify that above methods independently produce equivalent output, and that this is true of multiple identical obs, but not true of different obs sets
    mouchet_hmm = mouchet_mle_step(hmm, obs[1:249])

    new_hmm = linear_step(hmm, obs, obs_lengths)

    ms_sng = MS_HMMBase.ms_mle_step(hmm, Array(transpose(obs)), obs_lengths)

    dblobs = zeros(Int64, 2,250)
    dblobs[1,1:249] = obs[1:249]
    dblobs[2,1:249] = obs[1:249]
    dblobs_lengths=[249,249]
    dbl_hmm = linear_step(hmm, dblobs, dblobs_lengths)


    ms_dbl =MS_HMMBase.ms_mle_step(hmm, Array(transpose(dblobs)),dblobs_lengths)

    otherobs = deepcopy(dblobs)
    otherobs[2,1:249] = rand(1:4,249)
    if otherobs[1,1]==otherobs[2,1]
        otherobs[1,1]==1&&(otherobs[2,1]=2)
        otherobs[1,1]==2&&(otherobs[2,1]=3)
        otherobs[1,1]==3&&(otherobs[2,1]=4)
        otherobs[1,1]==4&&(otherobs[2,1]=1)
    end
    
    other_hmm = linear_step(hmm, otherobs, dblobs_lengths)

    ms_hmm = MS_HMMBase.ms_mle_step(hmm, Array(transpose(otherobs)),dblobs_lengths)

    for n in fieldnames(typeof(new_hmm[1]))
        if n == :D
            for (d, dist) in enumerate(D)
            @test isapprox(new_hmm[1].D[d].support,mouchet_hmm[1].D[d].support)
            @test isapprox(new_hmm[1].D[d].p,mouchet_hmm[1].D[d].p)
            @test new_hmm[1].D[d].support==ms_sng[1].D[d].support
            @test isapprox(new_hmm[1].D[d].p,ms_sng[1].D[d].p)
            @test new_hmm[1].D[d].support==dbl_hmm[1].D[d].support
            @test isapprox(new_hmm[1].D[d].p,dbl_hmm[1].D[d].p)
            @test new_hmm[1].D[d].support==ms_dbl[1].D[d].support
            @test isapprox(new_hmm[1].D[d].p,ms_dbl[1].D[d].p)
            @test new_hmm[1].D[d].support==other_hmm[1].D[d].support
            @test ms_hmm[1].D[d].support==other_hmm[1].D[d].support
            @test !isapprox(new_hmm[1].D[d].p, other_hmm[1].D[d].p)
            @test isapprox(ms_hmm[1].D[d].p, other_hmm[1].D[d].p, atol=.005)

            end
        else
            @test isapprox(getfield(new_hmm[1],n), getfield(mouchet_hmm[1],n))
            @test isapprox(getfield(new_hmm[1],n), getfield(ms_sng[1],n))

            @test isapprox(getfield(new_hmm[1],n), getfield(dbl_hmm[1],n))
            @test isapprox(getfield(new_hmm[1],n), getfield(ms_dbl[1],n))

            @test !isapprox(getfield(new_hmm[1],n), getfield(other_hmm[1],n))
            @test isapprox(getfield(ms_hmm[1],n), getfield(other_hmm[1],n),atol=.01)

        end
    end

    @test ms_sng[2] == new_hmm[2] == mouchet_hmm[2] != dbl_hmm[2] == ms_dbl[2] != other_hmm[2] == ms_hmm[2]

    @info "Testing fit_mle!..."

    #test fit_mle! function
    input_hmms= RemoteChannel(()->Channel{Tuple}(1))
    output_hmms = RemoteChannel(()->Channel{Tuple}(30))
    jobtuple=("Test",6,0,1)
    put!(input_hmms, (jobtuple, 2, hmm, 0.0, obs))
    linear_hmm_converger!(input_hmms, output_hmms, 1; max_iterations=4, verbose=true)
    wait(output_hmms)
    workerid, jobid, iterate, hmm3, log_p, delta, converged = take!(output_hmms)
    @test jobid == jobtuple
    @test iterate == 3
    @test assert_hmm(hmm3.π0, hmm3.π, hmm3.D)
    @test size(hmm3) == size(hmm) == (6,1)
    @test log_p < 1
    @test log_p == lin_obs_set_lh(hmm, obs)
    @test converged == false
    wait(output_hmms)
    workerid, jobid, iterate, hmm4, log_p, delta, converged = take!(output_hmms)
    @test jobid == jobtuple
    @test iterate == 4
    @test assert_hmm(hmm4.π0, hmm4.π, hmm4.D)
    @test size(hmm4) == size(hmm) == (6,1)
    @test log_p < 1
    @test log_p == lin_obs_set_lh(hmm3, obs)
    @test converged == false

    @info "Test convergence.."
    obs=zeros(Int64, 4, 1001)    
    for o in 1:size(obs)[1]
        obsl=rand(100:1000)
        obs[o,1:obsl]=rand(1:4,obsl)
    end
    input_hmms= RemoteChannel(()->Channel{Tuple}(1))
    output_hmms = RemoteChannel(()->Channel{Tuple}(30))
    put!(input_hmms, (jobtuple, 2, hmm, 0.0, obs))
    linear_hmm_converger!(input_hmms, output_hmms, 1; delta_thresh=.05, max_iterations=100, verbose=true)
    wait(output_hmms)
    workerid, jobid, iterate, hmm4, log_p, delta, converged = take!(output_hmms)
    while isready(output_hmms)
        workerid, jobid, iterate, hmm4, log_p, delta, converged = take!(output_hmms)
    end
    @test converged==1
end