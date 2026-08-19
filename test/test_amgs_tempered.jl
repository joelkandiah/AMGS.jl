"""
test_amgs_tempered.jl

Tests for `AMGS_Tempered_Sampler`.

At `temperature = 1` this must reduce exactly to base AMGS, so it runs the full
shared contract. The tempering itself is checked analytically: for a Gaussian
target, raising the density to the power `temp` gives another Gaussian with the
same mean and covariance scaled by `1/temp`, so the draws have a precisely known
distribution at every temperature.
"""

@testset "AMGS_Tempered_Sampler" begin
    # temperature = 1 is exactly the untempered posterior.
    test_sampler_contract("contract (temperature = 1)", () -> AMGS_Tempered_Sampler(0.234, 0.6))

    @testset "tempering rescales a Gaussian target by 1/temp" begin
        # π(x)^temp for x ~ N(m, Σ) is N(m, Σ/temp).
        for temp in (0.5, 2.0)
            spl = AMGS_Tempered_Sampler(0.234, 0.6; temperature=temp)
            ch = sample(
                StableRNG(SEED), corr_gaussian(), spl, 30_000;
                num_warmup=10_000, progress=false,
            )
            X = vecs(ch, @varname(x))
            expected_cov = CORR_COV ./ temp

            # Mean is unchanged by tempering.
            @test isapprox(vec(mean(X; dims=2)), CORR_MEAN; atol=0.2 / sqrt(temp))
            # Covariance is scaled by exactly 1/temp.
            @test isapprox(var(X[1, :]), expected_cov[1, 1]; rtol=0.25)
            @test isapprox(var(X[2, :]), expected_cov[2, 2]; rtol=0.25)
            @test isapprox(cov(X[1, :], X[2, :]), expected_cov[1, 2]; rtol=0.4)
        end
    end

    @testset "temperature = 1 matches the untempered sampler" begin
        # The tempered target at temp=1 is getlogjoint_internal, so the two
        # samplers must agree in distribution on a constrained model.
        mdl = normal_invgamma(YDATA)
        ch_t = sample(
            StableRNG(SEED), mdl, AMGS_Tempered_Sampler(0.234, 0.6), 20_000;
            num_warmup=5_000, progress=false,
        )
        ch_b = sample(
            StableRNG(SEED), mdl, AMGS_Sampler(0.234, 0.6), 20_000;
            num_warmup=5_000, progress=false,
        )
        @test isapprox(
            mean(scal(ch_t, @varname(m))), mean(scal(ch_b, @varname(m))); atol=0.1
        )
        @test isapprox(
            mean(scal(ch_t, @varname(s))), mean(scal(ch_b, @varname(s))); atol=0.3
        )
    end

    @testset "temperature < 1 flattens a constrained posterior" begin
        # Flattening must widen the posterior, and must keep the support valid.
        mdl = normal_invgamma(YDATA)
        hot = sample(
            StableRNG(SEED), mdl, AMGS_Tempered_Sampler(0.234, 0.6; temperature=0.3),
            20_000; num_warmup=5_000, progress=false,
        )
        cold = sample(
            StableRNG(SEED), mdl, AMGS_Tempered_Sampler(0.234, 0.6; temperature=1.0),
            20_000; num_warmup=5_000, progress=false,
        )
        @test all(scal(hot, @varname(s)) .> 0)          # link transform still holds
        @test std(scal(hot, @varname(m))) > std(scal(cold, @varname(m)))
    end

    @testset "tempered log-density is reported on every transition" begin
        ch = sample(
            StableRNG(SEED), corr_gaussian(),
            AMGS_Tempered_Sampler(0.234, 0.6; temperature=0.5), 500;
            num_warmup=200, progress=false,
        )
        lp = vec(ch[:lp_tempered])
        @test length(lp) == 500
        @test all(isfinite, lp)
        # The untempered decomposition is still carried alongside it.
        @test all(isfinite, vec(ch[:logjoint]))
        # Tempering at 0.5 halves the log joint (this model is prior-only).
        @test isapprox(lp, 0.5 .* vec(ch[:logjoint]); rtol=1e-8)
    end

    @testset "constructors" begin
        s = AMGS_Tempered_Sampler(0.234, 0.6; temperature=0.5)
        @test s.temperature == 0.5
        @test AMGS_Tempered_Sampler(0.234, 0.6).temperature == 1.0
        @test AMGS_Tempered_Sampler(0.234, 0.6, true, 1e-4; temperature=2.0).temperature == 2.0
        @test_throws ArgumentError AMGS_Tempered_Sampler(0.234, 0.6; temperature=0.0)
        @test_throws ArgumentError AMGS_Tempered_Sampler(0.234, 0.6; temperature=-1.0)
    end
end
