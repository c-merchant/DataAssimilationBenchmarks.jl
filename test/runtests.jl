##############################################################################################
module runtests
##############################################################################################
# imports and exports
using Test
using JLD2
##############################################################################################
# include test sub-modules
include("TestDataAssimilationBenchmarks.jl")
include("TestObsOperators.jl")
include("TestVarAD.jl")
include("TestDeSolvers.jl")
include("TestL96.jl")
include("TestGenerateTimeSeries.jl")
include("TestIEEE39bus.jl")
include("TestFilterExps.jl")
include("TestClassicSmootherExps.jl")
include("TestIterativeSmootherExps.jl")
include("TestSingleIterationSmootherExps.jl")
include("TestParallelExperimentDriver.jl")
##############################################################################################
# Run tests

@testset "ParentModule" begin
    @test TestDataAssimilationBenchmarks.splash()
end

# Test Observation Operators jacobian
@testset "Observation Operators" begin
    @test TestObsOperators.alternating_obs_jacobian_pos()
    @test TestObsOperators.alternating_obs_jacobian_zero()
    @test TestObsOperators.alternating_obs_jacobian_neg()
end

# Calculate the order of convergence for standard integrators
@testset "Calculate Order Convergence" begin
    @test TestDeSolvers.testEMExponential()
    @test TestDeSolvers.testRKExponential()
end

# Test L96 model equations for known behavior
@testset "Lorenz-96" begin
    @test TestL96.Jacobian()
    @test TestL96.EMZerosStep()
    @test TestL96.EMFStep()
end

# Test time series generation, saving output to default directory and loading
@testset "Generate Time Series" begin
    @test TestGenerateTimeSeries.testGenL96()
    @test TestGenerateTimeSeries.testLoadL96()
    @test TestGenerateTimeSeries.testGenIEEE39bus()
    @test TestGenerateTimeSeries.testLoadIEEE39bus()
end

# Test the model equations for known behavior
@testset "IEEE 39 Bus" begin
    @test TestIEEE39bus.test_synchrony()
end

# Test 3D-VAR
@testset "VAR-AutoDiff" begin
    @test TestVarAD.test3DCost()
    @test TestVarAD.test3DGrad()
    @test TestVarAD.test3DNewton()
    @test TestVarAD.test3DNewtonNoise()
    @test TestVarAD.test3DCycle()
    @test TestVarAD.test4DLag1Cost()
    @test TestVarAD.test4DLag1Grad()
    @test TestVarAD.test4DLag1Newton()
    @test TestVarAD.test4DLag1NewtonNoise()
    @test TestVarAD.test4DLag1Cycle()
    @test TestVarAD.test4DLagLCost()
    @test TestVarAD.test4DLagLGrad()
    @test TestVarAD.test4DLagLNewton()
    @test TestVarAD.test4DLagLCycle()
end

# Test filter state and parameter experiments
@testset "Filter Experiments" begin
    @test TestFilterExps.run_ensemble_filter_state_L96()
    @test TestFilterExps.analyze_ensemble_filter_state_L96()
    @test TestFilterExps.run_D3_var_filter_state_L96()
    @test TestFilterExps.analyze_D3_var_filter_state_L96()
    @test TestFilterExps.run_ensemble_filter_param_L96()
    @test TestFilterExps.analyze_ensemble_filter_param_L96()
    @test TestFilterExps.run_ensemble_filter_state_IEEE39bus()
    @test TestFilterExps.analyze_ensemble_filter_state_IEEE39bus()
end

# Test classic smoother state and parameter experiments
@testset "Classic Smoother Experiments" begin
    @test TestClassicSmootherExps.run_ensemble_smoother_state_L96()
    @test TestClassicSmootherExps.analyze_ensemble_smoother_state_L96()
    @test TestClassicSmootherExps.run_ensemble_smoother_param_L96()
    @test TestClassicSmootherExps.analyze_ensemble_smoother_param_L96()
end

# Test IEnKS smoother state and parameter experiments
@testset "Iterative Smoother Experiments" begin
    @test TestIterativeSmootherExps.run_sda_ensemble_smoother_state_L96()
    @test TestIterativeSmootherExps.analyze_sda_ensemble_smoother_state_L96()
    @test TestIterativeSmootherExps.run_sda_ensemble_smoother_param_L96()
    @test TestIterativeSmootherExps.analyze_sda_ensemble_smoother_param_L96()
    @test TestIterativeSmootherExps.run_sda_ensemble_smoother_state_L96()
    @test TestIterativeSmootherExps.analyze_sda_ensemble_smoother_state_L96()
    @test TestIterativeSmootherExps.run_sda_ensemble_smoother_param_L96()
    @test TestIterativeSmootherExps.analyze_sda_ensemble_smoother_param_L96()
end

# Test SIEnKS smoother state and parameter experiments
@testset "Single Iteration Smoother Experiments" begin
    @test TestSingleIterationSmootherExps.run_sda_ensemble_smoother_state_L96()
    @test TestSingleIterationSmootherExps.analyze_sda_ensemble_smoother_state_L96()
    @test TestSingleIterationSmootherExps.run_sda_ensemble_smoother_param_L96()
    @test TestSingleIterationSmootherExps.analyze_sda_ensemble_smoother_param_L96()
    @test TestSingleIterationSmootherExps.run_mda_ensemble_smoother_state_L96()
    @test TestSingleIterationSmootherExps.analyze_mda_ensemble_smoother_state_L96()
    @test TestSingleIterationSmootherExps.run_mda_ensemble_smoother_param_L96()
    @test TestSingleIterationSmootherExps.analyze_mda_ensemble_smoother_param_L96()
end

# Test parallel experiment constructors
@testset "Parallel experiment constructors" begin
    @test TestParallelExperimentDriver.test_ensemble_filter_adaptive_inflation()
    @test TestParallelExperimentDriver.test_D3_var_tuned_inflation()
    @test TestParallelExperimentDriver.test_ensemble_filter_param()
    @test TestParallelExperimentDriver.test_classic_ensemble_state()
    @test TestParallelExperimentDriver.test_classic_ensemble_param()
    @test TestParallelExperimentDriver.test_single_iteration_ensemble_state()
    @test TestParallelExperimentDriver.test_iterative_ensemble_state()
end


##############################################################################################
# end module

end
