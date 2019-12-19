# A basic example for Gaussian process regression with GaussianProcesses.jl
# See also http://stor-i.github.io/GaussianProcesses.jl/latest/

workdir = @__DIR__
println(workdir)
cd(workdir)

using LinearAlgebra
using PDMats
using RCall
using Distributions
using DataFrames
using Plots
using GaussianProcesses
const gP = GaussianProcesses
using Random

Random.seed!(3)

# set 'true' regression function
f⁰(x) = 2.0 + 0.2*sin(30*x) + (x>0.7)
Xnew = 0.0:.005:1.0
plot(Xnew,f⁰,label="true")

# generate data
σ = 0.2                                         # noise stdev
n = 70                                         # sample size
X = sort(rand(n))
y = f⁰.(X) + σ*randn(n)
scatter!(X,y,label="data",markersize=2.0)

# Set GP with fixed kernel parameters and observation noise
mZero = MeanZero()                      # Zero mean function
kern = SE(0.0,0.0)                      # Sqaured exponential kernel, first argument is length scale, second argument standard dev
logObsNoise = 0.0                       # log standard deviation of observation noise
gp = GP(X,y,mZero,kern,logObsNoise)     # Fit the GP

# Plotting
plot(gp; xlabel="x", ylabel="y", title="Gaussian process (no hyperpar tuning)", legend=false)
# add some draws from posterior
samples = rand(gp, Xnew, 3)
plot!(Xnew, samples)

# determine noise and kernel parameters using empirical Bayes
kern = SE(0.0,0.0)
gp_eb = GP(X,y,mZero,kern,logObsNoise)     # Fit the GP
# specified kernel parameters are
gP.get_params(kern)
print(  optimize!(gp_eb;kern=true,noise=true)   )   #Optimise the parameters, note that the first parameter is for logObsNoise
# optimized kerenl parameters are
gP.get_params(kern)
# so we have optimised paramters
# logObsNoise = -1.29
# logLengthScale = -1.23
# logSignalStandardDeviation = -0.60
plot(gp_eb; xlabel="x", ylabel="y", title="Gaussian process (empirical Bayes)",legend=false)


# MCMC
kern = SE(0.0,0.0)
gp_b = GP(X,y,mZero,kern,logObsNoise)
set_priors!(kern, [Normal(0.0,5.0), Normal(0.0,5.0)]) #
set_priors!(gp_b.logNoise, [Distributions.Normal(0.0, 5.0)])
ess_chain = ess(gp_b)
Plots.plot(ess_chain', label=["logObsNoise", "logLengthScale", "logSignalSD"])
plot(gp_b; xlabel="x", ylabel="y", title="Gaussian process (full mcmc)", legend=false)

# note the difference between empirical Bayes and full Bayes, this is mainly on the estimate of
