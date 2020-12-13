using RCall
using Plots
using Random
using Distributions
using DataFrames
using DelimitedFiles
using LinearAlgebra
using StatsFuns

workdir = @__DIR__
println(workdir)
cd(workdir)


#Random.seed!(2) # set RNG
ITER = 10_000 # nr of iterations
BURNIN = div(ITER,2)

# implement MH and MALA
abstract type MCMCmove end
struct RW <: MCMCmove end
struct MALA <: MCMCmove end

movetype = [RW(), MALA()][2]  # either Random Walk or MALA

# read data
y = readdlm("t.csv")
y = vec(Int.(y))
X = readdlm("X.csv",',')
p = size(X)[2]
scatter(X[:,1], X[:,2])

# set prior
σprior = sqrt(10.0) # std of prior
πprior = MvNormal(p,σprior)

# set proposal kernel
if movetype==MALA()
    σproposal = 2.5
    namelabel = "MALA"  # only for naming of figures
else
    σproposal = 1.0
    namelabel = "MH"
end
πproposal = MvNormal(p,σproposal)

# define logtarget and its gradient
ρ(x, θ) = logistic(dot(x,θ))
ρ(X::Matrix, θ) = [ρ(x, θ) for x in eachrow(X)]
ℓ(θ,x,y::Int) = logpdf(Bernoulli(ρ(x,θ)),y)     # l
ℓ(θ,X,y::Vector) = sum([ℓ(θ,X[i,:],y[i]) for i ∈ eachindex(y)])
logtarget(θ,X,y,πprior) = ℓ(θ,X,y) + logpdf(πprior, θ)
∇logtarget(θ,X,y,σprior) = X'*(y-ρ(X, θ)) - θ/σprior^2


function MHupdate!(::RW, θiters, X, y, (πproposal, πprior), (σproposal, σprior))
    θ = θiters[end]
    θᵒ = θ .+ rand(πproposal)
    logA = logtarget(θᵒ,X,y,πprior) - logtarget(θ,X,y,πprior) +
            logpdf(πproposal,θ-θᵒ) - logpdf(πproposal, θᵒ-θ)
    if log(rand()) < logA
        push!(θiters, θᵒ)
        acc = 1
    else
        push!(θiters, θ)
        acc = 0
    end
    acc
end

function MHupdate!(::MALA, θiters, X, y, (πproposal, πprior), (σproposal, σprior))
    θ = θiters[end]
    drift = 0.5 * σproposal * ∇logtarget(θ,X,y,σprior)
    θᵒ = θ .+ drift.+  rand(πproposal)
    driftᵒ = 0.5 * σproposal * ∇logtarget(θᵒ,X,y,σprior)
    logA = logtarget(θᵒ,X,y,πprior) - logtarget(θ,X,y,πprior) +
            logpdf(πproposal,θ-θᵒ-driftᵒ) - logpdf(πproposal, θᵒ-θ-drift)
    if log(rand()) < logA
        push!(θiters, θᵒ)
        acc = 1
    else
        push!(θiters, θ)
        acc = 0
    end
    acc
end

# Run MCMC
sumacc = 0
θiters = [fill(25.0,p)]


for i in 1:ITER
    global sumacc
    acc = MHupdate!(movetype, θiters, X, y, (πproposal, πprior), (σproposal, σprior))
    sumacc += acc
end

println("acceptance percentage equals: ", round(100*sumacc/ITER;digits=2))
println("Posterior mean for θ: ", mean(θiters[BURNIN:ITER]))

# Some plotting using ggplot (depends on R, and the libraries I call in there)
df = DataFrame(iterates=0:ITER,theta1 = first.(θiters), theta2 = last.(θiters))
@rput df
@rput namelabel
R"""
library(tidyverse)
library(ggplot2)
library(gridExtra)
theme_set(theme_light())
p1 <- df %>% ggplot(aes(x=theta1,y=theta2,colour=iterates)) + geom_point(alpha=0.8,size=0.7)
p2 <- df %>%  gather(key="parameter",value="value", theta1,theta2)%>%
 ggplot() + geom_path(aes(x=iterates, y=value)) + facet_wrap(~parameter)
grid.arrange(p1,p2)
figtitle <- paste0("logisticexample_iterates_",namelabel,".pdf")
pdf(figtitle,width=7,height=4.5)
grid.arrange(p1,p2)
dev.off()
"""

# contour plot for predictive density
mx = my = 20
xg = range(-5,5;length=mx)
yg = range(-5,8;length=my)
predprob = Vector{Float64}[]
for it in BURNIN:ITER
    push!(predprob,    [ρ([xg[i], yg[j]], θiters[it]) for i in eachindex(xg) for j in eachindex(yg)] )
end
dfpred = DataFrame(x=repeat(xg,inner=mx), y=repeat(yg, outer=my), pred=mean(predprob))
dfdata = DataFrame(x1=X[:,1], x2=X[:,2],y=y)
@rput dfpred
@rput dfdata
@rput namelabel
R"""
dfdata$y = as.factor(dfdata$y)
library(metR)
p <-     ggplot() + geom_point(data=dfdata, aes(x=x1,y=x2,colour=y)) +
     geom_contour(data= dfpred,aes(x=x,y=y,z=pred),colour='orange') +
      geom_text_contour(data= dfpred,aes(x=x,y=y,z=pred))
p
figtitle <- paste0("logisticexample_contour_",namelabel,".pdf")
pdf(figtitle,width=7,height=4.5)
show(p)
dev.off()
"""
