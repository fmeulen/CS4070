# Classification example from Rogers and Girolami, chapter 4.
# Now, instead of logistic regression, Gaussian process logistic regression.
# Hyperpars are determined using empirical Bayes
using RCall
using Plots
using Random
using Distributions
using DataFrames
using DelimitedFiles
using LinearAlgebra
using GaussianProcesses

cd(@__DIR__)
plotly() # plotting backend

# read data
y = vec(readdlm("t.csv"))
X = readdlm("X.csv",',')
p = size(X)[2]
scatter(X[:,1], X[:,2])

# to experiment, we may optionally add some extra data to complicated the classification task
addextrapoints = true#false#true
if addextrapoints # add some red points to the topright corner
      Xextra = [4.0 5.0; 4.1 5.1; 3.9 3.9; 5.0 5.0; 5.1 -0.1]
      yextra = zeros(5)
      X = vcat(X, Xextra)
      y = vcat(y, yextra)
end

# choose a kernel for the gp
kernchoice = [:matern, :se][2]

# fit gaussian process
mZero = MeanZero()
if kernchoice == :matern
      kern = Matern(3/2,zeros(2),0.0)     # Matern 3/2 ARD kernel
else
      kern = SE(0.0, 0.0)
end

# in the package, the response is required to be of type Array{Bool,1}
z = convert(Vector{Bool},y)

gp = GP(X',z,mZero,kern,BernLik())      # Fit the Gaussian process model

# get params as follows
GaussianProcesses.get_params(gp)

if kernchoice ==:matern
      set_priors!(gp.kernel,[Distributions.Normal(0.0,5.0) for i in 1:3])
else
      set_priors!(gp.kernel,[Distributions.Normal(0.0,5.0) for i in 1:2])
end

# For full Bayes, do (it appears unstable)
# samples = mcmc(gp,nIter=1000) # each column is an iteration

# empirical Bayes for hyperpars
optimize!(gp;kern=true)

ypred = predict_y(gp, X')

# Compare observed and predicted y
hcat(y, ypred[1])

# predict y on a grid of x vals
mx = my = 20
xg = range(-5,5;length=mx)
yg = range(-5,8;length=my)
Xnew_ =  [[xg[i], yg[j]] for i in eachindex(xg) for j in eachindex(yg)]
Xnew = hcat(Xnew_...)
ynew = predict_y(gp, Xnew)[1]

dfpred = DataFrame(x=Xnew[1,:], y = Xnew[2,:], pred=ynew)
dfdata = DataFrame(x1=X[:,1], x2=X[:,2],y=y)
@rput dfpred
@rput dfdata
#@rput namelabel
R"""
dfdata$y = as.factor(dfdata$y)
library(metR)
library(ggplot2)
p <-     ggplot() + geom_point(data=dfdata, aes(x=x1,y=x2,colour=y)) +
     geom_contour(data= dfpred,aes(x=x,y=y,z=pred),colour='orange',binwidth=0.25) +
      geom_text_contour(data= dfpred,aes(x=x,y=y,z=pred),binwidth=0.25)+theme_light()
p
#figtitle <- paste0("logisticexample_contour_",namelabel,".pdf")
#pdf(figtitle,width=7,height=4.5)
show(p)
#dev.off()
"""
