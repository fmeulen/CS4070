# Classification example from Rogers and Girolami, chapter 4.
# Now, instead of logistic regression, Gaussian process logistic regression.

using RCall
using Plots
using Random
using Distributions
using DataFrames
using DelimitedFiles
using LinearAlgebra
using GaussianProcesses

workdir = @__DIR__
println(workdir)
cd(workdir)
pyplot()

# read data
y = vec(readdlm("t.csv"))
X = readdlm("X.csv",',')
p = size(X)[2]
scatter(X[:,1], X[:,2])

addextrapoints = true
if addextrapoints # add some red points to the topright corner
      Xextra = [4.0 5.0; 4.1 5.1; 3.9 3.9; 5.0 5.0; 5.1 -0.1]
      yextra = zeros(5)
      X = vcat(X, Xextra)
      y = vcat(y, yextra)
end

kernchoice = [:matern, :se][1]

# fit gaussian process
mZero = MeanZero()
if kernchoice == :matern
      kern = Matern(3/2,zeros(2),0.0)     # Matern 3/2 ARD kernel
else
      kern = SE(0.0, log(10.01))
end


# response needs to be of type Array{Bool,1}
z = Array{Bool}(undef,length(y))
z .= y

gp = GP(X',z,mZero,kern,BernLik())      # Fit the Gaussian process model

# get params as follows
GaussianProcesses.get_params(gp)

if kernchoice ==:matern
      set_priors!(gp.kernel,[Distributions.Normal(0.0,2.0) for i in 1:3])
else
      set_priors!(gp.kernel,[Distributions.Normal(0.0,2.0) for i in 1:2])
end

# get posteriors of hyperpars
samples = mcmc(gp,nIter=1000) # each column is an iteration

ypred = predict_y(gp, X')

hcat(y, ypred[1])

# predict on grid x in (-5,5), y in (-5,5)

mx = my = 20
xg = range(-5,5;length=mx)
yg = range(-5,8;length=my)
Xpred_ =  [[xg[i], yg[j]] for i in eachindex(xg) for j in eachindex(yg)]
Xpred = hcat(Xpred_...)
ypred = predict_y(gp, Xpred)

dfpred = DataFrame(x=Xpred[1,:], y = Xpred[2,:], pred=ypred[1])
dfdata = DataFrame(x1=X[:,1], x2=X[:,2],y=y)
@rput dfpred
@rput dfdata
#@rput namelabel
R"""
dfdata$y = as.factor(dfdata$y)
library(metR)
library(ggplot2)
p <-     ggplot() + geom_point(data=dfdata, aes(x=x1,y=x2,colour=y)) +
     geom_contour(data= dfpred,aes(x=x,y=y,z=pred),colour='orange') +
      geom_text_contour(data= dfpred,aes(x=x,y=y,z=pred))+theme_light()
p
#figtitle <- paste0("logisticexample_contour_",namelabel,".pdf")
#pdf(figtitle,width=7,height=4.5)
show(p)
#dev.off()
"""
