# Bayesian NB regression using JAGS by
#Rafael S. de Souza, Bart Buelens, Ewan Cameron, Joseph Hilbe

#########################################
##Required libraries#####################
library(rjags)
library(ggmcmc)
library(ggplot2)
library(ggthemes)
library(pander)
library(Cairo)
library(plyr)
library(MASS)
library(scales)
library(grid)
library(runjags)
library(parallel)

##########################################


###########################################
##Auxiliar functions for ploting
# Function to allow parse labels in facet_wrap

facet_wrap_labeller <- function(gg.plot,labels=NULL) {
  #works with R 3.0.1 and ggplot2 0.9.3.1
  require(gridExtra)

  g <- ggplotGrob(gg.plot)
  gg <- g$grobs
  strips <- grep("strip_t", names(gg))

  for(ii in seq_along(labels))  {
    modgrob <- getGrob(gg[[strips[ii]]], "strip.text",
                       grep=TRUE, global=TRUE)
    gg[[strips[ii]]]$children[[modgrob$name]] <- editGrob(modgrob,label=labels[ii])
  }

  g$grobs <- gg
  class(g) = c("arrange", "ggplot",class(g))
  g
}
give.n <- function(x){

  return(c(y = 0.5, label = length(x)))
  #
}

# Arcsinh transformation
asinh_trans <- function(){
  trans_new(name = 'asinh', transform = function(x) asinh(x),
            inverse = function(x) sinh(x))
}

##################################################################################


# Define working directory if necessary
#dir<-getwd()
#setwd(dir)


###Script starts here#################

##Read data#################

GCS = read.csv(file="https://raw.githubusercontent.com/COINtoolbox/NB_GCs/refs/heads/master/Dataset/GCs.csv",header=TRUE,dec=".",sep="")
GCS = subset(GCS, !is.na(Mdyn)) # 1 removed
N_err<-GCS$N_GC_err
lowMBH<-GCS$lowMBH
upMBH<-GCS$upMBH
N = nrow(GCS)

######## NB GLM with errors in both axis ########################################################

### Vector of new values for prediction
MBHx = seq(from = 0.95 * min(GCS$MBH),
           to = 1.05 * max(GCS$MBH),
           length.out = 500)


## Define dataset for JAGS#######################################

jags.data <- list(
  N_GC = GCS$N_GC,
  MBH = GCS$MBH,
  errN_GC = GCS$N_GC_err,
  N = nrow(GCS),
  errMBH = upMBH,
  MBHx = MBHx,
  M = 500
)


#### JAGS model#######################################
model.NB <- "model{

### Priors

# Regression coefficients

beta.0~dnorm(0,0.000001)
beta.1~dnorm(0,0.000001)

#Size

size~dunif(0.001,10)

#Hyperpriors

meanx ~ dgamma(0.01,0.01)
varx ~ dgamma(0.01,0.01)

# 1.Likelihood function

for (i in 1:N){

N_GC[i]~dnegbin(p[i],size)

p[i]<-size/(size+mu[i])

eta[i]<-beta.0+beta.1*MBHtrue[i]

log(mu[i])<-log(exp(eta[i])+errorN[i]-errN_GC[i])

errorN[i]~dbin(0.5,2*errN_GC[i])# Errors in N_GC

MBHtrue[i] ~ dgamma(meanx^2/varx,meanx/varx)# True black hole mass (predictor)

MBH[i]~dnorm(MBHtrue[i],1/errMBH[i]^2)# Observed black hole mass


# 2.Discrepancy measures

YNew[i] ~ dnegbin(p[i],size)
expY[i] <- mu[i]
varY[i] <- mu[i] + pow(mu[i],2) / size
PRes[i] <-(N_GC[i] - expY[i])/sqrt(varY[i])
PResNew[i] <-(YNew[i] - expY[i])/sqrt(varY[i])
D[i]<-pow(PRes[i],2)
DNew[i]<-pow(PResNew[i],2)

# 3.Prediction of current data

  etaTrue[i]<-beta.0+beta.1*MBHtrue[i]
  log(muTrue[i])<-max(-20,min(20,etaTrue[i]))
  pTrue[i]<-size/(size+muTrue[i])
  prediction.NB[i]~dnegbin(pTrue[i],size)
}
Fit<-sum(D[1:N])
New<-sum(DNew[1:N])

# 4.Prediction new data

for (j in 1:M){
  etax[j]<-beta.0+beta.1*MBHx[j]
  log(mux[j])<-max(-20,min(20,etax[j]))
  px[j]<-size/(size+mux[j])
  prediction.NBx[j]~dnegbin(px[j],size)
}
}"
############

# Define function to generate initial values
inits<-function(){list(beta.0=rnorm(1,0,0.1),beta.1=rnorm(1,0,0.1),size=runif(1,0.1,5))}

# One for each chain
inits1 <- inits()
inits2 <- inits()
inits3 <- inits()
params <- c("beta.0","beta.1","size","PRes","prediction.NB","MBHtrue","Fit","New","prediction.NBx")




# Run jags in parallel
jags.neg <-  runjags::run.JAGS(data = jags.data,
                     inits = list(inits1,inits2,inits3),
                     model=model.NB,
                     n.chains = 3,
                     adapt=2000,
                     monitor=c(params),
                     burnin=20000,
                     sample=50000,
                     summarise=FALSE,
                     plots=FALSE
)

jagssamples.nb <- as.mcmc.list(jags.neg )
summary<-extend.jags(jags.neg,drop.monitor=c("PRes","prediction.NB","MBHtrue","Fit","New","prediction.NBx"), summarise=TRUE)

# Check the output
print(summary)

### Analysis ends here####



## Below are plot diagnostics
MBHtrue<-summary(as.mcmc.list(jags.neg, vars="MBHtrue"),quantiles=0.5)
pred.NBerr<-summary(as.mcmc.list(jags.neg, vars="prediction.NB"),quantiles=c(0.005,0.025,0.25,0.5,0.75,0.975, 0.995))
pred.NB2err<-data.frame(Type=GCS$Type,NGC=GCS$N_GC,MBHtrue=MBHtrue$quantiles,MBH=GCS$MBH,mean=pred.NBerr$statistics[1:45,1],lwr1=pred.NBerr$quantiles[1:45,3],lwr2=pred.NBerr$quantiles[1:45,2],lwr3=pred.NBerr$quantiles[1:45,1],upr1=pred.NBerr$quantiles[1:45,5],upr2=pred.NBerr$quantiles[1:45,6],upr3=pred.NBerr$quantiles[1:45,7])
pred.NBerrx<-summary(as.mcmc.list(jags.neg,vars="prediction.NBx"),quantiles=c(0.005,0.025,0.25,0.5,0.75,0.975, 0.995))
pred.NB2errx<-data.frame(MBHx=MBHx,mean=pred.NBerrx$statistics[,1],lwr1=pred.NBerrx$quantiles[,3],lwr2=pred.NBerrx$quantiles[,2],lwr3=pred.NBerrx$quantiles[,1],upr1=pred.NBerrx$quantiles[,5],upr2=pred.NBerrx$quantiles[,6],upr3=pred.NBerrx$quantiles[,7])



#N_low<-asinh(pred.NB2err$NGC-N_err)
N_low<-pred.NB2err$NGC-N_err
N_low[N_low<0]<-0

CairoPDF("..//Figures/MBHx.pdf",height=8,width=9)
ggplot(pred.NB2err,aes(x=MBH,y=NGC))+
  geom_ribbon(data=pred.NB2errx,aes(x=MBHx,y=mean,ymin=lwr1, ymax=upr1), alpha=0.45, fill="gray") +
  geom_ribbon(data=pred.NB2errx,aes(x=MBHx,y=mean,ymin=lwr2, ymax=upr2), alpha=0.35, fill="gray") +
  geom_ribbon(data=pred.NB2errx,aes(x=MBHx,y=mean,ymin=lwr3, ymax=upr3), alpha=0.25, fill="gray") +
  geom_point(aes(colour=Type,shape=Type),size=3.25,alpha=0.8)+
  geom_errorbar(aes(colour=Type,ymin=N_low,ymax=NGC+N_err),alpha=0.7,width=0.05)+
  geom_errorbarh(aes(colour=Type,xmin=MBH-GCS$lowMBH,
                                  xmax=MBH+upMBH),alpha=0.7,height=0.05)+
  geom_line(data=pred.NB2errx,aes(x=MBHx,y=mean),colour="gray25",linetype="dashed",size=1.2)+
  annotate("text", x = 6.63, y = 800, label = "Milky Way",size = 6.5)+
 # annotate("arrow",aes(x =  6.65, y = 600, xend = 6.61, yend = 200), arrow = arrow(length = unit(0.25, "cm")))+
  scale_y_continuous(trans = 'asinh',breaks=c(0,10,100,1000,10000,100000),labels=c("0",expression(10^1),expression(10^2),
                                                                                   expression(10^3),expression(10^4),expression(10^5)))+
  scale_colour_gdocs()+
  scale_shape_manual(values=c(19,2,8))+
  theme_hc()+
  ylab(expression(N[GC]))+
  xlab(expression(log~M[BH]/M['\u0298']))+theme(legend.position="top",plot.title = element_text(hjust=0.5),
                                                axis.title.y=element_text(vjust=0.75),
                                                axis.title.x=element_text(vjust=-0.25),
                                                text = element_text(size=25))
dev.off()



# Diagnostics plots

#Density
#HPDinterval(jagssamples.nb)

S.NB1<-ggs(jagssamples.nb ,family=c("beta"))
S.NB2<-ggs(jagssamples.nb,family=c("size"))

S.NB<-rbind(S.NB1,S.NB2,deparse.level=2)
S.NB$Parameter<- factor(S.NB$Parameter, levels = c("beta.0","beta.1", "size"))

S.NB$Parameter<-revalue(S.NB$Parameter, c("beta.0"=expression(beta[0]), "beta.1"=expression(beta[1]),
                                          "size"="k"))

g1<-ggs_density(S.NB)+
  scale_colour_economist(guide="none")+
  theme_hc()+
  scale_fill_economist()+
  #  theme_economist_white(gray_bg = F, base_size = 11, base_family = "sans")+
  theme(strip.background = element_rect(fill="gray95"),plot.background = element_rect(fill = 'white', colour = 'white'),
        legend.position="none",plot.title = element_text(hjust=0.5),
        axis.title.y=element_text(vjust=0.75),axis.text.x=element_text(size=25),
        strip.text.x=element_text(size=25),
        axis.title.x=element_text(vjust=-0.25),
        text = element_text(size=25))+xlab("Parameter  value")+ylab("Density")

CairoPDF("..//Figures/posterior_MBH.pdf",height=10,width=8)
facet_wrap_labeller(g1,labels=c(expression(beta[0]),expression(beta[1]),"k"))
dev.off()


# Mixing of chains

g0<-ggs_traceplot(S.NB)+
  scale_colour_economist(guide="none")+
  theme_hc()+
  scale_fill_economist()+
  #  theme_economist_white(gray_bg = F, base_size = 11, base_family = "sans")+
  theme(panel.margin = unit(2, "lines"),strip.background = element_rect(fill="gray95"),plot.background = element_rect(fill = 'white', colour = 'white'),
        legend.position="none",plot.title = element_text(hjust=0.5),
        axis.title.y=element_text(vjust=0.75),axis.text.x=element_text(size=25),
        strip.text.x=element_text(size=25),
        axis.title.x=element_text(vjust=-0.25),
        text = element_text(size=25))+
  ylab("Parameter value")+
  xlab("Iteration")+
  facet_grid(Parameter~.,labeller=label_parsed,scales = "free")+coord_cartesian(xlim=c(20000,50000))

CairoPDF("..//Figures/chain_NB.pdf",height=10,width=8)
g0
dev.off()





# Caterpillar

ggs_caterpillar(S.NB,family=list("beta","k"))+ scale_colour_economist(guide="none")+
  theme_hc()+
  scale_fill_economist()+
  #  theme_economist_white(gray_bg = F, base_size = 11, base_family = "sans")+
  theme(strip.background = element_rect(fill="gray95"),plot.background = element_rect(fill = 'white', colour = 'white'),
        legend.position="none",plot.title = element_text(hjust=0.5),
        axis.title.y=element_text(vjust=0.75),axis.text.x=element_text(size=25),
        strip.text.x=element_text(size=25),
        axis.title.x=element_text(vjust=-0.25),
        text = element_text(size=25))

# Model comparison

# Predicted vs Observed
Pred<-ggs(jagssamples.nb,family=c("New"))[,"value"]
Obs<-ggs(jagssamples.nb,family=c("Fit"))[,"value"]
sqrt(mean((Pred-Obs)^2))


# DIC
jags.DIC <- jags.model(
  data = jags.data,
  inits = inits1,
  textConnection(model.NB),
  n.chains = 3,
  n.adapt=2000
)

update(jags.DIC , 10000)
dicsamples.nb <- dic.samples(jags.DIC, params, n.iter = 25000,type="pD")

# Dispersion parameter
Pres<-summary(as.mcmc.list(jags.neg, vars="PRes"),quantiles=0.5)$quantiles
Dispersion = sum(Pres^2)/(N-3)# beta.0, beta.1 and k, 3 parameters

# Gelman-Rubin
gelman.diag(jagssamples.nb)


# Plot residuals vc galaxy type
clus_data<-data.frame(Pres=Pres,MBH=GCS$MBH,type=GCS$Type)
p <- ggplot(clus_data, aes(x=type, y=Pres),group=type)+ xlab("Galaxy Type") +
  ylab("Pearson Residuals")

pdf("Pres_MBH.pdf",height=5.5,width=9)
p + stat_boxplot(colour="gray",geom ='errorbar')+geom_boxplot(aes(group=type,colour=type,fill=type),outlier.shape = 19,colour="gray",fatten=2,size=1,outlier.size=2,outlier.colour = "gray",notchwidth = 0.35,notch=F,data=clus_data)+
  theme_hc()+
  scale_fill_economist()+
  theme(strip.background = element_rect(fill="gray95"),plot.background = element_rect(fill = 'white', colour = 'white'),
        legend.position="none",plot.title = element_text(hjust=0.5),
        axis.title.y=element_text(vjust=0.75),axis.text.x=element_text(size=25),
        strip.text.x=element_text(size=25),
        axis.title.x=element_text(vjust=-0.25),
        text = element_text(size=25))
dev.off()
