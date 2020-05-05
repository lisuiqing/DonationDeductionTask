setwd("C:/Users/lisuiqing/Documents")
library(rstan)
library(loo)
library(reshape2)
data1 <- read.csv('donate_exampleData.csv')
colnames(data1) <- c("subjID", "eff_e" ,"eff_s" , "equ_e" ,"equ_s" ,"decision" )
subjs <- reshape2::dcast(data1,subjID~decision)[,c(1)]
model_tradeoff_bias <- stan_model('donate_tradeoff_bias.stan')
options(max.print=20600)

#get each columns from raw data
decision <- data1$decision
eff_e<- data1$eff_e
eff_s<- data1$eff_s
equ_e<- data1$equ_e
equ_s<- data1$equ_s

decision_mat <- matrix(data<-5,nrow <- 100,ncol <- 18)
eff_e_mat <- matrix(data<-5,nrow <- 100,ncol <- 18)
eff_s_mat <- matrix(data<-5,nrow <- 100,ncol <- 18)
equ_e_mat <- matrix(data<-5,nrow <- 100,ncol <- 18)
equ_s_mat <- matrix(data<-5,nrow <- 100,ncol <- 18)
Tsubj <- vector("numeric",100)
N <- 100#input
T <- 18#input

for (h in 1:100) {
  Tsubj[h]<-18  #input
}


k <- 1
for (i in 1:100) {
  for (j in 1:18) {
    decision_mat[i,j]<-decision[k] #input 5 matrixs
    eff_e_mat[i,j]<-eff_e[k]
    eff_s_mat[i,j]<-eff_s[k]
    equ_e_mat[i,j]<-equ_e[k]
    equ_s_mat[i,j]<-equ_s[k]
    k <- k+1
  }
}

dat <- list(N = 100,
            T = 18,
            Tsubj = Tsubj,
            decision = decision_mat,
            eff_e = eff_e_mat,
            eff_s = eff_s_mat,
            equ_e = equ_e_mat,
            equ_s = equ_s_mat)


fit_tradeoff_bias <- rstan::sampling(object  = model_tradeoff_bias,
                            data    = dat,
                            #pars    = c("alpha", "lamda"), #pars that are interested ,
                            chains  = 4,
                            iter    = 2000,
                            warmup  = 1000,
                            thin    = 1,#A positive integer specifying the period for saving samples
                            control = list(adapt_delta   = 0.95,
                                           stepsize      = 1,
                                           max_treedepth = 10),
                            algorithm = "HMC",
                            core=4)

parVals <- rstan::extract(fit_tradeoff_bias, permuted = TRUE)
indPars  = "mean"
measure_indPars <- switch(indPars, mean = mean, median = median, mode = estimate_mode)
which_indPars <- c("alpha", "lamda","bias")
# Measure all individual parameters (per subject)
allIndPars <- as.data.frame(array(NA, c(N, length(which_indPars))))

for (i in 1:N) {
  allIndPars[i, ] <- mapply(function(x) measure_indPars(parVals[[x]][, i]), which_indPars)
}
allIndPars <- cbind(subjs, allIndPars)
colnames(allIndPars) <- c("subjID", "alpha", "lamda","bias2")

ypred <- matrix(nrow=100,ncol=18)
for (i in 1:N){
  for(j in 1:T){
    ypred[i,j] <- mapply(function(x) measure_indPars(parVals[[x]][, i,j]), "y_pred")
  }
}



modelData_tradeoff_bias                   <- list()
modelData_tradeoff_bias$model             <- "model_tradeoff_bias" #the name of model
modelData_tradeoff_bias$allIndPars        <- allIndPars #Summary of individual subjectsï¿???? parameters
modelData_tradeoff_bias$parVals           <- parVals #Posterior MCMC samples for all parameters
modelData_tradeoff_bias$fit               <- fit_tradeoff_bias #an rstan object is the output of RStanâ€™s stan() function
modelData_tradeoff_bias$rawdata           <- data1
modelData_tradeoff_bias$ypred                <- ypred

#a function for calculate LOOIC,WAIC
estimate<- function(obj_fit){
  IC <- list()
  lik  <- loo::extract_log_lik(stanfit = obj_fit, parameter_name = "log_lik")
  rel_eff <- loo::relative_eff(exp(lik),chain_id = rep(1:4, each = nrow(lik) / 4))
  IC$LOOIC<- loo::loo(lik, r_eff = rel_eff)
  IC$WAIC <- loo::waic(lik)
  looic=IC$LOOIC$estimates[3,1]
  waic=IC$WAIC$estimates[3,1]
  modelTable = data.frame(Model = NULL, LOOIC = NULL, WAIC = NULL)
  modelTable[1, "LOOIC"] = looic
  modelTable[1, "WAIC"]  = waic
  print(modelTable)
  return(IC)
}

estimate(fit_tradeoff_bias)
