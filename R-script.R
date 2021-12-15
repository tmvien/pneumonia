setwd("~/Desktop/STA9891/Final")
rm(list = ls())    #delete objects
library(glmnet)
library(randomForest)
library(ggplot2)
library(grid)
library(gridExtra)
library(viridis)
library(dplyr)
library(tidyverse)

set.seed(3)

data <- read_csv("https://raw.githubusercontent.com/tmvien/pneumonia/master/train.csv")
data <- data[-1,]
rownames(data) <- 1:nrow(data)

# confirm pixel range is 0-255
range(data[,-ncol(data)])

names(data)[ncol(data)] <- c("class")
# recode target as 0 and 1
data["class"] <- ifelse(data["class"] == "NORMAL", 0, 1)

sum(data["class"] == 1)/nrow(data)
sum(data["class"] == 0)/nrow(data)

# normalize the image pixels by diving by 255
data[,-ncol(data)] = data[,-ncol(data)]/255

# visualize some examples
samp.ind    <- sample(dim(data)[1], dim(data)[1])
data        <- data[samp.ind,]
data %>%
  mutate(instance = row_number()) %>%  
  gather(pixel, value, -class, -instance) %>%  
  tidyr::extract(pixel, "pixel", "(\\d+)", convert = TRUE) %>%  
  mutate(x = pixel %% 28 + 1, y = 28 - pixel %/% 28) %>%
  filter(instance <= 12) %>%  
  ggplot(aes(x, y, fill = value)) +  geom_tile() +  
  scale_fill_gradient(low="black", high="white") + 
  facet_wrap(~ instance + class)

# shuffle data
samp.ind    <- sample(dim(data)[1], dim(data)[1])
X           <- as.matrix(data[samp.ind, -ncol(data)])
y           <- data[samp.ind,]$class

# remove data
rm(data)

n                  <- dim(X)[1]
p                  <- dim(X)[2]
train.rate         <- 0.9
iterations         <- 50
K                  <- 10

# train and val auc matrices
train.auc            <-  matrix(0, nrow = iterations, ncol = 4)
colnames(train.auc)  <-  c("LRidge", "LLasso", "LElastic", "RF")

val.auc              <-  matrix(0, nrow = iterations, ncol = 4)
colnames(val.auc)    <-  c("LRidge", "LLasso", "LElastic", "RF")

# ridge, lasso, elasticNet coefficient matrices
rid.coef             <-  matrix(0, nrow = iterations, ncol = p+1)
lasso.coef           <-  matrix(0, nrow = iterations, ncol = p+1)
el.coef              <-  matrix(0, nrow = iterations, ncol = p+1)
# RF
rf.importance        <-  matrix(0, nrow = iterations, ncol = p)

# time for cv and model fit
time.cv              <-  matrix(0, nrow = iterations, ncol = 3)
colnames(time.cv)    <-  c("LRidge", "LLasso", "LElastic")
time.fit             <-  matrix(0, nrow = iterations, ncol = 4)
colnames(time.fit)   <-  c("LRidge", "LLasso", "LElastic", "RF")

# function for auc 
auc <- function(TPR, FPR){
  # sort inputs, best scores first 
  TPR     <-  sort(TPR)
  FPR     <-  sort(FPR)
  dFPR    <-  c(diff(FPR), 0)
  dTPR    <-  c(diff(TPR), 0)
  sum(TPR * dFPR) + sum(dTPR * dFPR)/2
}

# sensitivity function for auc
sens.cal <- function(y, prob) {
  # vector of thresholds
  thresh        <-  seq(0, 1, 0.01)
  
  # empty matrices
  sens  <-  matrix(0, nrow = length(thresh), ncol = 1)
  ind = 1
  for (i in thresh) {
    y.hat         <-    ifelse(prob > i, 1, 0)
    # FP, TP, FPR, TPR calculation
    TP            <-    sum(y.hat[y==1] == 1) # true positives = positives in the data that were predicted as positive
    P             <-    sum(y==1) # total positives in the data
    TPR           <-    TP/P # true positive rate = 1 - type 2 error = sensitivity = recall
    sens[ind,]    <-    TPR 
    ind           <-    ind + 1
  }
  return(sens)
}

# false positive rate functions for auc
fpr.cal <- function(y, prob) {
  # vector of thresholds
  thresh        <-  seq(0, 1, 0.01)
  
  # empty matrices
  fpr   <-  matrix(0, nrow = length(thresh), ncol = 1)
  ind = 1
  for (i in thresh) {
    y.hat         <-    ifelse(prob > i, 1, 0)
    # FP, TP, FPR, TPR calculation
    FP            <-    sum(y[y.hat==1] == 0) # false positives = negatives in the data that were predicted as positive
    N             <-    sum(y==0) # total negatives in the data
    FPR           <-    FP/N # false positive rate = type 1 error = 1 - specificity
    fpr[ind,]     <-    FPR
    ind           <-    ind + 1
  }
  return(fpr)
}

# begin iterations
for(m in 1:iterations) {
  #print(paste0("Iteration = ", m))
  cat(sprintf("iteration = %3.f \n", m))
  flush.console()
  
  train            <-  sample(n, n*train.rate)
  
  # decide weights for imbalanced data
  nP               <-  sum(y[train]) 
  nN               <-  length(train) - nP 
  w                <-  rep(1, length(train))
  w[y[train] == 1] <-  nN/nP
  # weight for rf
  w.rf             <-  c("0" = 1, "1" = nN/nP)
  
  ############## ridge 
  # begin cross-validation
  start.time   <- proc.time()
  cv.rid       <- cv.glmnet(X[train,], y[train],
                            alpha = 0,
                            family = "binomial",
                            weights = w,
                            nfolds = K,
                            type.measure="auc")
  end.time      <-  proc.time() - start.time
  time.cv[m,1]  <-  end.time["elapsed"]
  # begin model fit
  start.time    <-  proc.time()
  rid.fit       <-  glmnet(X[train,], y[train],
                           alpha = 0,
                           family = "binomial",
                           lambda = cv.rid$lambda.min,
                           weights = w,
                           standardize = F)
  end.time      <-  proc.time() - start.time
  time.fit[m,1] <-  end.time["elapsed"]
  #### calculate auc for train and valid
  # feature coefficients
  rid.coef[m,1]   <-     rid.fit$a0
  rid.coef[m,-1]  <-     as.vector(rid.fit$beta)
  # probability predictions
  prob.train.rid  <-     exp( X[train,]  %*% as.vector(rid.fit$beta) +  rid.fit$a0  )/(1 + exp(X[train,]  %*% as.vector(rid.fit$beta) +  rid.fit$a0  ))
  prob.val.rid    <-     exp( X[-train,] %*% as.vector(rid.fit$beta) +  rid.fit$a0  )/(1 + exp(X[-train,] %*% as.vector(rid.fit$beta) +  rid.fit$a0  ))
  # auc train and valid
  auc.train.rid             <-    auc( sens.cal(y[train],  prob.train.rid), fpr.cal(y[train], prob.train.rid ))
  auc.valid.rid             <-    auc( sens.cal(y[-train], prob.val.rid  ), fpr.cal(y[-train], prob.val.rid  ))
  train.auc[m,1]            <-    auc.train.rid
  val.auc[m,1]              <-    auc.valid.rid
  
  ############ lasso regression
  # begin cross validate
  start.time  <-  proc.time()
  cv.lasso    <-  cv.glmnet(X[train,], y[train],
                            alpha = 1,
                            family = "binomial",
                            weights = w,
                            nfolds = K,
                            type.measure="auc")
  end.time     <-  proc.time() - start.time
  time.cv[m,2] <-  end.time["elapsed"]
  # begin model fit
  start.time   <- proc.time()
  lasso.fit    <- glmnet(X[train,], y[train],
                         alpha = 1,
                         family = "binomial",
                         lambda = cv.lasso$lambda.min,
                         weights = w,
                         standardize = F)
  end.time       <-  proc.time() - start.time
  time.fit[m,2]  <-  end.time["elapsed"]
  #### calculate auc for train and valid
  # feature coefficients
  lasso.coef[m,1]   <-     lasso.fit$a0
  lasso.coef[m,-1]  <-     as.vector(lasso.fit$beta)
  # probability predictions
  prob.train.lasso  <-     exp( X[train,]  %*% as.vector(lasso.fit$beta) +  lasso.fit$a0  )/(1 + exp(X[train,]  %*% as.vector(lasso.fit$beta) +  lasso.fit$a0  ))
  prob.val.lasso    <-     exp( X[-train,] %*% as.vector(lasso.fit$beta) +  lasso.fit$a0  )/(1 + exp(X[-train,] %*% as.vector(lasso.fit$beta) +  lasso.fit$a0  ))
  # auc train and valid
  auc.train.lasso          <-    auc( sens.cal(y[train],  prob.train.lasso), fpr.cal(y[train], prob.train.lasso ))
  auc.valid.lasso          <-    auc( sens.cal(y[-train], prob.val.lasso  ), fpr.cal(y[-train], prob.val.lasso  ))
  train.auc[m,2]           <-    auc.train.lasso
  val.auc[m,2]             <-    auc.valid.lasso
  
  ############# elastic net
  # begin cross-validation
  start.time    <-   proc.time()
  cv.el         <-   cv.glmnet(X[train,], y[train],
                               alpha = 0.5, 
                               family = "binomial",
                               weights = w,
                               nfolds = K,
                               type.measure="auc")
  end.time      <-   proc.time() - start.time
  time.cv[m,3]  <-   end.time["elapsed"]
  # begin model fit
  start.time    <-   proc.time()
  el.fit        <-   glmnet(X[train,], y[train], 
                            alpha = 0.5, 
                            family = "binomial",
                            lambda = cv.el$lambda.min,
                            weights = w,
                            standardize = F)
  end.time      <-   proc.time() - start.time
  time.fit[m,3] <-   end.time["elapsed"]
  #### calculate auc for train and valid
  # feature coefficients
  el.coef[m,1]   <-     el.fit$a0
  el.coef[m,-1]  <-     as.vector(el.fit$beta)
  # probability predictions
  prob.train.el  <-     exp( X[train,]  %*% as.vector(el.fit$beta) +  el.fit$a0  )/(1 + exp(X[train,]  %*% as.vector(el.fit$beta) +  el.fit$a0  ))
  prob.val.el    <-     exp( X[-train,] %*% as.vector(el.fit$beta) +  el.fit$a0  )/(1 + exp(X[-train,] %*% as.vector(el.fit$beta) +  el.fit$a0  ))
  # auc train and valid
  auc.train.el          <-    auc( sens.cal(y[train],  prob.train.el), fpr.cal(y[train], prob.train.el ))
  auc.valid.el          <-    auc( sens.cal(y[-train], prob.val.el  ), fpr.cal(y[-train], prob.val.el  ))
  train.auc[m,3]        <-    auc.train.el
  val.auc[m,3]          <-    auc.valid.el
  
  ############### random forest 
  # begin model fit
  start.time     <-  proc.time()
  rf             <-  randomForest(x=X[train,], y=as.factor(y[train]),
                                  mtry = sqrt(p),
                                  classwt = w.rf,
                                  importance = T
  )
  end.time       <-  proc.time() - start.time
  time.fit[m,4]  <-  end.time["elapsed"]
  #### calculate auc for train and valid
  # feature importance
  rf.importance[m,]  <- rf$importance[,1] 
  # probability predictions
  prob.train.rf      <-  predict(rf, newdata = X[train,],  type = "prob")[,2]
  prob.val.rf        <-  predict(rf, newdata = X[-train,], type = "prob")[,2]
  # auc train and valid
  auc.train.rf          <-    auc( sens.cal(y[train],  prob.train.rf), fpr.cal(y[train], prob.train.rf ))
  auc.valid.rf          <-    auc( sens.cal(y[-train], prob.val.rf  ), fpr.cal(y[-train], prob.val.rf  ))
  train.auc[m,4]        <-    auc.train.rf
  val.auc[m,4]          <-    auc.valid.rf
}

plot(cv.rid, main="")
plot(cv.lasso, main="")
plot(cv.el, main="")

# colnames(rid.coef)      <- c("intercept", paste0("X.", 1:784))
# colnames(lasso.coef)    <- c("intercept", paste0("X.", 1:784))
# colnames(el.coef)       <- c("intercept", paste0("X.", 1:784))
# colnames(rf.importance) <- c(paste0("X.", 1:784))

# write.csv(rid.coef, file="rid_coef.csv", row.names=F)
# write.csv(lasso.coef, file="lasso_coef.csv", row.names=F)
# write.csv(el.coef, file="el_coef.csv", row.names=F)
# write.csv(rf.importance, file="rf_importance.csv", row.names=F)

train.auc <- data.frame(train.auc)
# write.csv(train.auc, file="train_auc.csv", row.names = F)
val.auc <- data.frame(val.auc)
# write.csv(val.auc, file="val_auc.csv", row.names = F)

# write.csv(time.cv, file="timecv.csv", row.names=F)
# write.csv(time.fit, file="timefit.csv", row.names=F)

train.auc <- read.csv("train_auc.csv")
val.auc   <- read.csv("val_auc.csv")

time.cv   <- read.csv("timecv.csv")
time.fit  <- read.csv("timefit.csv")

# print out median of val AUC
apply(val.auc, 2, median)

# auc train and valid set graph
limits <- c(0.95, 1)
breaks <- seq(limits[1], limits[2], by=.01)

g1 <- train.auc %>%
  gather(key=models, value=train.auc) %>%
  ggplot(aes(x=models, y=train.auc, fill=models)) +
  geom_boxplot() +
  scale_y_continuous(limits=limits, breaks=breaks) + 
  scale_fill_viridis(discrete = TRUE, alpha=0.8, option="D") +
  scale_color_viridis(discrete = TRUE) +
  theme_bw() +
  theme(
    legend.position="none",
    plot.title = element_text(size=20, face = "bold"),
    axis.text.x = element_text(angle = 90, hjust = 1, size=18,color="darkred"),
    axis.text.y = element_text(hjust = 1, size=18,color="darkred"),
    axis.title.y = element_text(size = rel(1.8), angle = 90)
  ) +
  ggtitle("Train AUC") +
  xlab("")

g2 <- val.auc %>%
  gather(key=models, value=val.auc) %>%
  ggplot(aes(x=models, y=val.auc, fill=models)) +
  geom_boxplot() +
  scale_y_continuous(limits=limits, breaks=breaks) + 
  scale_fill_viridis(discrete = TRUE, alpha=0.8, option="D") +
  scale_color_viridis(discrete = TRUE) +
  theme_bw() +
  theme(
    legend.position="none",
    plot.title = element_text(size=20, , face = "bold"),
    axis.text.x = element_text(angle = 90, hjust = 1, size=18,color="darkred"),
    axis.text.y = element_text(hjust = 1, size=18,color="darkred"),
    axis.title.y = element_text(size = rel(1.8), angle = 90)
  ) +
  ggtitle("Valid AUC") +
  xlab("")

grid.arrange(g1, g2, ncol=2)

# boxplot for CV time
time.cv <- data.frame(time.cv)
time.cv %>%
  gather(key=models, value=time) %>%
  ggplot(aes(x=models, y=time, fill=models)) +
  geom_boxplot() +
  #scale_y_continuous(limits=limits, breaks=breaks) + 
  scale_fill_viridis(discrete = TRUE, alpha=0.8, option="C") +
  scale_color_viridis(discrete = TRUE) +
  theme_bw() +
  theme(
    legend.position="none",
    plot.title = element_text(size=20, face = "bold"),
    axis.text.x = element_text(angle = 90, hjust = 1, size=18,color="darkred"),
    axis.text.y = element_text(hjust = 1, size=18,color="darkred"),
    axis.title.y = element_text(size = rel(1.8), angle = 90)
  ) +
  ggtitle("Time CV") +
  xlab("")

apply(time.cv, 2, mean)

# decide weights again for imbalanced data
nP              <-      sum(y)
nN              <-      length(y) - nP
w               <-      rep(1, length(y))
w[y == 1]       <-      nN / nP
# weight for rf
w.rf            <-      c("0" = 1, "1" = nN/nP)

# fit ridge to the whole data
a=0 # ridge
start.time       <-     proc.time()
cv.fit           <-     cv.glmnet(X, y, alpha = a, nfolds = K,  weights = w, family="binomial", type.measure="auc")
rd.fit           <-     glmnet(X, y, alpha = a, lambda = cv.fit$lambda.min, family="binomial", weights = w, standardize = F)
end.time         <-     proc.time() - start.time
time.rd          <-     end.time['elapsed']
betaS.rd         <-     data.frame(colnames(X), as.vector(rd.fit$beta))
colnames(betaS.rd)     <-     c( "features", "coefficients")

# fit lasso to the whole data
a=1 # lasso
start.time       <-     proc.time()
cv.fit           <-     cv.glmnet(X, y, alpha = a, nfolds = K,  weights = w, family="binomial", type.measure="auc")
ls.fit           <-     glmnet(X, y, alpha = a, lambda = cv.fit$lambda.min, family="binomial", weights = w, standardize = F)
end.time         <-     proc.time() - start.time
time.ls          <-     end.time['elapsed'] 
betaS.ls         <-     data.frame(colnames(X), as.vector(ls.fit$beta))
colnames(betaS.ls)     <-     c( "features", "coefficients")

# fit en to the whole data
a=0.5 # elastic-net
start.time       <-     proc.time()
cv.fit           <-     cv.glmnet(X, y, alpha = a, nfolds = K,  weights = w, family="binomial", type.measure="auc")
el.fit           <-     glmnet(X, y, alpha = a, lambda = cv.fit$lambda.min, family="binomial", weights = w, standardize = F)
end.time         <-     proc.time() - start.time
time.el          <-     end.time['elapsed']
betaS.el         <-     data.frame(colnames(X), as.vector(el.fit$beta))
colnames(betaS.el)     <-     c( "features", "coefficients")

# fit rf to the whole data
start.time       <-    proc.time()
rf               <-     randomForest(X, y=as.factor(y), mtry = sqrt(p), classwt = w.rf, importance = TRUE)
end.time         <-    proc.time() - start.time
time.rf          <-    end.time['elapsed']
betaS.rf         <-    data.frame(colnames(X), as.vector(rf$importance[,1]))
colnames(betaS.rf)     <-     c( "features", "coefficients")

time.rd
time.ls
time.el
time.rf

# write.csv(betaS.rd, file="betaS_rd.csv", row.names=F)
# write.csv(betaS.ls, file="betaS_ls.csv", row.names=F)
# write.csv(betaS.el, file="betaS_el.csv", row.names=F)
# write.csv(betaS.rf, file="betaS_rf.csv", row.names=F)

betaS.rd <- read.csv("betaS_rd.csv")
betaS.ls <- read.csv("betaS_ls.csv")
betaS.el <- read.csv("betaS_el.csv")
betaS.rf <- read.csv("betaS_rf.csv")

length(which(betaS.ls$coefficients != 0))
length(which(betaS.el$coefficients != 0))

# we need to change the order of factor levels by specifying the order explicitly.
betaS.el$features     =  factor(betaS.el$features, levels = betaS.el$features[order(betaS.el$coefficients, decreasing = TRUE)])
betaS.ls$features     =  factor(betaS.ls$features, levels = betaS.el$features[order(betaS.el$coefficients, decreasing = TRUE)])
betaS.rd$features     =  factor(betaS.rd$features, levels = betaS.el$features[order(betaS.el$coefficients, decreasing = TRUE)])
betaS.rf$features     =  factor(betaS.rf$features, levels = betaS.el$features[order(betaS.el$coefficients, decreasing = TRUE)])

options(repr.plot.width=25, repr.plot.height=15)
el =  ggplot(betaS.el, aes(x=features, y=coefficients, fill=features)) +
  geom_bar(stat = "identity")    +
  labs(x = element_blank()
       , y = element_blank()
       , title = expression(Elastic)
  ) +
  theme(
    panel.background = element_rect(fill="#E8EDFB")
    , axis.text.x = element_blank()
    , axis.text.y = element_text(size=15)
    , plot.title = element_text(size=20, , face = "bold") 
    , legend.position = "none"
  )

ls =  ggplot(betaS.ls, aes(x=features, y=coefficients, fill=features)) +
  geom_bar(stat = "identity")    +
  labs(x = element_blank()
       , y = element_blank()
       , title = expression(LASSO)
  ) +
  theme(
    panel.background = element_rect(fill="#E8EDFB")
    , axis.text.x = element_blank()
    , axis.text.y = element_text(size=15)
    , plot.title = element_text(size=20, , face = "bold")
    , legend.position = "none"
  )

rd =  ggplot(betaS.rd, aes(x=features, y=coefficients, fill=features)) +
  geom_bar(stat = "identity")    +
  labs(
    x = element_blank()
    , y = element_blank()
    , title = expression(Ridge)
  ) +
  theme(
    panel.background = element_rect(fill="#E8EDFB")
    , axis.text.x = element_blank()
    , axis.text.y = element_text(size=15)
    , plot.title = element_text(size=20, , face = "bold")
    , legend.position = "none"
  )

rf =  ggplot(betaS.rf, aes(x=features, y=coefficients, fill=features)) +
  geom_bar(stat = "identity")    +
  labs(
    x = element_blank()
    , y = element_blank()
    , title = expression(Randon~Forest)
  ) +
  theme(
    panel.background = element_rect(fill="#E8EDFB")
    , axis.text.x = element_blank()
    , axis.text.y = element_text(size=15)
    , plot.title = element_text(size=20, face = "bold")
    , legend.position = "none"
  )

grid.arrange(el, ls, rd, rf, nrow = 4)

# beta coeficients heatmap
g.rd <- betaS.rd %>%
  tidyr::extract(features, "pixel", "(\\d+)", convert = TRUE) %>%  
  mutate(x = pixel %% 28 + 1, y = 28 - pixel %/% 28) %>%
  ggplot(aes(x, y, fill = coefficients)) +  geom_tile() + 
  labs(title = "Ridge Features") + 
  scale_fill_gradient(low="black", high="white") 

g.ls <- betaS.ls %>%
  tidyr::extract(features, "pixel", "(\\d+)", convert = TRUE) %>%  
  mutate(x = pixel %% 28 + 1, y = 28 - pixel %/% 28) %>%
  ggplot(aes(x, y, fill = coefficients)) +  geom_tile() +  
  labs(title = "Lasso Features") +
  scale_fill_gradient(low="black", high="white") 

g.el <- betaS.el %>%
  tidyr::extract(features, "pixel", "(\\d+)", convert = TRUE) %>%  
  mutate(x = pixel %% 28 + 1, y = 28 - pixel %/% 28) %>%
  ggplot(aes(x, y, fill = coefficients)) +  geom_tile() +  
  labs(title = "ElasticNet Features") +
  scale_fill_gradient(low="black", high="white") 

g.rf <- betaS.rf %>%
  tidyr::extract(features, "pixel", "(\\d+)", convert = TRUE) %>%  
  mutate(x = pixel %% 28 + 1, y = 28 - pixel %/% 28) %>%
  ggplot(aes(x, y, fill = coefficients)) +  geom_tile() +  
  labs(title = "RF Features") +
  scale_fill_gradient(low="black", high="white") 

grid.arrange(g.rd, g.ls, g.el, g.rf, nrow = 2, ncol = 2)
