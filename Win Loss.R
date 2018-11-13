DCPcoll <- subset(x = Dyads.Clean.Passes, select = c(matchid, awayteam, hometeam, awayscore, homescore))
DCPcoll <- unique(DCPcoll)
head(sort(table(DCPcoll$matchid), decreasing = TRUE))

dat$Win = rep(NA, nrow(dat))

for(i in 1:nrow(dat)){
  tempind <- which(DCPcoll$matchid == dat$matchid[i])
  tempwinner <- ifelse(DCPcoll$awayscore[ceiling(i/2)] > DCPcoll$homescore[ceiling(i/2)], 
                       DCPcoll$awayteam[ceiling(i/2)], DCPcoll$hometeam[ceiling(i/2)])
  dat$Win[i] <- ifelse(dat$teamid[i] == tempwinner, 1, 0)
  print(i)
}

save(dat, file="H:/NBA SNA/Master List Centr.RData")

library(caret)

Train <- createDataPartition(dat$Win, p = 0.5, list = FALSE)
training <- dat[Train,]
testing <- dat[-Train,]

mod_fit <-  glm(data = training, formula = Win ~ incentr + outcentr +
                 reciprocity, family = binomial)
#+ teamid + totcentr  totaldegree + + Centrality 

summary(mod_fit)
exp(coef(mod_fit))

predict(mod_fit, newdata=testing, type="response")

dat.prob = predict(mod_fit, testing, type = "response")
dat.pred = rep("0", dim(training)[1])
dat.pred[dat.prob > .5] = "1"
table(dat.pred, training$Win)

mean(dat.pred == training$Win)

model <- glm(data = dat, formula = Win ~ incentr + outcentr + totcentr +
               totaldegree + reciprocity + Centrality + teamid, family = binomial)
summary(model)
exp(coef(model))
confint(model)

corm <- dat[, c(3,4,5,7,8,9)]
cor(corm)
install.packages("corrplot")
source("http://www.sthda.com/upload/rquery_cormat.r")
rquery.cormat(corm, type="flatten", graph=FALSE)

anova(model, test = "Chisq")

modelT <- glm(data=dat, formula = Win ~ teamid)
summary(modelT)
confint(modelT)
exp(coef(modelT))

BTfit <- lm(Win ~ incentr + outcentr + totcentr +
              totaldegree + reciprocity + Centrality + teamid, data=dat)
summary(BTfit)
exp(coef(BTfit))

anova(modelT, test = "Chisq")
