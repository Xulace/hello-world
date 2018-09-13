###
# degree code
###

dat <- MEM_Edge_List

inPass <- tapply(dat$Weight, as.factor(dat$Receiver), sum)
outPass <- tapply(dat$Weight, as.factor(dat$Passer), sum)
totalPass <- outPass + inPass
sum(totalPass)
mean(totalPass)

totalPass <- totalPass/max(totalPass)
sum(max(totalPass) - totalPass)/(length(totalPass)-1)

inPass <- inPass/max(inPass)
sum(max(inPass) - inPass)/(length(inPass)-1)

outPass <- outPass/max(outPass)
sum(max(outPass) - outPass)/(length(outPass)-1) ##for different-sized networks
