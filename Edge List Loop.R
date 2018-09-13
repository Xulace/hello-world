setwd("H:/NBA SNA")

data <- read.csv("Dyads Clean Passes.csv",header = TRUE, stringsAsFactors = FALSE)

masterlist<- list()
gameIDs <- unique(data$matchid)

for(i in 1:length(gameIDs)){
  tempGameID <- gameIDs[i]
  tempSubData <- subset(x = data, subset = matchid == tempGameID)
  tempTeams <- unique(tempSubData$id_t_br)
  
  tempSubDataT1 <- subset(x = tempSubData, subset = id_t_br == tempTeams[1])
  tempSubDataT2 <- subset(x = tempSubData, subset = id_t_br == tempTeams[2])
  
  tempT1players <- sort(unique(unique(tempSubDataT1$id_p_adv), unique(tempSubDataT1$id_p_pen)))
  tempT2players <- sort(unique(unique(tempSubDataT2$id_p_adv), unique(tempSubDataT2$id_p_pen)))
  
  tempT1Matrix <- matrix(data = NA, nrow = length(tempT1players), ncol = length(tempT1players))
  tempT2Matrix <- matrix(data = NA, nrow = length(tempT2players), ncol = length(tempT2players))
  
  for(j in 1:length(tempT1players)){
    for(k in 1:length(tempT1players)){
      tempT1Matrix[j,k] = sum(tempSubDataT1$id_p_adv == tempT1players[j]&tempSubDataT1$id_p_pen == tempT1players[k])
    }
  }
  rownames(tempT1Matrix) <- colnames(tempT1Matrix) <- tempT1players
  
  for(j in 1:length(tempT2players)){
    for(k in 1:length(tempT2players)){
      tempT2Matrix[j,k] = sum(tempSubDataT2$id_p_adv == tempT2players[j]&tempSubDataT2$id_p_pen == tempT2players[k])
    }
  }
  rownames(tempT2Matrix) <- colnames(tempT2Matrix) <- tempT2players
  
  tempT1EdgeList = data.frame(Passer = NA, Receiver = NA, Weight = NA, stringsAsFactors = FALSE)[-1,]
  for(j in 1:length(tempT1players)){
    for(k in 1:length(tempT1players)){
      if(tempT1Matrix[j,k]>0){
        tempT1EdgeList <- rbind(tempT1EdgeList,
        data.frame(
          rownames(tempT1Matrix)[j],
          colnames(tempT1Matrix)[k],
          tempT1Matrix[j,k]), stringsAsFactors = FALSE
        )
      }
    }
  }
  colnames(tempT1EdgeList) = c("Passer", "Receiver", "Weight")
  tempT1EdgeList$Passer = as.character(tempT1EdgeList$Passer)
  tempT1EdgeList$Receiver = as.character(tempT1EdgeList$Receiver)
  
  tempT2EdgeList = data.frame(Passer = NA, Receiver = NA, Weight = NA, stringsAsFactors = FALSE)[-1,]
  for(j in 1:length(tempT2players)){
    for(k in 1:length(tempT2players)){
      if(tempT2Matrix[j,k]>0){
        tempT2EdgeList <- rbind(tempT2EdgeList,
                                data.frame(
                                  rownames(tempT2Matrix)[j],
                                  colnames(tempT2Matrix)[k],
                                  tempT2Matrix[j,k]), stringsAsFactors = FALSE
        )
      }
    }
  }
  colnames(tempT2EdgeList) = c("Passer", "Receiver", "Weight")
  tempT2EdgeList$Passer = as.character(tempT2EdgeList$Passer)
  tempT2EdgeList$Receiver = as.character(tempT2EdgeList$Receiver)
  
tempList <- list(
  tempT1EdgeList,
  tempT2EdgeList
)
names(tempList) = tempTeams
masterlist[[i]] <- tempList
names(masterlist)[i] <- tempGameID
print(i/length(gameIDs))
}

save(masterlist, file="H:/MasterList.RData")
