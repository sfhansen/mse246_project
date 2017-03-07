graphics.off()
rm(list = ls())


df = readRDS('tranche_prob_df.rds')
df2 = readRDS('merged.rds')

# Create new data frame [Approval Year, Approval Amount, Default Year (NA if no def), Termination Year]
mat = rep(0,0); # Initialize
mat = cbind(mat, as.numeric(levels(df[,3]))[df[,3]]); # Append approval year
mat = cbind(mat, df[,4]); # Append approval amt
mat = cbind(mat, as.numeric(format(df2$ChargeOffDate, '%Y'))); # Append Default Year
mat = cbind(mat, as.numeric(levels(df[,3]))[df[,3]] + ceiling(df2$TermInMonths/12)) # Append term year

# Generate a simulation
# Determine active loans
cat("\014")
for (year in 1990:2013){
  print(year)
  active1 = rep(0,0);
  active5 = rep(0,0);
  for (i in 1:54794){
    if ((mat[i,1]<=year) && (is.na(mat[i,3]) || mat[i,3]>= year)){
      if (mat[i,4] >= year+1){
        active1 = rbind(active1, mat[i,])
      }
      if (mat[i,4] >= year+5){
        active5 = rbind(active5, mat[i,])
      }
    }
  }
  
  loss_percent1 = rep(0,0)
  loss_percent5 = rep(0,0)
  for (iter in 1:10000){
    
    # Pick 500 loans uniformly random from active loan list
    sim1 = sample(1:dim(active1)[1], 500)
    sim1 = active1[sim1,]
    sim5 = sample(1:dim(active5)[1], 500)
    sim5 = active5[sim5,]
    
    # Determine total value of portfolio
    value1 = sum(sim1[,2]/(sim1[,4]-sim1[,1]))
    value5 = sum(sim5[,2]/(sim5[,4]-sim5[,1]))
    
    # Determine default loans 
    def1 = sim1[!is.na(sim1[,3]),]
    def5 = sim5[!is.na(sim5[,3]),]
    
    # Determine total loss
    loss1 = sum(def1[,2]/(def1[,4]-def1[,1]))
    loss5 = 0;
    if (is.null(dim(def5))){
      loss5 = def5[2]/(def5[4]-def[1])
    } else {
      if ((dim(def5)[1]) > 1){
        for (i in 1:(dim(def5)[1])){
          loss5 = loss5 + def5[i,2]/(def5[i,4]-def5[i,1])
        }
      }
    }
    # Determine ratio of loss
    loss_percent1 = c(loss_percent1, loss1/value1);
    loss_percent5 = c(loss_percent5, loss5/value5);
  }
  
  # Initialize Vector of Tranche Losses
  tranche1_5_15 = rep(0, length(loss_percent1))
  tranche1_15_100 = rep(0, length(loss_percent1))
  tranche5_5_15 = rep(0, length(loss_percent5))
  tranche5_15_100 = rep(0, length(loss_percent5))
  
  # Tranche [.05, .15] loss for 1 year
  for (i in 1:length(loss_percent1)){
    if (loss_percent1[i]<.05){
      tranche1_5_15[i] = 0;
    } else {
      if (loss_percent1[i] > 0.15){
        tranche1_5_15[i] = 1;
      } else{
        tranche1_5_15[i] = 10*(loss_percent1[i] - .05)
      }
    }
  }
  
  # Tranche [.05, .15] loss for 5 year
  for (i in 1:length(loss_percent5)){
    if (loss_percent5[i]<.05){
      tranche5_5_15[i] = 0;
    } else {
      if (loss_percent5[i] > 0.15){
        tranche5_5_15[i] = 1;
      } else{
        tranche5_5_15[i] = 10*(loss_percent5[i] - .05)
      }
    }
  }
  
  # Tranche [.15, 1] loss for 1 year
  for (i in 1:length(loss_percent1)){
    if (loss_percent1[i]<.15){
      tranche1_15_100[i] = 0;
    } else {
      tranche1_15_100[i] = 100/85*(loss_percent1[i] - .15)
    }
  }
  
  # Tranche [.15, 1] loss for 1 year
  for (i in 1:length(loss_percent5)){
    if (loss_percent5[i]<.15){
      tranche5_15_100[i] = 0;
    } else {
      tranche5_15_100[i] = 100/85*(loss_percent5[i] - .15)
    }
  }
  
  # Plot results and save figures
  y_max = max(c(max(density(tranche1_5_15, bw = .0075, from = 0, to = 1)$y), max(density(tranche1_15_100, bw = .0075, from = 0, to = 1)$y)))
  png(filename = paste("./DefaultDist/1yr", year, "tranche.png", sep = "_"))
  plot(density(tranche1_5_15, bw = .0075, from = 0, to = 1), 
       main = paste("Loss for 1-year Portfolio in", year, "for tranches [5,15] and [15,100]", sep = " "), 
       xlab = "Loss", ylab = "Density", 
       xlim = c(0,1), ylim = c(0, min(c( 15, y_max))), col = 'blue');
  lines(density(tranche1_15_100, bw = .0075, from = 0, to = 1), col = 'red')
  legend('top', c('Tranche [0,15]', 'Tranche [15,100]'), col = c('blue', 'red'), lwd = 2)
  dev.off()
  
  y_max = max(c(max(density(tranche5_5_15, bw = .0075, from = 0, to = 1)$y), max(density(tranche5_15_100, bw = .0075, from = 0, to = 1)$y)))
  png(filename = paste("./DefaultDist/5yr", year, "tranche.png", sep = "_"))
  plot(density(tranche5_5_15, bw = .0075, from = 0, to = 1), 
       main = paste("Loss for 5-year Portfolio in", year, "for tranches [5,15] and [15,100]", sep = " "), 
       xlab = "Loss", ylab = "Density", 
       xlim = c(0,1), ylim = c(0, min(c( 15, y_max))), col = 'blue');
  lines(density(tranche5_15_100, bw = .0075, from = 0, to = 1), col = 'red')
  legend('top', c('Tranche [5,15]', 'Tranche [15,100]'), col = c('blue', 'red'), lwd = 2)
  dev.off()
  
  y_max = max(c(max(density(tranche1_5_15, bw = .0075, from = 0, to = 1)$y), max(density(tranche5_15_100, bw = .0075, from = 0, to = 1)$y)))
  png(filename = paste("./DefaultDist/full", year, "tranche.png", sep = "_"))
  plot(density(tranche5_5_15, bw = .0075, from = 0, to = 1), 
       main = paste("Loss Distribution for 1- and 5-year Portfolio in", year, "for tranches [5,15] and [15,100]", sep = " "), 
       xlab = "Loss", ylab = "Density", 
       xlim = c(0,1), ylim = c(0, min(c( 15, y_max))), col = 'blue');
  lines(density(tranche5_5_15, bw = .0075, from = 0, to = 1), col = 'red')
  lines(density(tranche5_15_100, bw = .0075, from = 0, to = 1), col = 'green')
  lines(density(tranche5_15_100, bw = .0075, from = 0, to = 1), col = 'orange')
  legend('top', c('1-yr Tranche [5,15]', '5-yr Tranche [5,15]', '1-yr Tranche [15,100]', '5-yr Tranche [15,100]'), col = c('blue', 'red', 'green', 'orange'), lwd = 2)
  dev.off()
}
