#here i load the CSV file that contains removed correlated features

dataset<- read.csv("data/finalallcor.csv", header = T , sep=",")
# here I defined the normalization function 
doit <- function(x) {(x - min(x, na.rm=TRUE))/(max(x,na.rm=TRUE) -
                                                 min(x, na.rm=TRUE))}
# I selcted numeric features and apply the normalization function, be carefull in your dataset it may be different colomns
dataset[ ,c(3,6,7,11,14:68,70,72,74,77,79:81,83,88,91,94,97,100:102)]<- as.data.frame(lapply(dataset[ ,c(3,6,7,11,14:68,70,72,74,77,79:81,83,88,91,94,97,100:102)], doit))
# here I calculate the variance of each colomn and multiply the variance to whole that colomn
dataset[ ,c(3)]<-var(dataset[ ,c(3)])*dataset[ ,c(3)]
dataset[ ,c(6)]<-var(dataset[ ,c(6)])*dataset[ ,c(6)]
dataset[ ,c(7)]<-var(dataset[ ,c(7)])*dataset[ ,c(7)]
dataset[ ,c(11)]<-var(dataset[ ,c(11)])*dataset[ ,c(11)]
for(i in 14:68){
  dataset[ ,c(i)]<-var(dataset[ ,c(i)])*dataset[ ,c(i)]
  i<-i+1
}
dataset[ ,c(70)]<-var(dataset[ ,c(70)])*dataset[ ,c(70)]
dataset[ ,c(72)]<-var(dataset[ ,c(72)])*dataset[ ,c(72)]
dataset[ ,c(74)]<-var(dataset[ ,c(74)])*dataset[ ,c(74)]
dataset[ ,c(77)]<-var(dataset[ ,c(77)])*dataset[ ,c(77)]
for(j in 79:81){
  dataset[ ,c(j)]<-var(dataset[ ,c(j)])*dataset[ ,c(j)]
  j<-j+1
}
dataset[ ,c(83)]<-var(dataset[ ,c(83)])*dataset[ ,c(83)]
dataset[ ,c(88)]<-var(dataset[ ,c(88)])*dataset[ ,c(88)]
dataset[ ,c(91)]<-var(dataset[ ,c(91)])*dataset[ ,c(91)]
dataset[ ,c(94)]<-var(dataset[ ,c(94)])*dataset[ ,c(94)]
dataset[ ,c(97)]<-var(dataset[ ,c(97)])*dataset[ ,c(97)]
for(k in 100:102){
  dataset[ ,c(k)]<-var(dataset[ ,c(k)])*dataset[ ,c(k)]
  k<-k+1
}


# here I calculate the disssimilarity function "gower"
library(cluster)
library(dynamicTreeCut)
diss<-daisy(dataset[ ,c(26,27,28,30,31,31,33,36)], metric = c("gower"),stand = TRUE)
# I convert it to matrix
simm.mat<- as.matrix(diss)
#I calculate the hierarchical clustering
CLU<-hclust(diss)
#I plot it
plot(CLU)

