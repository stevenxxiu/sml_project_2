dataset<- read.csv("dataprocessed.csv", header = T , sep=";")
set.seed(7)
# load the library
library(mlbench)
library(caret)
# in this line I just select the numeric columns and calculate the correlation
correlationMatrix <- cor(dataset[ ,28:90])
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.8)
#removing Highly correlated column
dataset1<- dataset[ , -c(#here you put the column that you want to remove the minus sign means that you remove the colomn)]
 # here you write you new csv file 
write.csv(dataset1, file = "datacorpopulationchange.csv",row.names=FALSE, na="")

