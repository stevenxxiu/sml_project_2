
data = read.csv('data/binarized.csv')
data[is.na(data)] = 0
data_lm1 = step(lm(Unemployed...~1, data=data), as.formula(paste('~. + ', paste(names(data)[-which(names(data) == 'Unemployed...')], collapse=' + '))))
