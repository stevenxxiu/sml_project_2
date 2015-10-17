
data = read.csv('data/binarized.csv')
data[is.na(data)] = 0
data_lm1 = step(lm(Unemployed...~1, data=data), as.formula(paste('~. + ', paste(names(data)[-which(names(data) == 'Unemployed...')], collapse=' + '))))
data_anova = anova(data_lm1)
par(mar=c(7.1, 4.1, 4.1, 2.1))
plot(data_anova$`Sum Sq`[1:5], ylim=c(0, sum(data_anova$`Sum Sq`)), ylab='anova sum of squares', xaxt='n', xlab='')
axis(1, 1:5, labels=c(
    "Poor\nEnglish\nproficiency, %",
    "2007 ERP\nage 20-24, %",
    "Top\noccupation,\n%",
    "4th top\ncountry\nof birth\n= Philippines",
    "Born in\nnon-English\nspeaking\ncountry,\npersons"
), padj=1)

data = merge(read.csv('data/housing.csv'), read.csv('data/binarized.csv'), by='Community Name', all=TRUE)
