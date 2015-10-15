library(compare)
library(cluster)
library(dynamicTreeCut)
library(FNN)
library(mlbench)
library(caret)
library(mclust)
library(vegan)

categorical <- c(
  'Community Name', 'Region', 'Map reference', 'Grid reference', 'Location', 'LGA',
  'Primary Care Partnership', 'Medicare Local', 'DHS Area', 'Top industry', '2nd top industry - persons',
  '3rd top industry - persons', 'Top occupation', '2nd top occupation - persons', '3rd top occupation - persons',
  'Top country of birth', '2nd top country of birth', '3rd top country of birth', '4th top country of birth',
  '5th top country of birth', 'Top language spoken', '2nd top language spoken', '3rd top language spoken',
  '4th top language spoken', '5th top language spoken', 'Nearest Public Hospital',
  'Nearest public hospital with maternity services', 'Nearest public hospital with emergency department',
  'Medicare Access Points', 'Bush Nursing Centres'
)

# geography
group_1 <- c(
  'Population Density',
  'Travel time to GPO (minutes)',
  'Distance to GPO (km)',
  'LGA',
  'Primary Care Partnership',
  'Medicare Local',
  'Area (km^2)',
  'DHS Area'
)

# hospital
group_2 <- c(
  'Public hospital separations, 2012-13',
  'Nearest Public Hospital',
  'Travel time to nearest public hospital',
  'Distance to nearest public hospital',
  'Obstetric type separations, 2012-13',
  'Nearest public hospital with maternity services',
  'Time to nearest public hospital with maternity services',
  'Distance to nearest public hospital with maternity services',
  'Presentations to emergency departments, 2012-13',
  'Nearest public hospital with emergency department',
  'Travel time to nearest public hospital with emergency department',
  'Distance to nearest public hospital with emergency department',
  'Presentations to emergency departments due to injury',
  'Presentations to emergency departments due to injury, %',
  'Category 4 & 5 emergency department presentations',
  'Category 4 & 5 emergency department presentations, %'
)

# land use
group_3 <- c(
  'Commercial (km^2)',
  'Commercial (%)',
  'Industrial (km^2)',
  'Industrial (%)',
  'Residential (km^2)',
  'Residential (%)',
  'Rural (km^2)',
  'Rural (%)',
  'Other (km^2)',
  'Other (%)'
)

#socio-demographic
group_4 <- c(
  'Number of Households',
  'Average persons per household',
  'Occupied private dwellings',
  'Occupied private dwellings, %',
  'Population in non-private dwellings',
  'Public Housing Dwellings',
  '% dwellings which are public housing',
  'Dwellings with no motor vehicle',
  'Dwellings with no motor vehicle, %',
  'Dwellings with no internet',
  'Dwellings with no internet, %',
  'Equivalent household income <$600/week',
  'Equivalent household income <$600/week, %',
  'Personal income <$400/week, persons',
  'Personal income <$400/week, %',
  'Number of families',
  'Female-headed lone parent families',
  'Female-headed lone parent families, %',
  'Male-headed lone parent families',
  'Male-headed lone parent families, %',
  '% residing near PT',
  'IRSD (min)',
  'IRSD (max)',
  'IRSD (avg)',
  'Primary school students',
  'Secondary school students',
  'TAFE students',
  'University students',
  'Holds degree or higher, persons',
  'Holds degree or higher, %',
  'Did not complete year 12, persons',
  'Did not complete year 12, %',
  'Unemployed, persons',
  'Unemployed, %',
  'Volunteers, persons',
  'Volunteers, %',
  'Requires assistance with core activities, persons',
  'Requires assistance with core activities, %',
  'Aged 75+ and lives alone, persons',
  'Aged 75+ and lives alone, %',
  'Unpaid carer to person with disability, persons',
  'Unpaid carer to person with disability, %',
  'Unpaid carer of children, persons',
  'Unpaid carer of children, %',
  'Top industry',
  'Top industry, %',
  '2nd top industry - persons',
  '2nd top industry, %',
  '3rd top industry - persons',
  '3rd top industry, %',
  'Top occupation',
  'Top occupation, %',
  '2nd top occupation - persons',
  '2nd top occupation, %',
  '3rd top occupation - persons',
  '3rd top occupation, %'
)

# subgroup of socio-demographic (education)
group_5 <- c(
  'Dwellings with no internet',
  'Dwellings with no internet, %',
  'Primary school students',
  'Secondary school students',
  'TAFE students',
  'University students',
  'Holds degree or higher, persons',
  'Holds degree or higher, %',
  'Did not complete year 12, persons',
  'Did not complete year 12, %',
  'Unemployed, persons',
  'Unemployed, %',
  'Volunteers, persons',
  'Volunteers, %'
)

# services
group_6 <- c(
  'Public Hospitals',
  'Private Hospitals',
  'Community Health Centres',
  'Bush Nursing Centres',
  'Allied Health',
  'Alternative Health',
  'Child Protection and Family',
  'Dental',
  'Disability',
  'General Practice',
  'Homelessness',
  'Mental Health',
  'Pharmacies',
  'Aged Care (High Care)',
  'Aged Care (Low Care)',
  'Aged Care (SRS)',
  'Kinder and/or Childcare',
  'Primary Schools',
  'Secondary Schools',
  'P12 Schools',
  'Other Schools',
  'Centrelink Offices',
  'Medicare Offices',
  'Medicare Access Points'
)

group_7 <- c(
  'Aboriginal or Torres Strait Islander, persons',
  'Aboriginal or Torres Strait Islander, %',
  'Born overseas, persons',
  'Born overseas, %',
  'Born in non-English speaking country, persons',
  'Born in non-English speaking country, %',
  'Speaks LOTE at home, persons',
  'Speaks LOTE at home, %',
  'Poor English proficiency, persons',
  'Poor English proficiency, %',
  'Top country of birth',
  'Top country of birth, persons',
  'Top country of birth, %',
  '2nd top country of birth',
  '2nd top country of birth, persons',
  '2nd top country of birth, %',
  '3rd top country of birth',
  '3rd top country of birth, persons',
  '3rd top country of birth, %',
  '4th top country of birth',
  '4th top country of birth, persons',
  '4th top country of birth, %',
  '5th top country of birth',
  '5th top country of birth, persons',
  '5th top country of birth, %',
  'Top language spoken',
  'Top language spoken, persons',
  'Top language spoken, %',
  '2nd top language spoken',
  '2nd top language spoken, persons',
  '2nd top language spoken, %',
  '3rd top language spoken',
  '3rd top language spoken, persons',
  '3rd top language spoken, %',
  '4th top language spoken',
  '4th top language spoken, persons',
  '4th top language spoken, %',
  '5th top language spoken',
  '5th top language spoken, persons',
  '5th top language spoken, %'
)

group <- group_1

num_nn <- 5

dataset <- read.csv("data/input.csv", check.names = FALSE)

# select features
indices0 <- match(group, names(dataset))
dataset_selected <- dataset[ , c(indices0)]

# delete highly correlated
indices <- match(group, names(dataset_selected))
cat_indices <- match(categorical, names(dataset_selected))
if(length(intersect(cat_indices, indices)) > 0){
  corr_matrix <- cor(dataset_selected[ , -c(intersect(cat_indices, indices))])
} else{
  corr_matrix <- cor(dataset_selected)
}
highly_correlated <- findCorrelation(corr_matrix, cutoff=0.8)

if(length(highly_correlated) != 0){
  dataset_uncorr <- dataset_selected[ , -c(highly_correlated)]
} else{
  dataset_uncorr <- dataset_selected
}
dataset_uncorr

ws <- rep(1, length(names(dataset_uncorr)))
ws[match('Top language spoken, %', names(dataset_uncorr))] <- 100

# generate dissimilarity matrix using gower distance
diss<-daisy(dataset_uncorr, metric = c("gower"),stand = TRUE, weights = ws)
simm.mat<- as.matrix(diss)

# perform MDS
fit <- cmdscale(simm.mat, eig=TRUE, k=2) # k is the number of dim
# isomap
#fit <- isomap(simm.mat, k=2, fragmentedOK = TRUE)
#summary(fit)
#print(fit$points)
#plot(fitxlab="Coordinate 1", ylab="Coordinate 2", main="MDS",	type="p")
#text(x, y, labels=1:34, cex= 0.7, pos=3)

# plot solution 
x <- fit$points[,1]
y <- fit$points[,2]
plot(x, y, xlab="Coordinate 1", ylab="Coordinate 2", main="MDS",	type="p")
text(x, y, labels=1:34, cex= 0.7, pos=3)


# find 5 nearest neighbours in MDS
nn <- get.knnx(fit$points, fit$points, num_nn + 1)
mds_nn <- nn$nn.index[, -1]

# plot location
coord_dataset <- read.csv("data/coordinates.csv")
coords <- coord_dataset[ , c(2,3)]
x_coord <- coords[,1]
y_coord <- coords[,2]
plot(x_coord, y_coord, xlab="Coordinate 1", ylab="Coordinate 2", main="Location",	type="p")
text(x_coord, y_coord, labels=1:34, cex= 0.7, pos=3)

# find 5 nearest neighbours in Location
loc_nn <- get.knnx(coords, coords, num_nn + 1)
location_nn <- loc_nn$nn.index[, -1]

similarity_index <- 0
for(i in 1:34){
  m <- match(mds_nn[i, ], location_nn[i, ])
  m <- m[!is.na(m)]
  similarity_index <- similarity_index + length(m)
}
similarity_index / (num_nn * 34)

#loc_clust <- Mclust(coords)
#summary(loc_clust, what = 'classification')
#plot(loc_clust, what = 'classification')
#text(x_coord, y_coord, labels=1:34, cex= 0.7, pos=3)

#xyMclust <- Mclust(fit$points, G = 5)
#summary(xyMclust, what = 'classification')
#plot(xyMclust, what = 'classification')
#text(x, y, labels=1:34, cex= 0.7, pos=3)



