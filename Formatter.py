import csv
from sklearn.feature_extraction import DictVectorizer

'''
This class is in charge of converting any type of data we may need
into data readable by scikit-learn or any other package we may need
to use.
'''

class Formatter:
    def __init__(self):
        self.categoricalFeats = [
            "Region",
            "Map reference",
            "Grid reference",
            "Location (Direction)",
            "LGA",
            "Primary Care Partnership",
            "Medicare Local",
            "DHS Area",
            "Top industry",
            "2nd top industry - persons",
            "3rd top industry - persons",
            "Top occupation",
            "2nd top occupation - persons",
            "3rd top occupation - persons",
            "Top country of birth",
            "2nd top country of birth",
            "3rd top country of birth",
            "4th top country of birth",
            "5th top country of birth",
            "Top language spoken",
            "2nd top language spoken",
            "3rd top language spoken",
            "4th top language spoken",
            "5th top language spoken",
            "Nearest Public Hospital",
            "Nearest public hospital with maternity services",
            "Nearest public hospital with emergency department",
        ]


    '''
    This method converts a csv file containing all the features of every suburb
    into a scikit-learn readable data structures. First, we turn the data into
    dictionaries (e.g. {'Community Name': 'Ascot Vale (Suburb)', ... ,'Pharmacies': 4, ...}).
    Then we convert it using scikit-learn's DictVectorizer.

    Note that:
        - Feature "Location" is split into two features:
            1. Location (distance km) - records the distance in km
            2. Location (direction) - records the direction (e.g. NW)
        - Some continuous values have the value "<5", this was turned into 1
        - "n/a" values were turned into None
    '''
    def csvToScikitLearn(self, dataFile, outputFile):

        testDataFile = open(dataFile, 'r')
        reader = csv.reader(testDataFile)

        featureNames = next(reader)

        matrix = []

        for aLine in reader:
            instance = {}
            i = 0
            for featValue in aLine:

                if featureNames[i] == "Location":
                    aFeatValue = featValue.split()
                    distance = aFeatValue[0]
                    direction = aFeatValue[1]
                    instance["Location (distance km)"] = float(distance.replace("km", ""))
                    instance["Location (direction)"] = direction
                    i += 1
                    continue

                if featureNames[i] in self.categoricalFeats or i == 0:
                    if featValue == "n/a":
                        instance[featureNames[i]] = None
                    else:
                        instance[featureNames[i]] = featValue
                else:
                    if "<" in featValue:
                        featValue = "1"
                    featValue = featValue.replace(",","")
                    if featValue == "n/a":
                        instance[featureNames[i]] = None
                    else:
                        instance[featureNames[i]] = float(featValue)
                i += 1
            matrix.append(instance)

        vec = DictVectorizer()
        vec.fit_transform(matrix).toarray()
        with open(outputFile, "wb") as f:
            writer = csv.writer(f)
            writer.writerow(vec.get_feature_names())
            writer.writerows(vec.fit_transform(matrix).toarray())