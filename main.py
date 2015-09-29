from Formatter import Formatter


def createScikitLearnFile():
    dataFile = '/Users/davidamores/Documents/UniMelb/Semester 3/Statistical Machine Learning/Project 2/matrixMod.csv'
    outputFile = '/Users/davidamores/Documents/UniMelb/Semester 3/Statistical Machine Learning/Project 2/output.csv'
    formatter = Formatter()
    formatter.csvToScikitLearn(dataFile, outputFile)


createScikitLearnFile()