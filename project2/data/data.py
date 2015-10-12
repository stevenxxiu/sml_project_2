
import csv
import pickle

from sklearn.feature_extraction import DictVectorizer


def conv(in_sr, out_sr, pickle_sr):
    '''
    This method converts a csv file containing all the features of every suburb
    into a scikit-learn readable data structures. First, we turn the data into
    dictionaries (e.g. {'Community Name': 'Ascot Vale (Suburb)', ... , 'Pharmacies': 4, ...}).
    Then we convert it using scikit-learn's DictVectorizer.

    Note that:
        - Feature 'Location' is split into two features:
            1. Location (distance km) - records the distance in km
            2. Location (direction) - records the direction (e.g. NW)
        - Some continuous values have the value '<5', this was turned into 1
        - 'n/a' values were turned into None
    '''
    categorical_names = [
        'Community Name', 'Region', 'Map reference', 'Grid reference', 'Location (Direction)', 'LGA',
        'Primary Care Partnership', 'Medicare Local', 'DHS Area', 'Top industry', '2nd top industry - persons',
        '3rd top industry - persons', 'Top occupation', '2nd top occupation - persons', '3rd top occupation - persons',
        'Top country of birth', '2nd top country of birth', '3rd top country of birth', '4th top country of birth',
        '5th top country of birth', 'Top language spoken', '2nd top language spoken', '3rd top language spoken',
        '4th top language spoken', '5th top language spoken', 'Nearest Public Hospital',
        'Nearest public hospital with maternity services', 'Nearest public hospital with emergency department',
    ]

    reader = csv.reader(in_sr)
    names = next(reader)[1:]
    matrix = []
    labels = []
    for values in reader:
        instance = {}
        labels.append(values[0])
        for name, value in zip(names, values[1:]):
            if name == 'Location':
                feat_values = value.split()
                distance = feat_values[0]
                direction = feat_values[1]
                instance['Location (distance km)'] = float(distance.replace('km', ''))
                instance['Location (direction)'] = direction
                continue
            if name in categorical_names:
                instance[name] = None if value == 'n/a' else value
            else:
                if '<' in value:
                    value = '1'
                value = value.replace(',', '')
                instance[name] = None if value == 'n/a' else float(value)
        matrix.append(instance)

    dict_vectorizer = DictVectorizer()
    X = dict_vectorizer.fit_transform(matrix).toarray()
    names = dict_vectorizer.feature_names_
    writer = csv.writer(out_sr)
    writer.writerow(names)
    writer.writerows(X)
    categorical_names = list(name for name in names if '=' in name)
    numeric_names = list(name for name in names if '=' not in name)
    pickle.dump((labels, names, categorical_names, numeric_names, X), pickle_sr)


def main():
    with open('data/input.csv') as in_sr, open('data/binarized.csv', 'w') as out_sr, \
            open('data/binarized.pickle', 'wb') as pickle_sr:
        conv(in_sr, out_sr, pickle_sr)

if __name__ == '__main__':
    main()
