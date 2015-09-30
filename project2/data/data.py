
import csv
import pickle

from sklearn.feature_extraction import DictVectorizer


def conv(in_sr, out_sr, pickle_sr):
    '''
    This method converts a csv file containing all the features of every suburb
    into a scikit-learn readable data structures. First, we turn the data into
    dictionaries (e.g. {'Community Name': 'Ascot Vale (Suburb)', ... ,'Pharmacies': 4, ...}).
    Then we convert it using scikit-learn's DictVectorizer.

    Note that:
        - Feature 'Location' is split into two features:
            1. Location (distance km) - records the distance in km
            2. Location (direction) - records the direction (e.g. NW)
        - Some continuous values have the value '<5', this was turned into 1
        - 'n/a' values were turned into None
    '''
    categorical_feats = [
        'Region',
        'Map reference',
        'Grid reference',
        'Location (Direction)',
        'LGA',
        'Primary Care Partnership',
        'Medicare Local',
        'DHS Area',
        'Top industry',
        '2nd top industry - persons',
        '3rd top industry - persons',
        'Top occupation',
        '2nd top occupation - persons',
        '3rd top occupation - persons',
        'Top country of birth',
        '2nd top country of birth',
        '3rd top country of birth',
        '4th top country of birth',
        '5th top country of birth',
        'Top language spoken',
        '2nd top language spoken',
        '3rd top language spoken',
        '4th top language spoken',
        '5th top language spoken',
        'Nearest Public Hospital',
        'Nearest public hospital with maternity services',
        'Nearest public hospital with emergency department',
    ]
    reader = csv.reader(in_sr)
    feature_names = next(reader)
    matrix = []
    for line in reader:
        instance = {}
        for i, feat_value in enumerate(line):
            if feature_names[i] == 'Location':
                feat_values = feat_value.split()
                distance = feat_values[0]
                direction = feat_values[1]
                instance['Location (distance km)'] = float(distance.replace('km', ''))
                instance['Location (direction)'] = direction
                continue
            if feature_names[i] in categorical_feats or i == 0:
                if feat_value == 'n/a':
                    instance[feature_names[i]] = None
                else:
                    instance[feature_names[i]] = feat_value
            else:
                if '<' in feat_value:
                    feat_value = '1'
                feat_value = feat_value.replace(',', '')
                if feat_value == 'n/a':
                    instance[feature_names[i]] = None
                else:
                    instance[feature_names[i]] = float(feat_value)
        matrix.append(instance)
    dict_vectorizer = DictVectorizer()
    X = dict_vectorizer.fit_transform(matrix).toarray()
    writer = csv.writer(out_sr)
    writer.writerow(dict_vectorizer.get_feature_names())
    writer.writerows(X)
    pickle.dump((dict_vectorizer.get_feature_names(), categorical_feats, X), pickle_sr)


def main():
    with open('data/input.csv') as in_sr, open('data/binarized.csv', 'w') as out_sr, \
            open('data/binarized.pickle', 'wb') as pickle_sr:
        conv(in_sr, out_sr, pickle_sr)

if __name__ == '__main__':
    main()
