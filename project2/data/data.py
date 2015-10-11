
import csv
import pickle
import numpy as np
from collections import OrderedDict

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
        'Community Name',
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

    geo_names = [
        "Location",
        "Population Density",
        "Travel time to GPO (minutes)",
        "Distance to GPO (km)",
        "LGA",
        "Primary Care Partnership",
        "Medicare Local	Area (km^2)",
        "DHS Area"
    ]

    land_use_names = [
        "Commercial (km^2)",
        "Commercial (%)",
        "Industrial (km^2)",
        "Industrial (%)",
        "Residential (km^2)",
        "Residential (%)",
        "Rural (km^2)",
        "Rural (%)",
        "Other (km^2)",
        "Other (%)"
    ]

    reader = csv.reader(in_sr)
    feature_names = next(reader)[1:]
    matrix = []
    labels = []
    for values in reader:
        instance = {}
        labels.append(values[0])
        for name, value in zip(feature_names, values[1:]):
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
    writer = csv.writer(out_sr)
    writer.writerow(dict_vectorizer.feature_names_)
    writer.writerows(X)
    pickle.dump((
        dict_vectorizer.feature_names_,
        OrderedDict((i, name) for i, name in enumerate(dict_vectorizer.feature_names_) if '=' in name),
        OrderedDict((i, name) for i, name in enumerate(dict_vectorizer.feature_names_) if '=' not in name),
        OrderedDict((i, name) for i, name in enumerate(dict_vectorizer.feature_names_) if any(l in name for l in geo_names)),
        OrderedDict((i, name) for i, name in enumerate(dict_vectorizer.feature_names_) if any(l in name for l in land_use_names)),
        labels,
        X
    ), pickle_sr)


def main():
    with open('data/input.csv') as in_sr, open('data/binarized.csv', 'w') as out_sr, \
            open('data/binarized.pickle', 'wb') as pickle_sr:
        conv(in_sr, out_sr, pickle_sr)

if __name__ == '__main__':
    main()
