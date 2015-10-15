
import csv
import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import MDS, Isomap
from sklearn.preprocessing import normalize


def add_labels(labels, points, ax):
    for label, x, y in zip(range(1, len(labels) + 1), points[:, 0], points[:, 1]):
        ax.annotate(
            label, xy=(x, y), xytext=(10, 10),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            size='xx-small',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
        )


def plot_location(labels, ax):
    with open('data/coordinates.csv') as sr:
        reader = csv.DictReader(sr)
        all_locations = dict((row['Community Name'], (float(row['East']), -float(row['South']))) for row in reader)
        points = np.vstack(all_locations[label] for label in labels)
        ax.scatter(points[:, 0], points[:, 1], s=20, c='b')
        ax.set_title('Location')
        add_labels(labels, points, ax)


def mult_scl(X, labels):
    print('labels:')
    for i, label in zip(range(1, len(labels) + 1), labels):
        print('{}: {}'.format(i, label))

    isomap = Isomap()
    points = isomap.fit(np.nan_to_num(X)).embedding_
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    plot_location(labels, ax3)
    ax1.scatter(points[:, 0], points[:, 1], s=20, c='r')
    ax1.set_title('Isomap')
    add_labels(labels, points, ax1)

    mds = MDS()
    points = mds.fit(np.nan_to_num(X)).embedding_
    ax2.scatter(points[:, 0], points[:, 1], s=20, c='g')
    ax2.set_title('MDS')
    add_labels(labels, points, ax2)

    plt.show()


def normalize_instances(X, names, numeric_names):
    numeric_rows = set(i for i, name in enumerate(names) if any(name == numeric_name for numeric_name in numeric_names))
    Xp = []
    for i, row in enumerate(X.T):
        if i in numeric_rows:
            Xp.append(normalize(np.nan_to_num(row))[0])
        else:
            Xp.append(np.nan_to_num(0.2*row))
    Xp = np.array(Xp)
    return Xp.T


def main():
    # XXX we should allow each feature to have it's own metric
    # region group_1
    group_1 = [
        # geography
        # 'Map reference',
        # 'Grid reference',
        # 'Location',
        'Population Density',
        'Travel time to GPO (minutes)',
        'Distance to GPO (km)',
        'LGA',
        'Primary Care Partnership',
        'Medicare Local	Area (km^2)',
        'DHS Area',
        # land use
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
    ]
    # endregion
    # region group_2
    group_2 = [
        # population
        '2012 ERP age 0-4, persons',
        '2012 ERP age 0-4, %',
        '2012 ERP age 5-9, persons',
        '2012 ERP age 5-9, %',
        '2012 ERP age 10-14, persons',
        '2012 ERP age 10-14, %',
        '2012 ERP age 15-19, persons',
        '2012 ERP age 15-19, %',
        '2012 ERP age 20-24, persons',
        '2012 ERP age 20-24, %',
        '2012 ERP age 25-44, persons',
        '2012 ERP age 25-44, %',
        '2012 ERP age 45-64, persons',
        '2012 ERP age 45-64, %',
        '2012 ERP age 65-69, persons',
        '2012 ERP age 65-69, %',
        '2012 ERP age 70-74, persons',
        '2012 ERP age 70-74, %',
        '2012 ERP age 75-79, persons',
        '2012 ERP age 75-79, %',
        '2012 ERP age 80-84, persons',
        '2012 ERP age 80-84, %',
        '2012 ERP age 85+, persons',
        '2012 ERP age 85+, %',
        '2012 ERP, total',
        '2007 ERP age 0-4, persons',
        '2007 ERP age 0-4, %',
        '2007 ERP age 5-9, persons',
        '2007 ERP age 5-9, %',
        '2007 ERP age 10-14, persons',
        '2007 ERP age 10-14, %',
        '2007 ERP age 15-19, persons',
        '2007 ERP age 15-19, %',
        '2007 ERP age 20-24, persons',
        '2007 ERP age 20-24, %',
        '2007 ERP age 25-44, persons',
        '2007 ERP age 25-44, %',
        '2007 ERP age 45-64, persons',
        '2007 ERP age 45-64, %',
        '2007 ERP age 65-69, persons',
        '2007 ERP age 65-69, %',
        '2007 ERP age 70-74, persons',
        '2007 ERP age 70-74, %',
        '2007 ERP age 75-79, persons',
        '2007 ERP age 75-79, %',
        '2007 ERP age 80-84, persons',
        '2007 ERP age 80-84, %',
        '2007 ERP age 85+, persons',
        '2007 ERP age 85+, %',
        '2007 ERP, total',
        '% change, 2007-2012, age 0-4',
        '% change, 2007-2012, age 5-9',
        '% change, 2007-2012, age 10-14',
        '% change, 2007-2012, age 15-19',
        '% change, 2007-2012, age 20-24',
        '% change, 2007-2012, age 25-44',
        '% change, 2007-2012, age 45-64',
        '% change, 2007-2012, age 65-69',
        '% change, 2007-2012, age 70-74',
        '% change, 2007-2012, age 75-79',
        '% change, 2007-2012, age 80-84',
        '% change, 2007-2012, age 85+',
        '% change, 2007-2012, total',
        # social-demographic
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
        '3rd top occupation, %',
        # diversity
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
        '5th top language spoken, %',
    ]
    # endregion
    # region group_3
    group_3 = [
        # services
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
        'Medicare Access Points',
        # hospital
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
        'Category 4 & 5 emergency department presentations, %',
    ]
    # endregion
    with open('data/binarized.pickle', 'rb') as pickle_sr:
        labels, names, categorical_names, numeric_names, X = pickle.load(pickle_sr)
        group = group_1
        rows = set(i for i, name in enumerate(names) if any(name.startswith(group_name) for group_name in group))
        X = normalize_instances(X, names, numeric_names)[:, sorted(rows)]
        mult_scl(X, labels)

if __name__ == '__main__':
    main()
