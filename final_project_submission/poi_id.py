#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers

### remove the total record from dictionary
data_dict.pop("TOTAL", 0)

### remove an entry which is not related to POI
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)

### remove this data point as it's feature contains 0.0 or NaN
data_dict.pop("LOCKHART EUGENE E", 0)

features_list =  ['poi', 'exercised_stock_options', 'other', 'expenses', 'fraction_to_poi', 'total_stock_value', 'bonus', 'salary', 'deferred_income', 'long_term_incentive', 'fraction_from_poi']

### Task 3: Create new feature(s)

def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator)
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
    """
    if poi_messages == 'NaN':
        fraction = 0.
    elif all_messages == 'NaN':
        fraction = 0.
    else:
        fraction = float(poi_messages)/float(all_messages)
        #print "All Messages ", all_messages
        #fraction = 0.


    return fraction

remove_person = []

for name in data_dict:

    data_point = data_dict[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    data_point["fraction_from_poi"] = fraction_from_poi


    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    data_point["fraction_to_poi"] = fraction_to_poi

    data_dict[name] = data_point

print "Total data points", len(data_dict)


my_dataset = data_dict

### Check individual data point for NaN or 0.0
# for person in my_dataset:
#     if person == 'LOCKHART EUGENE E':
#         print person
#         for feature in features_list:

#             print feature, my_dataset[person][feature]

#         print
#         print

### Store to my_dataset for easy export below.
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()    # Provided to give you a starting point. Try a varity of classifiers.

## first classifier
# from sklearn.tree import DecisionTreeClassifier
# clf = DecisionTreeClassifier()
# clf = clf.fit(features, labels)
# count = 1
# for item in clf.feature_importances_:
#     print features_list[count], item
#     count+=1

## second classifier
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(bootstrap=True, compute_importances=None,
#             criterion='gini', max_depth=None, max_features=None,
#             max_leaf_nodes=None, min_density=None, min_samples_leaf=1,
#             min_samples_split=10, n_estimators=1, n_jobs=1,
#             oob_score=False, random_state=None, verbose=0)
# clf = clf.fit(features, labels)
# count = 1
# for item in clf.feature_importances_:
#     print features_list[count], item
#     count+=1

## third classifier
from sklearn.ensemble import ExtraTreesClassifier
# clf = ExtraTreesClassifier()
clf = ExtraTreesClassifier(bootstrap=False, compute_importances=None,
           criterion='gini', max_depth=15, max_features=None,
           max_leaf_nodes=None, min_density=None, min_samples_leaf=1,
           min_samples_split=1, n_estimators=1, n_jobs=1, oob_score=False,
           random_state=None, verbose=0)
clf = clf.fit(features, labels)
count = 1
print
print "Features Importance"
print "===================="
for item in clf.feature_importances_:
    print features_list[count], item
    count+=1

print

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# from sklearn.grid_search import GridSearchCV
# from sklearn.cross_validation import StratifiedShuffleSplit
# parameters = {'n_estimators':[1,5,10,15,20],'criterion':['gini','entropy'],'max_features':['sqrt','log2',None], 'min_samples_split':[1,2,4,6,8,10],'max_depth': [None, 4, 10, 15]}

# cross_validation = StratifiedShuffleSplit(labels, 100, random_state = 42)
# grid_search = GridSearchCV(clf, parameters, cv=cross_validation, scoring = 'recall')
# grid_search.fit(features,labels)

# print('Best score: {}'.format(grid_search.best_score_))
# print('Best parameters: {}'.format(grid_search.best_params_))
# print grid_search.best_estimator_

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)