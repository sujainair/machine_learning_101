#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### -> Comments from original test
# -> My comments (includes redundant code in the final script)

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# I used a scatter plot below to check the data points and found some features had very few data points, so I did not use them
# Also removed email address
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'bonus', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'long_term_incentive', 'restricted_stock'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
# I used the scatter plot below to find certain people who are not poi but have outlier values.
del data_dict['TOTAL']
del data_dict['FREVERT MARK A']  # salary
del data_dict['LAVORATO JOHN J']  # bonus
del data_dict['MARTIN AMANDA K']  # long_term_incentive

# # Below is the scatter plot code I used to visualize the data points for poi and non-poi, especially to check for outliers.
# import matplotlib.pyplot as plt
# poi_values = [item['poi'] for item in data_dict.values()]
# for feat in features_list :
#     feat_values = [0.0 if item[feat] == 'NaN' else item[feat] for item in data_dict.values()]    
#     plt.scatter(poi_values, feat_values)
#     plt.ylabel(feat)
#     plt.show()
# # Used this to check the person with top values for features that stood out in the plot
# top_nos = 4
# myMax = [0]* top_nos
# myIndices = [-1]*top_nos
# for index, item in enumerate(feat_values):
#     if item > min(myMax):
#         minIndex = myMax.index(min(myMax))
#         myMax[minIndex] = item
#         myIndices[minIndex] = index
# for value in myIndices:
#     print data_dict.keys()[value]
#     print poi_values[value]
# print myMax

# # Used this to check the person with bottom values for features that stood out in the plot
# bot_nos = 5
# myMin = [max(feat_values)]* bot_nos
# myIndices = [-1]*bot_nos
# for index, item in enumerate(feat_values):
#     if item < max(myMin):
#         maxIndex = myMin.index(max(myMin))
#         myMin[maxIndex] = item
#         myIndices[maxIndex] = index
# for value in myIndices:
#     print data_dict.keys()[value]
#     print poi_values[value]
# print myMin

### Task 3: Create new feature(s)
to_msg = [0.0 if item['to_messages'] == 'NaN' else item['to_messages'] for item in data_dict.values()]
from_msg = [0.0 if item['from_messages'] == 'NaN' else item['from_messages'] for item in data_dict.values()]
to_poi = [0.0 if item['from_this_person_to_poi'] == 'NaN' else item['from_this_person_to_poi'] for item in data_dict.values()]
from_poi = [0.0 if item['from_poi_to_this_person'] == 'NaN' else item['from_poi_to_this_person'] for item in data_dict.values()]
shared_poi = [0.0 if item['shared_receipt_with_poi'] == 'NaN' else item['shared_receipt_with_poi'] for item in data_dict.values()]
# Since people send/receive different amounts of messages, the email data is more representative as ratios below
to_poi_ratio = [0.0 if float(b) == 0.0 else float(a) / float(b) for a,b in zip(to_poi, from_msg)]
from_poi_ratio = [0.0 if float(b) == 0.0 else float(a) / float(b) for a,b in zip(from_poi, to_msg)]
shared_poi_ratio = [0.0 if float(b) + float(c) == 0.0 else float(a) / (float(b) + float(c)) for a,b,c in zip(shared_poi, to_msg, from_msg)]
# Add these to the data_dict
index = 0
for key, value in data_dict.items():
    value['to_poi_ratio'] = to_poi_ratio[index]
    value['from_poi_ratio'] = from_poi_ratio[index]
    value['shared_poi_ratio'] = shared_poi_ratio[index]
    index = index + 1
    data_dict[key] = value
features_list.extend(['to_poi_ratio', 'from_poi_ratio', 'shared_poi_ratio'])
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = False)
labels, features = targetFeatureSplit(data)
from sklearn.feature_selection import SelectKBest, f_classif
sel = SelectKBest(f_classif, 10)
features_new = sel.fit_transform(features, labels)

## Having looked at the scores below decided to use 10 features
# print sel.scores_
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### Provided to give you a starting point. Try a variety of classifiers.
# NB seems to work best here.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

# from sklearn.tree import DecisionTreeClassifier
# clf = DecisionTreeClassifier()

# from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier(2)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

### Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=55)
clf.fit(features_train, labels_train)
print clf.score(features_test, labels_test) # 0.86 quite reasonable
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)