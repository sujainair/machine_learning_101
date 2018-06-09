# Learning Machine Learning
In my endeavour of staying entertained while feeling productive, I did [Intro to Machine Learning on Udacity](https://classroom.udacity.com/courses/ud120). At the end, like any good course, I had to complete this project.
## Project Details
As this is a free course now, you can easily check out the project details [here](https://classroom.udacity.com/courses/ud120/lessons/3335698626/concepts/33160186280923). In short, the purpose is to summarize all that was taught in the course with a simple project. The actual script I edited is `final_project/poi_id.py`, everything else was provided as a wrapper by Udacity. You can also find their course repo on [github](https://github.com/udacity/ud120-projects).
## What I did
If you read the comments in the script, it should be easy to understand my take on the problem. Admittedly, I did not spend a lot of time to understand the data subjectively and focused mostly on the code, recalling things from the course and as required by the project, achieving better than .3 precision and recall.  
1. Using a scatter plot I looked at the data to find features that had mostly unknowns ('NaN') and find outliers in some features.  
2. In case of emails, I preferred to use ratios to total emails of the person rather than absolute numbers.  
3. I tired a few classifiers, some I left as comments in the code but NB happened to have the best result.  
## Run the script
Make sure your working directory is `.` relative to the scripts you run here.  
Run the script `final_project/poi_id.py` to generate a few .pkl files and the script also spits out the score on the test data (0.86).  
Run the script `final_project/test.py` to get an analysis of Accuracy, Prediction, Recall, etc. for the classifier. The output for NB is as below
```
GaussianNB(priors=None)
	Accuracy: 0.85967	Precision: 0.46981	Recall: 0.40850	F1: 0.43702	F2: 0.41945
	Total predictions: 15000	True positives:  817	False positives:  922	False negatives: 1183	True negatives: 12078
```