# Big Data Analytics using Machine Learning
Rahul Thayyil Sivan

we are going to use the topmost two machine learning
libraries and perform predictive analysis using various models
embedded in them. Youtube data-set is used to explore what
makes videos popular on various platforms. We are implementing
two models from Scikit-learn which are,
- Bayesian Ridge model
- Gradient Boosting Regressor model

Also two models from MLlib which are, 
- Gradient-Boosted Trees Regressor model
- One Vs Rest model  

we are going explore the YouTube dataset
which contains details of videos and their features like
categories, place of origin, view count, like count, comment,
description, etc. Our goal is to implement and evaluate models
to predict views that indicate the popularity of video.

## Data Loading and Preprocessing

In this project data loading and prepossessing phase is
implemented on pandas data-frame. Exploratory data analysis
and feature engineering are performed over the data-set and
plotted using the matLib library. Some features which have
a high range are re-scaled using the log function to avoid
numerical instability problems. The re-scaled view log is
generated and set as a label column which in turn is used for
prediction. The resulting pandas data-frame is used for data
analytics.

## Training and Modelling 

In the Modelling phase, the panda’s data frame is
split into train and test data-frame using train test split
method in the sklearn library. Principal component analysis
(PCA) is implemented to reduce the dimension of data-frame
by keeping relevant features. The prepossessed data-frame
consisted of 38 columns which can be reduced to 34 columns
by implementing PCA. Our methodology is split into four subsections
based on the two libraries used for ML.
