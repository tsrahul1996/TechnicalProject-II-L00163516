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

## BayesianRidge model

The model is imported from sklearn module using
sklearn.linear model.BeyesianRidge commands
and is trained with the train data-frame. It is then evaluated
using predict method, the result is used to deriver the
square value and mean squared error. We got a 0.86 r square value which is showing a high level of correlation with
prediction and label thus indicating our model is good. 0.44
mse value for both test and train was observed. The model is
then trained with a train data-frame after PCA transformation.
After evaluation, the r square value is reduced to 0.85 and mse
value to 0.50 for both the test and train data-frame which is
a small trade-off between information loss and dimensionality
reduction. The model is subjected to hyperparameter tuning using
GridSearchCV method from sklearn library. alpha init
and lambda init parameters are tuned and best model is
derived with ’alpha init’: 8.0, ’lambda init’:
0.1. The new model is evaluated and the r square value
is still 0.85 and mse value with 0.50 for train and test
data-sets, there is a slight decrease in mse value which was
0.505963871152161 to 0.5059638711521997 which indicates
the performance of the model increased considerably small
proportion after hyperparameter tuning.

### Gradient Boosting Regressor model

Gradient Boosting constructs an incremental model in a
stage-wise manner; it is capable of optimizing any differentiable
loss functions. Every stage involves fitting a regression
tree to the provided loss function’s negative gradient
[2]. It is an ensemble method and is imported
from sklearn.ensemble. The model is hyper tuned with
GridSearchCV and PCA transformed data as train data.
The parameters learning rate=0.03, max depth=2,
n estimators=700,subsample=0.6 found to be the
best estimator with the r square value of 0.85 , mse value
of 0.47 and 0.45 for test and train data respectively.


