# Breast Cancer Classification

A short task of predicting breast cancer and finding out the perfect model for it using the [UCI Wisconsin Breast Cancer dataset]([https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data](https://www.kaggle.com/datasets/jeandedieunyandwi/breast-cancer-dataset-uci-ml)
<br>

### Technologies used:
- Jupyter Notebook
- Numpy
- Pandas
- Scikit Learn
- Matplotlib

### Dataset:
- `radius` (mean of distances from center to points on the perimeter)
- `texture` (standard deviation of gray-scale values)
- `perimeter`
- `area`
- `smoothness` (local variation in radius lengths)
- `compactness` (perimeter^2 / area - 1.0)
- `concavity` (severity of concave portions of the contour)
- `concave points` (number of concave portions of the contour)
- `symmetry` 
- `fractal dimension` ("coastline approximation" - 1)

There can be two results: `WDBC-Malignant`(1) and `WDBC-Benign`(0)

The features are all columns.

### Process:
At first we check the amount of missing values we have in our dataset. We then drop the columns that will have no significant affect in our predictions. If there are any missing labels, we drop that entire row.

We then split the data into features(x) and labels(y).

There are no missing data.

There are also no categorical data.

Next, we use feature scaling. We normalize our data. Later, we looked at what the scores would've been if we standardized it.

we spilt the data into training and test sets. I've considered the test size to be 20% of the total data.

We then use 6 different models on the data to find which one gives us the best case:
- `Linear Model` (Accuracy score: 96%)
- `Support Vector Machine` (Accuracy score: 97%)
- `Stohastic Gradient Descent` (Accuracy score: 96%)
- `Nearest Neighbours` (Accuracy score: 96%)
- `Guassian Processes` (Accuracy score: 95%)
- `Naive Bayes` (Accuracy score: 90%) **Worst Performing**
- `Decision Trees` (Accuracy score: 91%)
- `Random Forest` (Accuracy score: 98%) **Best performing**
- `Majority Voting` (Accuracy score: 97%)

![image](https://user-images.githubusercontent.com/68951276/175816946-de8e6511-5f19-40d0-aa18-bd98c6bab4c2.png)


### Conclusion:
The best model for this case, a binaryclassification problem, is the **Random Forest** (RandomForestClassifier), having an accuracy score of 98%. 

Random Forest is suitable for situations when we have a large dataset, and interpretability is not a major concern. It also provides very high accuracy. 

However, the main limitation of random forest is that a large number of trees can make the algorithm too slow and ineffective for real-time predictions. In general, these algorithms are fast to train, but quite slow to create predictions once they are trained.
