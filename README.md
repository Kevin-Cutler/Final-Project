# Predicting IMDB Movie Scores

![IMDB_Logo_2016](https://user-images.githubusercontent.com/88118759/150700799-37b89265-d9af-44fa-9e21-e0dc1b8e6544.svg)



## Overview
_____________________________
The goal of this project is to take two different movie datasets in order to develop a machine learning model that can predict whether or not it has a Favorable or Unfavorable movie score. Both of the datasets were taken from kaggle.com. The dataset top_4000_movies has data from 4000 different movies for features such as production budget and gross revenue for each movie. The other dataset called movies also found from kaggle.com, has 7512 values for movie title and contains other variables such as rating, genre, year, votes, and cast information. The movies dataset contains the IMDb score that was used as the outcome data and was broken up into scores above 6 as favorable, and below 6 as unfavorable. The two datasets were merged on the same movie title to use more features in our machine learning model. A random forest classifier model was used to predict the primary outcome of favorable vs unfavorable IMDb score as well as determine the most important features in the dataset. For this project our team met using slack and discussed topics in zoom meetings to collaborate.

The type of logistic regression modeling we will use to predict IMDB scores is known as a classification algorithm where the model learns patterns from the data, which will allow the model to predict if based on the independent variables we provide if a IMDB movie score should be "Favorable" or "Unfavorable" for each piece of data. We bucket movie scores to be set as Unfavorable if a movie recieved a score from (1 to less than 6) and a Favorable score for 6-10. The movie score rankings in our kaggle dataset is between 1-10. We selected this dataset because we want to settle the dispute around which variables play a role in positively impacting a IMDB score.

## Advantages in Random Forest Models:
____________________________________________
For our machine learning model we decided to use a random forest model. These type of supervised machine learning models are great for clustering points of data into functional groups. We are using over 2000 rows of data and random forest models are ideal against overfitting each of those weak learners and training the different pieces of the data.
A big reason we decided to use supervised random forest model and determine the most important variable in predicting IMBD movie scores is that variables can be ranked by importance of input in a natural way. The model can handle thousands of input variables without variable deletion. The model works well on large datasets and is great with handling outliers and nonlinear data.

## Disadvantages in Random Forest Models:
____________________________________________
Being that random forest uses multiple decision trees it is known to be slow at generating predictions. The time consuming process of decision making with random forest models is that trees are dependant on processing the same given input and then making predictions on the data.

# Database and Data Exploration
_________________________________________________________

Data was first cleaned in jupyternotebooks using Pandas.  The files were cleaned and prepared using the ERD made in QuickDBD and presented in the image below.
![ERD_quick_db](https://user-images.githubusercontent.com/88444529/151272050-784114b2-8acb-49cb-a745-fe19f0243f7d.PNG)

The movies.csv file was read into pandas and created into a dataframe.  Prior to any data cleaning there were 7363 rows in the movies_df dataframe.  The dataframe was inspected for duplicate entries and null values.  Each of the rows with duplicate entries for "movie_name" were dropped from the dataset.  There were 163 duplicate values that were dropped and 2162 rows that contained null values that were dropped.  The resulting movies_df dataframe had 5201 rows with unique values for "movie_name".
![image](https://user-images.githubusercontent.com/88444529/151614347-3a3f6d84-97db-41b2-b135-c8e6dcde32cb.png)

This same process was done for the top_4000_movies csv file from kaggle.  Upon reading in the dataframe there were 4000 rows.  The csv file was read in using Pandas and a dataframe was created top_4000_movies_df.  The dataframe contained 48 duplicate entries for "Movie Title" and 21 rows with null values.  Both the duplicate entries and rows with null values were dropped from the dataframe, and the resulting dataframe had 3883 rows with unique values for "Movie Title".
![image](https://user-images.githubusercontent.com/88444529/151615118-e86b789a-63c9-4118-8194-2a46f5629e6c.png)

SQLalchemy was used to connect our jupyter notebook to pgAdmin and our PostgreSQL data base.  A connection string and engine were created to connect to our database and export the cleaned dataframes top_4000_movies_df and movies_df to pgAdmin.  They were exported into our database named "movies_final" into tables called "top_4000_movies" and "movies" for the dataframes, movies and top_4000_movies respectively.
![image](https://user-images.githubusercontent.com/88444529/151619296-4f9d6685-a4a7-4f64-aa6a-a366ef040344.png)

![image](https://user-images.githubusercontent.com/88444529/151619380-6063ed47-8505-4e83-8160-0f4fefe6ff32.png)

![image](https://user-images.githubusercontent.com/88444529/151619471-2de79bdf-b5d2-4904-bf49-7b8f7045fb34.png)

Upon exporting the dataframes into the "movies_final" database a join was done on the movies and top_4000_movies tables on the matching primary keys of each table, "Movie_Title" from top_4000_movies and "movie_name" from movies.  The resulting joined table from our database was read back into our jupyter notebook for analysis using SQLalchemy again.
![image](https://user-images.githubusercontent.com/88444529/151620127-6b248c38-9b71-4234-aa12-8c9e5bc7e0fd.png)
![image](https://user-images.githubusercontent.com/88444529/151620369-17e7bb11-cccb-4026-831b-a12c3bd2036e.png)

 Above you see that we have our single dataframe and now we need to determine which columns we need to remove because it wont provide us with useful information to determine IMDB movie scores. We chose to drop "name" column and keep "Movie Tile". We will only need one of these two columns. We decided to drop "Release Date" and keep "Year". Last we dropped the "Country" column we agreed that it did not play a role in the IMDB movie score. Next we will remove null values which leaves us with 2847 rows (See Below) which is a suitable amount of data for our machine learning model.   
 
 ![image](https://user-images.githubusercontent.com/88467263/151077295-b8d678b6-2d7a-4199-b362-417e4034f2ca.png)


Our goal is to determine IMDB movie scores and we decided to categorize our movie scores into two buckets since currently our datasets scores range from between (1-10). Instead of determining each movie score we want to know what factores predict "Favorable" and "Unfavorable" movie scores. For this step we created a function to transform movie scores 6 and greater to a string called "Favorable" and for scores less than 6 we will change those to "Unfavorable". This way when we start to Hot Encode our categorical data we get a 0 for "Favorable IMDB Scores" and 1 for "Unfavorable IMDB Scores".

![image](https://user-images.githubusercontent.com/88467263/151449957-51db6991-e884-4376-937c-d188a40d6c08.png)


Next we notice that there are columns in our dataframe that have values that are very large, these columns are "Production Budget", "domestic Gross", "Worldwide Gross", "Votes", "Budget", "Gross". Before we can feed our dataset into a machine learning model we will need to scale these values down from 9 digits to 3 to 4 digits which will help our machine learning model make a more accurate decision. 

![image](https://user-images.githubusercontent.com/88467263/151451025-3c92fc26-3a08-4823-9b5c-cedbcd0fa110.png)


Before we split or features and target variable the next step is to encode our categorical columns to numerical data to be fed into our model using sklearn label encoder library.
![image](https://user-images.githubusercontent.com/88467263/151451225-a825228e-6bce-481c-9648-727d829b59e6.png)

### Skilearn Label Encoder on our Dataframe
_______________________________________________________________________

![image](https://user-images.githubusercontent.com/88467263/151451397-e82be90d-d063-4740-aff9-85762e0336b1.png)


# Preparing our Model
______________________________________________________

We start our preprocessing steps by dropping our target variable "Score" column and assigning it to "y" variable. We assign the features to the variable "X". Looking at our "X" variable using pandas describe() we have 2847 records for our model. Having a significant amount of data is key for our machine learning model  to allow it to make predictions based on patterns in the data. Using value_counts() on our "y" variable our target we see that there are 2065 "Favorable IMDB scores" and 782 "Unfavorable IMDB scores". We initate the next step which is to split the training and test sets. The model uses the training dataset to learn from it. It then uses the testing dataset to assess its performance. With machine learning the data must be split, training the model on the whole dataset will prevent the model from knowing how to  perform when it encounters new data. This makes it a pivotal step to set aside a portion of your dataset to evaluate your model.

![image](https://user-images.githubusercontent.com/88467263/151451774-9dc90e11-1d2d-45a5-949e-6804c167dc7c.png)

# We split into the training and testing sets.

![image](https://user-images.githubusercontent.com/88467263/151451833-8574d623-3e16-48f5-84e3-6a027e9ca1a1.png)


# We create the StandardScaler instance, Fit the scaler with the training set, and scale the data.
![image](https://user-images.githubusercontent.com/88467263/151669220-8cbe1330-e92f-41c3-9424-61953210bcda.png)


We use the random forest classifier - RandomForestClassifier() prior to fitting the random forest model with our X_train_scaled and y_train training data. The  RandomForestClassifier() takes in multiple parameters and for our model we will use n_estimators = 200 and random_state = 78. The n_estimators is a key step because it will set the number of decision trees for the alogorithm to create. If we use a large amount of n_estimators it has the potential to slow down the model due to more training required with the higher number. After testing multipe values we determined that 200 was an effective amount for our dataset. After we create the random forest instance, we need to fit the model with our training sets.

![image](https://user-images.githubusercontent.com/88467263/151669469-9c018611-500b-4b7e-b80e-3199b9466d40.png)




# After fitting the model, we make our predictions using the scaled testing data.

 ![image](https://user-images.githubusercontent.com/88467263/151669837-929a079b-5801-4542-a504-f92c3852896a.png)
 
# Evaluate the Model Confusion Matrix Results
 
Looking at the results of our confusion matrix below, we were able to obtain a precision of 85% for predicting Favorable IMDB movie scores that are actually Favorable in the   dataset provided to the model. For predicting movie scores as being Unfavorable for movies that indeed recieved a actual Unfavorable score our model obtained a 80%. Precision is the measure of how reliable a positive classification is. So at accurately classifying IMDB movies scores at being part of actually Favorable and actually Unfavorable scores groups this model performs good when looking at precision. This indicates a low number of false positives for incorrectly classifying IMDB scores which signifies a positive classification model. The recall for classifying Actual Favorable IMDB movie scores is 95% and for classifying Actual Unfavorable IMDB movie scores is 55% which is not the most ideal but 5% higher for making a accurate predicition. Recall is the ability of the classifier to find all the positive samples. The recall is low for predicting Actual Unfavorable IMDB movie scores, which is determined by the TN(109)/(TN(109)+FP(88)) = .55. The F1 score is also low (65%) but not terrible. The f1 score for predicting Actual  Favorable IMDB movie scores is 89% which is also good. F1 score is a weighted average of the true positive rate (recall) and precision, where the best score is 1.0
and the worst is 0.0. Our F1 score is 89.72%.
 


In summary, this random forest model is good at classifying Actual  Favorable IMDB movie scores because the model's accuracy, 84%, and F1 scores are good.
 
![image](https://user-images.githubusercontent.com/88467263/151677635-e12cf6b2-134b-4931-b0fe-0669d28bf513.png)


# Summary of Performance: Confusion Matrix

*  Out of 515 movies that obtained a "Favorable IMDB Score" (Actual 0), 488
   were predicted to have a "Favorable IMDB Score" (Predicted 0),
   which is known as true positives. 
   
*  Out of 515 movies that obtained a "Favorable IMDB Score" (Actual 0),
   27 were predicted to have a "Unfavorable IMDB Score" (Predicted 1),
   which are considered false negatives.
   
*  Out of 197 movies that obtained a "Unfavorable IMDB Score" (Actual 1),
   88 were predicted to have a "Favorable IMDB Score" (Predicted 0)
   and are considered false positives.
   
*  Out of 197 movies that obtained a "Unfavorable IMDB Score" (Actual 1),
   109 were predicted to be have a "Unfavorable IMDB Score" (Predicted 1)
   and are considered true negatives.
   
## Heatmap to Visualize Performance

![image](https://user-images.githubusercontent.com/88467263/151703957-aa76017f-362c-40f2-8039-7c8ace61bde7.png)

# Key Takeaways Model Performance
______________________________________

* The accuracy of our model is 0.84% , which can also be calculated as follows:
  True Positives (TP) + True Negatives (TN)) / Total = 0.84%.
  
* Precision: Precision is the measure of how reliable a positive classification 
  is. From our results, the precision for the good IMDB movie scores can be
  determined  by the ratio TP/(TP + FP), which is 0.85 . A low precision is indicative 
  of a large number of false positives—of the 136 movie IMDB scores we predicted to obtain 
  bad scores, 27 actually recieved a good IMDB movie scores.
  
* Recall: Recall is the ability of the classifier to find all
  the positive samples. It can be determined by the ratio: TP/(TP + FN),
  or 0.95 for the good IMDB movie scores and 0.55 for the bad IMDB movie scores. A low
  recall is indicative of a large number of false negatives.
  
* F1 score: F1 score is a weighted average of the true positive
  rate (recall) and precision, where the best score is 1.0
  and the worst is 0.0. Our F1 score is 89.72%.
  
* Support: Support is the number of actual occurrences of the class in the specified dataset.
  For our results, there are 515 actual occurrences for the good IMDB movie scores and 197
  actual occurrences for bad IMDB movie scores.


![image](https://user-images.githubusercontent.com/88467263/151670774-82c43436-d474-41e3-a2ea-d828ce674200.png)

# Display Feature Importance using a Bar Graph Below

![image](https://user-images.githubusercontent.com/88467263/151670866-0a461a19-5186-495d-9ab0-2e5952765dc2.png)

# Votes vs Score Scatter Plot

* In our plot we can show correlation between Votes vs Score.

![image](https://user-images.githubusercontent.com/88467263/151676700-a7bf9ab5-2d94-45a7-a7e2-59cebd1c2df8.png)

__________________________________________________

![image](https://user-images.githubusercontent.com/88467263/151676710-9fdf45c7-849a-4684-a481-6cee2b207e9d.png)


# Runtime vs Score Scatter Plot

* In our plot we can show correlation between Runtime vs Score.

![image](https://user-images.githubusercontent.com/88467263/151676739-9271d6d2-878b-42a6-8cb9-72db5003bad0.png)

__________________________________________________________________

![image](https://user-images.githubusercontent.com/88467263/151676754-c02feefd-cf09-48ea-ab17-d0cc3720c228.png)


# Domestic Gross vs Score Scatter Plot

* In our plot we can show correlation between Domestic Gross vs Score.

![image](https://user-images.githubusercontent.com/88467263/151676833-b06bfb72-907e-4be6-bd7f-c4db01a9988c.png)

____________________________________________________________________

![image](https://user-images.githubusercontent.com/88467263/151676844-50b4df05-b53e-4751-acbd-17f847220324.png)

# Correlation Matrix for Movie Features
__________________________________________

* With a correlation matrix we continue to identify  the highest correlation to IMDB Score being 'Votes' and Runtime' which align with our results.

![image](https://user-images.githubusercontent.com/88467263/151677458-7199e948-9ba5-430d-a231-a1c2025272dd.png)
![image](https://user-images.githubusercontent.com/88467263/151677474-b35d9ff9-5dd8-4c81-b015-ef44b0f81b63.png)


## We found that the Random Forest Model has the highest probability for predicting IMDB Movie Scores
____________________________________________________________

![image](https://user-images.githubusercontent.com/88467263/151680039-d97b8c3e-99e9-4fab-ae08-bb1be22e6115.png)

# Summary:

Our choice to use a ranodom forest model was proven to be a good solution in classifying favorable IMDB movie scores vs classifying unfavorable IMDB movie scores. The good things about random forest models is that it analyzes the data by building several smaller, simpler decision trees and combining decision trees to create a strong learner, which has much better decision-making power. The randomly sampled data is a weaker learner when created but as the model combines those weaker learners the model can become well trained and effective at making accurate predictions. Our model was able to obtain a accuracy of 84% at identifying IMDB movie scores and classifying them. We were able to answer the question around which variables to focus on when determining IMDB movie scores. Based on the experiment we show that "Votes" has the most impact on IMDB scores, followed by "Runtime" and Domestic Gross". So to our investors who want to know which areas to invest research in we would share those variables as the most compelling catalyst for obtaining a Favorable or Unfavorable IMDB movie score.

* Link to Google slides: [here](https://docs.google.com/presentation/d/1K3iR-3VI6Z6oexiieo5eO8K5bZ4lB0vluZBnga12fdw/edit?usp=sharing)
