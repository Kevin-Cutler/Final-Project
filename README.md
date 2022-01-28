# Predicting IMDB Movie Scores

![IMDB_Logo_2016](https://user-images.githubusercontent.com/88118759/150700799-37b89265-d9af-44fa-9e21-e0dc1b8e6544.svg)



## Overview
_____________________________
The goal of this project is to take two different movie datasets in order to develop a machine learning model that can predict whether or not it has a Favorable or Unfavorable movie score.  Both of the datasets were taken from kaggle.com.  The dataset top_4000_movies has data from 4000 different movies for features such as production budget and gross revenue for each movie.  The other dataset called movies also found from kaggle.com, has 7512 values for movie title and contains other variables such as rating, genre, year, votes, and cast information.  The movies dataset contains the IMDb score that was used as the outcome data and was broken up into scores above 6 as favorable, and below 6 as unfavorable.  The two datasets were merged on the same movie title to use more features in our machine learning model.  A random forest classifier model was used to predict the primary outcome of favorable vs unfavorable IMDb score as well as determine the most important features in the dataset. For this project our team met using slack and discussed topics in zoom meetings to collaborate.


Our goal in our analysis is to predict the IMDB user score by determining if a movie recieved a favorable or unfavorable IMDB score. This type of model is  known as a classification algorithm where the model learns patterns from the data, which will allow the model to predict if based on the independent variables we provide if a IMDB movie score should be "Favorable" or "Unfavorable" for each piece of data. We bucketmovie scores to be set as Unfavorable if a movie recieved a score from (1 to less than 6) and a Favorable score for 6-10. The movie score rankings in our kaggle dataset is between 1-10. For our project we have selected 2 movie datasets from kaggle to be the source of our research and analysis. One of our datasets movies.csv captures columns that include IMDB score, IMDB user votes, genre, year released, writer, directer, star, country, budget, gross, and runtime as well as the genre rating. We combine this dataset with an additional csv from kaggle which is complied of the top 4000 movies ranked by production budget and containing release date, domestic gross revenue and worldwide revenue. We selected this dataset because we want  to settle the dispute around which variables play a role in positively impacting a IMDB score. 

For our machine learning model we decided to use a random forest model. These type of supervised machine learning models are great for clustering points of data into functional groups. We are using over 2000 rows of data and random forest models are ideal against overfitting each of those weak learners and training the different pieces of the data.
A big reason we decided to use supervised random forest model and determine the most important variable in predicting IMBD movie scores is that variables can be ranked by importance of input in a natural way. The model can handle thousands of input variables without variable deletion. The model works well on large datasets and is great with handling outliers and nonlinear data.


# Data Exploration
_________________________________________________________

Once we loaded our datasets into our notebook next step is to determine what data we must transform, we leverage pandas library for data cleaning. Our dataset needs to be transformed before being fed into our machine learning model. We will be merging the two datasets on our movie name columns but we have to make sure that there are no duplicates which will cause errors when joining the two into one dataframe.

![ERD_quick_db](https://user-images.githubusercontent.com/88444529/151272050-784114b2-8acb-49cb-a745-fe19f0243f7d.PNG)

To predict IMDB scores we will be using a random forest model and we need a dataset with enough rows and suitable data to provide our model with enough information to predict our IMDB scores. After reading in our datasets we checked to determine how man duplicates were in each csv. Our movies.csv had 156 duplicates and the top 4000 movies had 48 duplicate rows. First we removed the movie name duplicates which allowed us to merge the two into one dataset. Now that we have removed duplicate movie names we merged our datasets into a single dataframe.

![image](https://user-images.githubusercontent.com/88467263/151076415-3e3134ab-2dc0-4f01-842d-b3439b8fdf33.png)

 Above you see that we have our single dataframe and now we need to determine which columns we need to remove because it wont provide us with useful information to determine IMDB movie scores. We chose to drop "name" column and keep "Movie Tile". We will only need one of these two columns. We decided to drop "Release Date" and keep "Year". Last we dropped the "Country" column we agreed that it did not play a role in the IMDB movie score. Next we will remove null values which leaves us with 2847 rows (See Below) which is a suitable amount of data for our machine learning model.   
 
 ![image](https://user-images.githubusercontent.com/88467263/151077295-b8d678b6-2d7a-4199-b362-417e4034f2ca.png)


Our goal is to determine IMDB movie scores and we decided to categorize our movie scores into two buckets since currently our datasets scores range from between (1-10). Instead of determining each movie score we want to know what factores predict "Favorable" and "Unfavorable" movie scores. For this step we created a function to transform movie scores 6 and greater to a string called "Favorable" and for scores less than 6 we will change those to "Unfavorable". This way when we start to Hot Encode our categorical data we get a 0 for "Favorable IMDB Scores" and 1 for "Unfavorable IMDB Scores".

![image](https://user-images.githubusercontent.com/88467263/151449957-51db6991-e884-4376-937c-d188a40d6c08.png)


Next we notice that there are columns in our dataframe that have values that are very large, these columns are "Production Budget", "domestic Gross", "Worldwide Gross", "Votes", "Budget", "Gross". Before we can feed our dataset into a machine learning model we will need to scale these values down from 9 digits to 3 to 4 digits which will help our machine learning model make a more accurate decision. 

![image](https://user-images.githubusercontent.com/88467263/151451025-3c92fc26-3a08-4823-9b5c-cedbcd0fa110.png)


Before we split or features and target variable the next step is to encode our categorical columns to numerical data to be fed into our model using sklearn label encoder library.
![image](https://user-images.githubusercontent.com/88467263/151451225-a825228e-6bce-481c-9648-727d829b59e6.png)

### With Sklearn Label Encoder our Dataframe is ready to be Split
_______________________________________________________________________

![image](https://user-images.githubusercontent.com/88467263/151451397-e82be90d-d063-4740-aff9-85762e0336b1.png)


# Preparing our Model
______________________________________________________

We start our preprocessing steps by dropping our target variable "Score" column and assigning it to "y" variable. We assign the features to the variable "X". Looking at our "X" variable using pandas describe() we have 2847 records for our model. Having a significant amount of data is key for our machine learning model  to allow it to make predictions based on patterns in the data. Using value_counts() on our "y" variable our target we see that there are 2065 "Favorable IMDB scores" and 782 "Unfavorable IMDB scores". We initate the next step which is to split the training and test sets. The model uses the training dataset to learn from it. It then uses the testing dataset to assess its performance. With machine learning the data must be split, training the model on the whole dataset will prevent the model from knowing how to  perform when it encounters new data. This makes it a pivotal step to set aside a portion of your dataset to evaluate your model.

![image](https://user-images.githubusercontent.com/88467263/151451774-9dc90e11-1d2d-45a5-949e-6804c167dc7c.png)

![image](https://user-images.githubusercontent.com/88467263/151451833-8574d623-3e16-48f5-84e3-6a027e9ca1a1.png)


* Description of preliminary feature engineering and preliminary feature selection, including the decision-making process


* Description of how data was split into training and testing sets


* Explanation of model choice, including limitations and benefits

 
##  Reminders : 
 * Do we know how our model tells the story with the Data?
 
 * What Kind of Message will the dashboard display?

* Link to Google slides: [here](https://docs.google.com/presentation/d/1K3iR-3VI6Z6oexiieo5eO8K5bZ4lB0vluZBnga12fdw/edit?usp=sharing)
