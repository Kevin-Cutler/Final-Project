# Predicting IMDB Movie Scores

![IMDB_Logo_2016](https://user-images.githubusercontent.com/88118759/150700799-37b89265-d9af-44fa-9e21-e0dc1b8e6544.svg)



## Overview
_____________________________
The goal of this project is to take two different movie datasets in order to develop a machine learning model that can predict whether or not it has a Favorable or Unfavorable movie score.  Both of the datasets were taken from kaggle.com.  The dataset top_4000_movies has data from 4000 different movies for features such as production budget and gross revenue for each movie.  The other dataset called movies also found from kaggle.com, has 7512 values for movie title and contains other variables such as rating, genre, year, votes, and cast information.  The movies dataset contains the IMDb score that was used as the outcome data and was broken up into scores above 6 as favorable, and below 6 as unfavorable.  The two datasets were merged on the same movie title to use more features in our machine learning model.  A random forest classifier model was used to predict the primary outcome of favorable vs unfavorable IMDb score as well as determine the most important features in the dataset. For this project our team met using slack and discussed topics in zoom meetings to collaborate.


The type of modeling we will use to predict IMDB scores is known as a classification algorithm where the model learns patterns from the data, which will allow the model to predict if based on the independent variables we provide if a IMDB movie score should be "Favorable" or "Unfavorable" for each piece of data. We bucket movie scores to be set as Unfavorable if a movie recieved a score from (1 to less than 6) and a Favorable score for 6-10. The movie score rankings in our kaggle dataset is between 1-10.  We selected this dataset because we want  to settle the dispute around which variables play a role in positively impacting a IMDB score. 

For our machine learning model we decided to use a random forest model. These type of supervised machine learning models are great for clustering points of data into functional groups. We are using over 2000 rows of data and random forest models are ideal against overfitting each of those weak learners and training the different pieces of the data.
A big reason we decided to use supervised random forest model and determine the most important variable in predicting IMBD movie scores is that variables can be ranked by importance of input in a natural way. The model can handle thousands of input variables without variable deletion. The model works well on large datasets and is great with handling outliers and nonlinear data.


# Data Exploration
_________________________________________________________

Once we loaded our datasets into our notebook we use needed to determine what data we need to transform and we leverage pandas library for data cleaning. Our dataset need to be transformed before being fed into our machine learning model. We will be merging the two datasets on our movie name columns but we have to make sure that there are no duplicates which will cause errors when joinging the two into one dataframe.
![ERD_quick_db](https://user-images.githubusercontent.com/88444529/151272050-784114b2-8acb-49cb-a745-fe19f0243f7d.PNG)
To predict IMDB scores we will be using a mchine learning model and we need a dataset with enough rows and suitable data to provide our model with enough data to predict our IMDB scores. After reading in our datasets we checked to determine how man duplicates were in each csv. Our movies.csv had 156 duplicates and the top 4000 movies had 48 duplicate rows. First we removed the movie name duplicates which allowed us to merge the two into one dataset. Now that we have removed duplicate movie names we merged our datasets into a single dataframe.

![image](https://user-images.githubusercontent.com/88467263/151076415-3e3134ab-2dc0-4f01-842d-b3439b8fdf33.png)

 Above you see that we have our single dataframe and now we need to determine which columns we need the least that will provide us with no useful meaning in determining IMDB movie scores. We chose to drop "name" column and keep "Movie Tile". We will only need one of these two columns. We decided to drop "Release Date" and keep "Year". Last we dropped the "Country" column we agreed that it did not play a role in the IMDB movie score. Next we will remove null values which leaves us with 2847 rows (See Below) which is a suitable amount of data for our machine learning model. Next we  see that there are columns in our dataframe that are extremely large, these columns are "Production Budget", "domestic Gross", "Worldwide Gross", "Votes", "Budget", "Gross". Before we can feed our dataset into a machine learning model we will need to scale these values down from 9 digits to 3 to 4 digits which will help our machine learning model make a more accurate decision. Belore we split or independent and dependent features the next step is to encode our categorical columns to numerical data to be fed into our model using sklearn labelencoder library. 
 
 ![image](https://user-images.githubusercontent.com/88467263/151077295-b8d678b6-2d7a-4199-b362-417e4034f2ca.png)

# Database and Data Exploration
_________________________________________________________

Data was first cleaned in jupyternotebooks using Pandas.  The movies.csv file was read into pandas and created into a dataframe.  Prior to any data cleaning there were 7363 rows in the movies_df dataframe.  The dataframe was inspected for duplicate entries and null values.  Each of the rows with duplicate entries for "movie_name" were dropped from the dataset.  There were 163 duplicate values that were dropped and 2162 rows that contained null values that were dropped.  The resulting movies_df dataframe had 5201 rows with unique values for "movie_name".
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

