# Predicting IMDB Movie Scores

![IMDB_Logo_2016](https://user-images.githubusercontent.com/88118759/150700799-37b89265-d9af-44fa-9e21-e0dc1b8e6544.svg)



## Overview
_____________________________
The goal of this project is to take two different movie datasets in order to develop a machine learning model that can predict whether or not it has a Favorable or Unfavorable movie score.  Both of the datasets were taken from kaggle.com.  The dataset top_4000_movies has data from 4000 different movies for features such as production budget and gross revenue for each movie.  The other dataset called movies also found from kaggle.com, has 7512 values for movie title and contains other variables such as rating, genre, year, votes, and cast information.  The movies dataset contains the IMDb score that was used as the outcome data and was broken up into scores above 6 as favorable, and below 6 as unfavorable.  The two datasets were merged on the same movie title to use more features in our machine learning model.  A random forest classifier model was used to predict the primary outcome of favorable vs unfavorable IMDb score as well as determine the most important features in the dataset. For this project our team met using slack and discussed topics in zoom meetings to collaborate.


Our goal in our analysis is to predict the IMDB user score by determining if a movie recieved a favorable or unfavorable IMDB score. We determine the scale to be set at Unfavorable if a movie recieved a score from (1 to less than 6) and a Favorable score for 6-10. The movie score rankings in our kaggle dataset is between 1-10. For our project we have selected 2 movie datasets from kaggle to be the source of our research and analysis. One of our datasets movies.csv captures columns that include IMDB score, IMDB user votes, genre, year released, writer, directer, star, country, budget, gross, and runtime as well as the genre rating. We combine this dataset with an additional csv from kaggle which is complied of the top 4000 movies ranked by production budget and containing release date, domestic gross revenue and worldwide revenue. We selected this dataset because we want  to settle the dispute around which variables play a role in positively impacting a IMDB score. 

# Description of the data exploration phase of the project
_________________________________________________________

Once we loaded our datasets into our notebook we use needed to determine what data we need to transform and we leverage pandas library for data cleaning. Our dataset need to be transformed before being fed into our machine learning model. We will be merging the two datasets on our movie name columns but we have to make sure that there are no duplicates which will cause errors when joinging the two into one dataframe. To predict IMDB scores we will be using a mchine learning modeal and we need a dataset with enough rows and suitable data to provide our model with enough data to predict our IMDB scores. After reading in our datasets we checked to determine how man duplicates were in each csv. Our movies.csv had 156 duplicates and the top 4000 movies had 48 duplicate rows. First we removed the movie name duplicates which allowed us to merge the two into one dataset. Now that we have removed duplicate movie names we merged our datasets into a single dataframe.

![image](https://user-images.githubusercontent.com/88467263/151076415-3e3134ab-2dc0-4f01-842d-b3439b8fdf33.png)

 Above you see that we have our single dataframe and now we need to determine which columns we need the least that will provide us with no useful meaning in determining IMDB movie scores. We chose to drop "name" column and keep "Movie Tile". We will only need one of these two columns. We decided to drop "Release Date" and keep "Year". Last we dropped the "Country" column we agreed that it did not play a role in the IMDB movie score. Next we will remove null values which leaves us with 2847 rows (See Below) which is a suitable amount of data for our machine learning model. Next we  see that there are columns in our dataframe that are extremely large, these columns are "Production Budget", "domestic Gross", "Worldwide Gross", "Votes", "Budget", "Gross". Before we can feed our dataset into a machine learning model we will need to scale these values down from 9 digits to 3 to 4 digits which will help our machine learning model make a more accurate decision. Belore we split or independent and dependent features the next step is to encode our categorical columns to numerical data to be fed into our model using sklearn labelencoder library. 
 
 ![image](https://user-images.githubusercontent.com/88467263/151077295-b8d678b6-2d7a-4199-b362-417e4034f2ca.png)





# Description of the analysis phase of the project
______________________________________________________

* Description of preliminary data preprocessing

* Description of preliminary feature engineering and preliminary feature selection, including the decision-making process


* Description of how data was split into training and testing sets


* Explanation of model choice, including limitations and benefits

 
##  Reminders : 
 * Do we know how our model tells the story with the Data?
 
 * What Kind of Message will the dashboard display?

* Link to Google slides: [here](https://docs.google.com/presentation/d/1K3iR-3VI6Z6oexiieo5eO8K5bZ4lB0vluZBnga12fdw/edit?usp=sharing)
