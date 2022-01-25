# Final-Project

## Content

* Selected Topic - What variable are most important in predicted IMDB movie scores in a machine learning model.
* Reason why we selected topic - There is alot of discussion with which variables play a role in the IMDB movie rating scores. We wanted to create a model that supports a deciding factor.
* Descirption of Source of Data - Both or ur movie data sources were downloaded from Kaggle.
* Question we hope to answer with the date - Whate factor influences the IMDB movie score the most.

## Communication Protocals
* We will be using zoom, slack, and github to communicate.



## Decide on a topic for the project—think of a question that can be answered using data.
       
    We want to predict the IMDB movie score using the data provided in the CSV.
    

# Reason topic was selected - Description of the source of data - Questions the team hopes to answer with the data
_____________________________________________________
Our goal in our analysis is to predict the IMDB user score by determining if a movie recieved a favorable or unfavorable IMDB score. We determine the scale to be set at Unfavorable if a movie recieved a score from (1 to less than 6) and a Favorable score for 6-10. The movie score rankings in our kaggle dataset is between 1-10. For our project we have selected 2 movie datasets from kaggle to be the source of our research and analysis. One of our datasets movies.csv captures columns that include IMDB score, IMDB user votes, genre, year released, writer, directer, star, country, budget, gross, and runtime as well as the genre rating. We combine this dataset with an additional csv from kaggle which is complied of the top 4000 movies ranked by production budget and containing release date, domestic gross revenue and worldwide revenue. We selected this dataset because we want  to settle the dispute around which variables play a role in positively impacting a IMDB score. 

# Description of the data exploration phase of the project
_________________________________________________________

Once we loaded our datasets into our notebook we use needed to determine what data we need to transform and we leverage pandas library for data cleaning. Our dataset need to be transformed before being fed into our machine learning model. We will be merging the two datasets on our movie name columns but we have to make sure that there are no duplicates which will cause errors when joinging the two into one dataframe. To predict IMDB scores we will be using a mchine learning modeal and we need a dataset with enough rows and suitable data to provide our model with enough data to predict our IMDB scores. After reading in our datasets we checked to determine how man duplicates were in each csv. Our movies.csv had 156 duplicates and the top 4000 movies had 48 duplicate rows. First we removed the movie name duplicates which allowed us to merge the two into one dataset.

![image](https://user-images.githubusercontent.com/88467263/151076415-3e3134ab-2dc0-4f01-842d-b3439b8fdf33.png)

 Now that we have removed duplicate movie names we merged our datasets into a single dataframe. Below you see that we have our single dataframe and now we need to determine which columns we need the least that will provide us with no useful meaning in determining IMDB movie scores. We chose to drop "name" column and keep "Movie Tile". We have already merged the two datasets into one and we only need one of these two columns. We decided to drop "Release Date" and keep "Year". Last we dropped the "Country" column we agreed that it did not play a role in the IMDB movie score. and removed null values it is time to determine what columns we want to keep and which ones to drop. determine how many null rows we have because these will need to be removed.
 
 ![image](https://user-images.githubusercontent.com/88467263/151077295-b8d678b6-2d7a-4199-b362-417e4034f2ca.png)





# Description of the analysis phase of the project
 
##  Reminders : 
 * Do we know how our model tells the story with the Data?
 
 * What Kind of Message will the dashboard display?
