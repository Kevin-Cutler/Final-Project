-- Creating tables for movies
CREATE TABLE movies (
	movie_name VARCHAR NOT NULL,
	rating VARCHAR(15),
	genre VARCHAR NOT NULL,
	year_release VARCHAR (4) NOT NULL,
	score NUMERIC NOT NULL,
	votes NUMERIC NOT NULL,
	director VARCHAR NOT NULL,
	writer VARCHAR NOT NULL,
	star VARCHAR NOT NULL,
	country VARCHAR NOT NULL,
	budget NUMERIC,
	gross NUMERIC NOT NULL,
	company VARCHAR NOT NULL,
	runtime NUMERIC NOT NULL,
	PRIMARY KEY (movie_name)
);

-- Create a table for top_4000_movies_df
CREATE TABLE top_4000_movies (
	Release_Date DATE,
	Movie_Title VARCHAR NOT NULL,
	Production_Budget INT NOT NULL,
	Domestic_Gross INT NOT NULL,
	Worldwide_Gross NUMERIC NOT NULL,
	PRIMARY KEY (Movie_Title)
);
-- Join the two tables together on the movie_name and movie_title
SELECT movies.movie_name, 
	movies.rating, 
	movies.genre,
	movies.year_release,
	movies.score,
	movies.votes,
	movies.director,
	movies.writer,
	movies.star,
	movies.country,
	movies.budget,
	movies.gross,
	movies.company,
	movies.runtime,
	top_4000_movies.Release_Date,
	top_4000_movies.Production_Budget,
	top_4000_movies.Domestic_Gross,
	top_4000_movies.Worldwide_Gross
INTO merged_movies
FROM movies
LEFT JOIN top_4000_movies ON movies.movie_name = top_4000_movies.Movie_Title

SELECT * FROM merged_movies
