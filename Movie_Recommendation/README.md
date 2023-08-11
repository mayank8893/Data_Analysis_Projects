# Movie_Recommendation

For this project, I will look at the movie dataset from https://files.grouplens.org/datasets/movielens/ml-25m.zip. This dataset contains two main files: **movies** with information on title, genre and movie id of the movie and the **ratings** file which contains information about rating, user id and movie id. First I will design a **search engine using tfidf** which gives the 5 most similar movie names to what you have input. 

Then I have designed a **movie recommendation engine**, where:
1. I have found users who liked the same movie as us (similar users).
2. Combined a list of other movies liked by these users.
3. Made a list of movies that were liked by 10% or more of similar users.
4. Found out how much all users like these movies.
5. Created a score by **calculating the differential** between movies liked by similar users and by all users.
6. Used this score to **recommend 10 movies** that you could watch.

So just plug in a movie and get a recommendation to watch your next movie tonight.

<img width="740" alt="Screen Shot 2023-08-11 at 9 52 22 AM" src="https://github.com/mayank8893/Movie_Recommendation/assets/69361645/3ead70b6-f81a-4553-bf05-1e72f24a7ceb">
