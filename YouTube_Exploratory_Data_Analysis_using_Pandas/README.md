# YouTube_Exploratory_Data_Analysis_using_Pandas

For this project, I have looked at data from YouTube containing information about the published date, trending date, category, views, likes, dislikes and comment counts for trending videos in 10 countries. The data can be found here: https://www.kaggle.com/datasets/datasnaek/youtube-new.

As a first step of the data analysis process, I have looked at the data and converted data types such that they are easily manipulated during analysis. After that I examined for any missing, N/A values. Once, that was done, I combined the different data frames from all the countries into one combined data frame. I then inserted a category column into the data frame by reading the categories from a JSON file. This concluded the **data cleaning** portion of the data analysis.

Once the data was cleaned, I performed **exploratory data analysis (EDA)** and answered questions like:
1. What was the ratio of likes and dislikes in different categories?
2. What was the category of the trending videos in each country?
3. What were the top 5 videos trending in each country?
4. Is the most liked video also the most trending video?
5. What was the number of days between publishing a video and when it became trending?
6. What were the most liked categories?
7. Which categories got the most comments?
8. What were the most frequently words occurring in tags and descriptions?
9. Were there any correlations between views, likes, dislikes and comments?



<img width="522" alt="Screenshot 2023-07-31 at 10 28 00 AM" src="https://github.com/mayank8893/YouTube_Exploratory_Data_Analysis_using_Pandas/assets/69361645/3b3a0f8f-6f17-429d-8955-051b30903e0c">
<img width="409" alt="Screenshot 2023-07-31 at 10 28 16 AM" src="https://github.com/mayank8893/YouTube_Exploratory_Data_Analysis_using_Pandas/assets/69361645/b1a0ff1a-aa33-4a66-8238-c02e8a6bb8dd">
<img width="281" alt="Screenshot 2023-07-31 at 10 28 27 AM" src="https://github.com/mayank8893/YouTube_Exploratory_Data_Analysis_using_Pandas/assets/69361645/2957683c-808f-4c8f-93c4-3caba279b4c0">
<img width="692" alt="Screenshot 2023-07-31 at 10 28 37 AM" src="https://github.com/mayank8893/YouTube_Exploratory_Data_Analysis_using_Pandas/assets/69361645/807076ba-11be-4530-8f52-0f4ed69fba03">
<img width="392" alt="Screenshot 2023-07-31 at 10 28 48 AM" src="https://github.com/mayank8893/YouTube_Exploratory_Data_Analysis_using_Pandas/assets/69361645/d30fee37-e65a-4976-9c88-e291860c121c">
