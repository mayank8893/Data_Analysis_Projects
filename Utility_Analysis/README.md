# Utility_Analysis

This is an **end to end data analysis** prject. I have looked at my utility bill from November 2018 to July 2023. I first went through my bills and put the data into an excel sheet. I **created columns for Date, Total Bill, Electricity Bill, Water Bill and Sewer Bill**. I then exported this to a csv file and loaded it into **Google BigQuery** to answer a few questions via SQL. I found out that:

1. The highest Total Bill was in April 2019 for $240.85. However, this was due to a higher water bill due to a water leak and therefore was an outlier.
2. The highest Electric Bill was in Feb 2022 for $155.04.
3. The lowest Total Bill was in Nov 2021 for $74.13.
4. The Average total bill was $137.08 with the most contribution coming from electric bill with an average of $94.02.
5. We had 3 months with greater than $200 total bill and 7 months with less than $100 total bill.
6. I also calculated the average for each year and month. I found that September was the costliest month and november the cheapest.

I then loaded the data into **python and cleaned it** by checking for any null/na values and converting Date into Pandas datetime. I then:

1. Plotted total and electricity bill as a function of time and found that in general, fall and spring months give lower bills as compared to winter and summer. Also total bill is highly correlated to our electricity bill.
2. Plotted average total bill for each month year over year and found that in general in March, May, November and December we have lower bills. While July, August, September have higher bills. The one huge spike in April 2019 was attributed to a water leak and therefore a huge water bill.
3. Plotted average bill year over year and month over month.
4. Plotted percentage deviation from mean total bill for each month.
5. Grouped the months into seasons and plotted the average bill for each season.
6. Finally made a prediction for our future bills using ARIMA.

<img width="837" alt="Screenshot 2023-08-06 at 2 07 46 PM" src="https://github.com/mayank8893/Data_Analysis_Projects/assets/69361645/15920810-1fcc-4572-9267-487f1125c046">


<img width="767" alt="Screenshot 2023-08-06 at 2 08 03 PM" src="https://github.com/mayank8893/Data_Analysis_Projects/assets/69361645/ae2343ad-217b-42e4-a326-aad4281fd2ea">


<img width="561" alt="Screenshot 2023-08-06 at 2 08 14 PM" src="https://github.com/mayank8893/Data_Analysis_Projects/assets/69361645/379168fb-041e-43f9-a350-d2a4cf20b3eb">
