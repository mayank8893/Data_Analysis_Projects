#!/usr/bin/env python
# coding: utf-8

# # Customer segmentation Analysis
# 
# In this project we will explore a dataset of product sales in Brazil. The dataset has information about the geological location of sales, revenue distribution, customer distribution, delivery information, customer behavior and product popularity.

# In[1]:


# importing all the necessary libraries.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# reading in the datasets.
geo = pd.read_csv('geolocation_olist_public_dataset.csv')
order = pd.read_csv('olist_public_dataset_v2.csv')
cust = pd.read_csv('olist_public_dataset_v2_customers.csv')
pay = pd.read_csv('olist_public_dataset_v2_payments.csv')
trans = pd.read_csv('product_category_name_translation.csv')


# ### Geological location data exploration.
# 
# Exploring the data with respect to geological location.

# In[3]:


geo.head()


# In[4]:


# dropping duplicates.
geo = geo.drop_duplicates(subset=None, keep='first', inplace=False)


# **Grouping zip codes to same latitude and longitude by taking their mean.**

# In[5]:


centroid = geo.groupby('zip_code_prefix').agg({
    'lat': 'median',
    'lng': 'median',
    'city': pd.Series.mode
}).reset_index()
centroid['count'] = geo.groupby('zip_code_prefix').size().reset_index(
    name='counts')['counts']
centroid.head()


# In[ ]:





# In[27]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Distribution of markets in Brazil.
latitudes = centroid["lat"]
longitudes = centroid["lng"]
counts = centroid["count"]

# Create a scatter plot
plt.scatter(longitudes, latitudes, s=10, c=counts, alpha=0.3)

# Adjust the size range of points
plt.ylim(min(latitudes) - 1, max(latitudes) + 1)
plt.xlim(min(longitudes) - 1, max(longitudes) + 1)

# Add colorbar
cbar = plt.colorbar()
cbar.set_label("Count")

# Add labels and title
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Scatter Plot of Centroid Data")


# As we can see, the market is distributed highly in Southern Brazil approximately around lat: -22 and lng: -46.  If the company is interested in expanding the market it is **highly recommended to expand the market in north and west Brazil.**

# ### Revenue Distribution
# 
# We will join the order dataset and geological dataset before doing analysis.

# In[16]:


order.head()


# In[17]:


# dropping duplicatess.
order = order.drop_duplicates(subset=None, keep='first', inplace=False)


# In[18]:


# merging the data.
geo_order = pd.merge(centroid,
                     order,
                     how='right',
                     left_on='zip_code_prefix',
                     right_on='customer_zip_code_prefix')


# In[19]:


# grouping the orders by zip code:
geo_rev = geo_order.groupby('customer_zip_code_prefix').agg({
    'lat':
    'median',
    'lng':
    'median',
    'order_products_value':
    'sum'
}).reset_index()


# In[31]:


latitudes = geo_rev["lat"]
longitudes = geo_rev["lng"]
order_values = geo_rev["order_products_value"]

# Create a scatter plot with colormap representing density
plt.scatter(longitudes, latitudes, c=order_values, cmap="YlOrRd", s=10, alpha=0.6)

# Add colorbar
cbar = plt.colorbar()
cbar.set_label("Count")

# Set map style
plt.style.use("dark_background")

# Add labels and title
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Density Map of Geo_rev Data")


# As shown in the map above, we can conclude there are three main hotspots which are located in south-eastern Brazil. These three places are the market with the highest revenue. **To raise company's revenue, it is highly recommended to add more market around these three hotspots.** 

# ### Customer Distribution

# In[32]:


# Grouping customers by zip code.
geo_order = geo_order.dropna(subset=['zip_code_prefix'])
geo_cust = geo_order.groupby('customer_zip_code_prefix').agg({
    'lat':
    'median',
    'lng':
    'median',
    'customer_id':
    'count'
}).reset_index()


# In[35]:


# geological distribution of customers.
latitudes = geo_cust["lat"]
longitudes = geo_cust["lng"]
customer_ids = geo_cust["customer_id"]

# Create a scatter plot with colormap representing density
plt.scatter(longitudes, latitudes, c=customer_ids, cmap="YlOrRd", s=10, alpha=0.6)

# Add colorbar
cbar = plt.colorbar()
cbar.set_label("Customer ID")

# Set map style
#plt.style.use("seaborn-dark")

# Add labels and title
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Density Map of Geo_cust Data")


# **Most of the customers are located in southeastern Brazil which leads to most sales in this region.**

# With this geological approach, there are a few things I can recommend:
# 
# 1. For branding expansion, it is best for the company to expand the market at the west and north Brazil. However, with low income from these location, it would be better to build only a few market for each city or area since the interest of the customer is currently low.
# 2. For increase in revenue, it is best for the company to expand the market at these three hotspots:
#     lat:-23, lng:-46
#     lat: -22, lng:-43
#     lat: -19, lng:-43
#     Since these hotspots have the highest number of customer and revenue, it shows that the customer around these hotspots interested with the market.

# ### Order date and time exploration

# In[36]:


# converting relevant columns to datetime
geo_order['order_delivered_customer_date'] = pd.to_datetime(
    geo_order['order_delivered_customer_date'])
geo_order['order_estimated_delivery_date'] = pd.to_datetime(
    geo_order['order_estimated_delivery_date'])


# In[37]:


# dropping the rows for which delivery date was empty as the order must have been cancelled.
geo_order = geo_order.dropna(subset=['order_delivered_customer_date'])


# In[38]:


# calculating difference between delivery date and estimated delivery date.
# ideally this should be 0 or negative.
geo_order['range_time'] = (
    geo_order['order_delivered_customer_date'] -
    geo_order['order_estimated_delivery_date']).astype('timedelta64[D]')


# In[39]:


# calculating if an order was delayed or if it arrived early.
delay = geo_order['range_time'][geo_order['range_time'] > 0]
early = abs(geo_order['range_time'][geo_order['range_time'] < 0])


# In[40]:


# binning and plotting the early distribution.
bins = [0, 3, 7, 14, 30, 90]
labels = [
    '1-3 days', '4-7 days', '7-14 days', '14-30 days', 'More than 1 month'
]
early = pd.cut(early, bins=bins, labels=labels)
sns.countplot(y=early)


# Its good that the order are arriving usually early by 7-30 days, but **this shows a high level of inaccuracy in the delivery date estmation.**

# In[41]:


# binning and plotting the delay distribution.
bins = [0, 3, 7, 14, 30, 90, 300]
labels = [
    '1-3 days', '4-7 days', '7-14 days', '14-30 days', '1-3 month',
    'More than 3 months'
]
delay = pd.cut(delay, bins=bins, labels=labels)
sns.countplot(y=delay)


# Most of the orders are delayed by 4-7 days. We can push these to 1-3 days to improve customer satisfaction.

# In[42]:


# checking out which locations have delayed deliveries.
geo_order['range_time_default'] = geo_order['range_time'].apply(
    lambda x: 1 if x > 0 else 0)


# In[43]:


# grouping by zipcode.
delay_place = geo_order.groupby('zip_code_prefix').agg({
    'lat':
    'median',
    'lng':
    'median',
    'range_time_default':
    'mean'
})


# In[46]:


latitudes = delay_place["lat"]
longitudes = delay_place["lng"]
range_times = delay_place["range_time_default"]

# Create a scatter plot with colormap representing density
plt.scatter(longitudes, latitudes, c=range_times, cmap="YlOrRd", s=10, alpha=0.6)

# Add colorbar
cbar = plt.colorbar()
cbar.set_label("Range Time Default")

# Set map style
plt.style.use("ggplot")

# Add labels and title
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Density Map of Delay_place Data")


# In[47]:


# printing percentage of early, delayed and on-time deliveries. 
normal = geo_order['range_time'][geo_order['range_time'] == 0]
print('normal:',
      round(normal.count() * 100 / geo_order['range_time'].count(),
            3), '%', '\nearly:',
      round(early.count() * 100 / geo_order['range_time'].count(), 3), '%',
      '\ndelay:',
      round(delay.count() * 100 / geo_order['range_time'].count(), 3), '%')


# ### Order timing analysis
# 
# We can look at the order timing and use to decide what time is the best for promotion as that will be the time when most customers are online looking at products.

# In[48]:


# changing timestamp to datetime.
order['order_purchase_timestamp'] = pd.to_datetime(
    order['order_purchase_timestamp'])


# In[49]:


# plotting hourly distribution of orders.
best_time_hours = (order.order_purchase_timestamp).dt.hour
sns.distplot(best_time_hours)


# In[52]:


# plotting daily distribution of orders.
best_time_day = order.order_purchase_timestamp.apply(lambda x: x.weekday())
sns.distplot(best_time_day)


# In[53]:


# plotting monthly distribution of orders.
best_time_month = (order.order_purchase_timestamp).dt.month
sns.distplot(best_time_month)


# Based on these three plots we can say that:
# 
# 1. In between midnight and 8 AM is not a good time for promotion.
# 2. Saturdays and Sundays are less preferable times for promotion.
# 3. October is a less preferable month for promotion.

# ### Customer Behavior exploration.
# 
# In this section we will explore customer behavior with respect to their ordering habits.

# In[54]:


cust.head()


# In[55]:


# dropping duplicates.
cust = cust.drop_duplicates(subset=None, keep='first', inplace=False)


# In[56]:


# merge with the order table on customer id.
order_cust = pd.merge(cust,
                      order,
                      how='right',
                      left_on='customer_id',
                      right_on='customer_id')


# In[57]:


# calculate the days from most recent order to every order from customer.
order_cust = order_cust.dropna(subset=['order_aproved_at'])
order_cust['order_aproved_at'] = pd.to_datetime(order_cust['order_aproved_at'])
order_cust['latest'] = max(order_cust.order_aproved_at)
order_cust[
    'range_time'] = order_cust['latest'] - order_cust['order_aproved_at']
recent = order_cust.groupby('customer_unique_id').agg({
    'range_time': 'min'
}).astype('timedelta64[D]')
order_cust.head()


# In[59]:


# getting frequency of orders and 
frequent = order_cust.groupby('customer_unique_id').agg(
    {'order_items_qty': 'sum'})
monetary = order_cust.groupby('customer_unique_id').agg(
    {'order_products_value': 'sum'})
frequent.head()
monetary.head()


# In[65]:


# finding the earlist purchase time by each customer.
tenure = order_cust.groupby('customer_unique_id').agg({
    'range_time': 'max'
}).astype('timedelta64[D]')


# In[66]:


# creating a dataframe of recency, frequency, monetary value and tenure of each customer.
rfmt = pd.concat([recent, frequent, monetary, tenure], axis=1)
rfmt.columns = ['recency', 'frequency', 'monetary', 'tenure']


# In[67]:


rfmt.describe()


# In[69]:


# assigning rank to each customer. 1 will be lowest and 5 will be highest.
label1 = [5, 4, 3, 2, 1]
label2 = [1, 2, 3, 4, 5]
rfmt['R'] = pd.qcut(rfmt['recency'],
                    q=[0, 0.2, 0.4, 0.6, 0.8, 1],
                    labels=label1)
rfmt['F'] = pd.cut(rfmt['frequency'], bins=[0, 1, 2, 3, 5, 900], labels=label2)
rfmt['M'] = pd.qcut(rfmt['monetary'],
                    q=[0, 0.2, 0.4, 0.6, 0.8, 1],
                    labels=label2)
rfmt['T'] = pd.qcut(rfmt['tenure'],
                    q=[0, 0.2, 0.4, 0.6, 0.8, 1],
                    labels=label2)


# In[70]:


rfmt = rfmt.reset_index()


# In[72]:


# if recency is lowest, frequency is highest and monetary value is highest they are the best customers.
best = rfmt[rfmt.R.isin([4, 5]) & rfmt.F.isin([4, 5]) & rfmt.M.isin([4, 5])]
best['segment'] = 'BEST'

# loyal spenders.
loyal_spender = rfmt[rfmt['R'].isin([4, 5]) & rfmt['F'].isin([2, 3])
                     & rfmt['M'].isin([2, 3]) & rfmt['T'].isin([3, 4, 5])]
loyal_spender['segment'] = 'LOYAL SPENDER'

# potential loyal customers.
potential_loyal = rfmt[rfmt['R'].isin([4, 5]) & rfmt['F'].isin([2, 3])
                       & rfmt['M'].isin([2, 3]) & rfmt['T'].isin([1, 2])]
potential_loyal['segment'] = 'POTENTIAL LOYAL'

#loyal customer.
loyal_cust = rfmt[rfmt['R'].isin([4, 5]) & rfmt['F'].isin([1])
                  & rfmt['M'].isin([1]) & rfmt['T'].isin([3, 5])]
loyal_cust['segment'] = 'LOYAL CUSTOMER'

# new customer.
new_cust = rfmt[rfmt['R'].isin([4, 5]) & rfmt['F'].isin([1])
                & rfmt['M'].isin([1]) & rfmt['T'].isin([1, 2])]
new_cust['segment'] = 'NEW CUSTOMER'

# promising customer.
promising = rfmt[rfmt['R'].isin([2, 3]) & rfmt['F'].isin([3, 5])
                 & rfmt['M'].isin([3, 5])]
promising['segment'] = 'PROMISING'

# can not lose customers.
cant_lose = rfmt[rfmt['R'].isin([1]) & rfmt['F'].isin([4, 5])
                 & rfmt['M'].isin([4, 5])]
cant_lose['segment'] = 'CAN NOT LOSE'

# about to sleep customers.
about_to_sleep = rfmt[rfmt['R'].isin([1]) & rfmt['F'].isin([4, 5])
                      & rfmt['M'].isin([4, 5])]
about_to_sleep['segment'] = 'ABOUT TO SLEEP'

# Hibernating Customers.
hibernating = rfmt[rfmt['R'].isin([1]) & rfmt['F'].isin([2, 3])
                   & rfmt['M'].isin([2, 3])]
hibernating['segment'] = 'HIBERNATING'

lost = rfmt[rfmt['R'].isin([1]) & rfmt['F'].isin([1]) & rfmt['M'].isin([1])
            & rfmt['T'].isin([1])]
lost['segment'] = 'HIBERNATING'


# In[74]:


# adding these categories to our customer order table.
rfmt_segment = pd.concat([
    best, loyal_spender, potential_loyal, new_cust, loyal_cust, promising,
    cant_lose, about_to_sleep, hibernating, lost
])


# In[75]:


rfmt_segment.sample(10)


# We have thus categorized customers. **We can give points for purchased products to the best customers. Furthermore, we can promote things that interest new customers from their previous purchase.**

# ### Popularity Analysis.
# 
# Looking at the popularity of products.

# In[76]:


trans.head()


# In[77]:


# merging order data with product data.
order_trans = pd.merge(trans,
                       order,
                       how='right',
                       on=['product_category_name', 'product_category_name'])
order_en = order_trans.drop(['product_category_name'], axis=1)


# In[78]:


order_en.head()


# In[79]:


# plot percentage of respective ratings.
n = order_en.groupby('review_score')['review_score'].agg(['count'])

prod_count = order_en['product_id'].nunique()

cust_count = cust['customer_unique_id'].nunique() - prod_count

rating_count = order_en['review_score'].count() - cust_count

ax = n.plot(kind='barh', legend=False, figsize=(15, 10))
plt.title(
    'Total pool: {:,} Products, {:,} Customers, {:,} Ratings given'.format(
        prod_count, cust_count, rating_count),
    fontsize=20)
plt.axis('off')

for i in range(1, 6):
    ax.text(n.iloc[i - 1][0] / 4,
            i - 1,
            'Rating {}: {:.0f}%'.format(i,
                                        n.iloc[i - 1][0] * 100 / n.sum()[0]),
            color='white',
            weight='bold')


# **12% of ratings are 1 star**. This is not good news. The company should work on fixing this.

# In[81]:


# listing the the most popular products.
pop_prod = pd.DataFrame(
    order_en.groupby('product_category_name_english')['review_score'].count())
most_popular = pop_prod.sort_values('review_score', ascending=False)
most_popular.head(10)


# In[82]:


plt.rcParams['figure.figsize'] = (10, 10)
most_popular.head(30).plot(kind="barh")


# Special thanks to Gita Kartika Suriah for the dataset and guiding code on GitHub. 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


pop_prod = pd.DataFrame(
    order_en.groupby('product_id')['review_score'].count())
most_popular = pop_prod.sort_values('review_score', ascending=False)
most_popular.head(10)


# In[ ]:


plt.rcParams['figure.figsize'] = (10, 10)
most_popular.head(30).plot(kind="barh")


# In[ ]:




