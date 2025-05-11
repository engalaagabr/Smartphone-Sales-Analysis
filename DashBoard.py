import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import KBinsDiscretizer

# Page Config
st.set_page_config(page_title='Smartphone Sales Dashboard', page_icon='📱', layout='wide')

# Load the data
file_path = 'smartphone_sales_dataset.csv'
data = pd.read_csv(file_path)

# Standardize Column Names
# This ensures consistent column access regardless of spacing or case differences
data.columns = data.columns.str.strip().str.replace(' ', '_').str.lower()

# Verify Required Columns
required_columns = ['os', 'price_usd', 'profit', 'sales_revenue', 'rating', 'brand']
missing_columns = [col for col in required_columns if col not in data.columns]

if missing_columns:
    raise ValueError(f"Missing required columns in the dataset: {missing_columns}")

# Data Preprocessing
# Handling missing values
for col in data.columns:
    if data[col].dtype in ['float64', 'int64']:
        data[col].fillna(data[col].median(), inplace=True)
    else:
        data[col].fillna(data[col].mode()[0], inplace=True)

# Feature Engineering
# 1. Price Segmentation
price_bins = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
data['price_segment'] = price_bins.fit_transform(data[['price_usd']])
data['price_segment'] = data['price_segment'].map({0: 'Low', 1: 'Mid', 2: 'High'})

# 2. Profit Margin
data['profit_margin'] = (data['profit'] / data['price_usd']) * 100


# Sidebar Filters
st.sidebar.header('Filter Options')
selected_os = st.sidebar.multiselect('Operating System', options=data['os'].unique(), default=data['os'].unique())
selected_segment = st.sidebar.multiselect('Price Segment', options=data['price_segment'].unique(), default=data['price_segment'].unique())

filtered_data = data[(data['os'].isin(selected_os)) & (data['price_segment'].isin(selected_segment))]

# Dashboard Title
st.title('📊 Smartphone Sales Dashboard')
st.markdown('A comprehensive dashboard to explore smartphone sales, profit margins, and customer ratings.')

# Key Metrics
st.subheader('Key Metrics')
col1, col2, col3 = st.columns(3)
col1.metric('Total Sales Revenue', f"${filtered_data['sales_revenue'].sum():,.0f}")
col2.metric('Total Profit', f"${filtered_data['profit'].sum():,.0f}")
col3.metric('Average User Rating', f"{filtered_data['rating'].mean():.2f}")

# Top 10 Best-Selling Brands
st.subheader('Top 10 Best-Selling Brands')
top_brands = filtered_data.groupby('brand')['sales_revenue'].sum().sort_values(ascending=False).head(10)
fig3 = px.bar(top_brands, x=top_brands.index, y=top_brands.values, title='Top 10 Best-Selling Brands', labels={'x': 'Brand', 'y': 'Total Sales Revenue'})
st.plotly_chart(fig3)

# Price vs Profit Relationship
st.subheader('Price vs Profit Relationship')
fig5 = px.scatter(filtered_data, x='price_usd', y='profit', color='os', title='Price vs Profit Relationship', labels={'price_usd': 'Price (USD)', 'profit': 'Profit'})
st.plotly_chart(fig5)


# Profit Margin Distribution
st.subheader('Profit Margin Distribution')
fig7 = px.histogram(filtered_data, x='profit_margin', nbins=30, title='Profit Margin Distribution', labels={'profit_margin': 'Profit Margin (%)'})
st.plotly_chart(fig7)

# User Ratings Distribution
st.subheader('User Ratings Distribution')
fig8 = px.histogram(filtered_data, x='rating', nbins=20, title='User Ratings Distribution', labels={'rating': 'User Rating'})
st.plotly_chart(fig8)

# Correlation Heatmap
st.subheader('Correlation Heatmap')
plt.figure(figsize=(12, 8))
numeric_cols = filtered_data.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt='.2f')
st.pyplot(plt)

st.markdown('---')
st.caption('Thanks')


