import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pytz
import matplotlib.dates as mdates

# Load the CSV file
df = pd.read_csv('lerobot_datasets_last.csv')

# Convert the 'creation_date' column to datetime and ensure it is timezone-aware
df['creation_date'] = pd.to_datetime(df['creation_date']).dt.tz_convert('UTC')

# Calculate the date one year ago from today, making it timezone-aware
one_year_ago = datetime.now(pytz.UTC) - timedelta(days=500)

# Filter the data for the last year
df_last_year = df[df['creation_date'] >= one_year_ago]

grouped_data = df_last_year.groupby(['creation_date', 'robot_type']).size().reset_index(name='count')

cumulative_data = grouped_data.groupby('creation_date')['count'].sum().cumsum().reset_index()

plt.figure(figsize=(14, 7))
plt.plot(cumulative_data['creation_date'], cumulative_data['count'], marker='o', color='b', linewidth=2)
plt.title('LeRobot dataset uploaded last year', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Cumulative Number of Datasets', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.tight_layout()
plt.show()

top_n_robots = 10 
most_frequent_robots = grouped_data.groupby('robot_type')['count'].sum().nlargest(top_n_robots).index

filtered_data = grouped_data[grouped_data['robot_type'].isin(most_frequent_robots)]
sorted_data = filtered_data.groupby('robot_type')['count'].sum().sort_values(ascending=False)

plt.figure(figsize=(14, 7))
sorted_data.plot(kind='bar', color=['#FF9999','#66B2FF','#99FF99','#FFCC99','#FF6666'])
plt.title(f'Number of Datasets for the {top_n_robots} most frequent robots', fontsize=16, fontweight='bold')
plt.xlabel('Robot Type', fontsize=14)
plt.ylabel('Number of Datasets', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
