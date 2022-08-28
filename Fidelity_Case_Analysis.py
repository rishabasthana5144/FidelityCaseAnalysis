# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 14:47:56 2022

@author: asus
"""

import pandas as pd
%matplotlib qt
df = pd.read_csv('accidents_data.csv')

print(df.shape)
print(df.columns)

# get year month and day
df['timsetamp1'] = pd.DatetimeIndex(df['timestamp'])
df['Year'] = df['timsetamp1'].dt.year
df['Month'] = df['timsetamp1'].dt.month
df['Day'] = df['timsetamp1'].dt.day

df.head()

df_head = df.head()

# Subset data from Jan 2014
df = df[df['Year'] >= 2014]
df.shape

# City
df['borough'].unique()


# Total accidents
df.shape # 346320

# Total injured
df[[x for x in df.columns if 'injured' in x]].sum()

df['Total_injured'] = df[[x for x in df.columns if 'injured' in x]].sum(axis = 1)

df[[x for x in df.columns if 'injured' in x]].sum()


# Total killed
df['Total_killed'] = df[[x for x in df.columns if 'killed' in x]].sum(axis = 1)


df[[x for x in df.columns if 'killed' in x]].sum()


# Top Reasons of collisions 
lst = []
for col in [x for x in df.columns if 'contributing_factor_vehicle' in x]:
    print(col)
    print(df[col].value_counts())
    lst.extend(list(df[col]))


from collections import Counter
lst = dict(sorted(Counter(lst).items(), key = lambda x: x[1], reverse=True))
lst.pop('Unspecified')
lst

dt = pd.DataFrame({'Reason': list(lst.keys()),
                  'Freq': list(lst.values())})

# Removing blank and keeping top 10
dt = dt[1:6]

import matplotlib.pyplot as plt
import seaborn as sns
f, ax = plt.subplots(figsize=(15, 15))
sns.set_color_codes("pastel")
sns.barplot(y=dt.Reason, x=dt.Freq,
            color="r")

#ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(ylabel="Top 10 Reasons",
       xlabel="Collisions Count")
ax.set(xlim = (0,100000))
xlabels = ['{:,.0f}'.format(x) + 'K' for x in ax.get_xticks()/1000]
ax.set_xticklabels(xlabels, size=18)
ax.set_yticklabels(dt.Reason, size=18, rotation=45)
ax.set_xlabel('Collisions Count', size = 18)
ax.set_ylabel('Collisions Causes', size = 18)



# Top reasons by fatalaties
lst = []
for col in [x for x in df.columns if 'killed' in x]:
    col
    print(col)
    tmp = df[df[col] > 0]
    for col in [x for x in tmp.columns if 'contributing_factor_vehicle' in x]:
        print(col)
        print(tmp[col].value_counts())
        lst.extend(list(tmp[col]))
from collections import Counter
lst = Counter(lst)
lst = dict(sorted(Counter(lst).items(), key = lambda x: x[1], reverse=True))
lst.pop('Unspecified')


dt = pd.DataFrame({'Reason': list(lst.keys()),
                  'Freq': list(lst.values())})

# Removing blank and keeping top 10
dt = dt[1:6]

import matplotlib.pyplot as plt
import seaborn as sns
f, ax = plt.subplots(figsize=(15, 15))
sns.set_color_codes("pastel")
sns.barplot(y=dt.Reason, x=dt.Freq,color="r")

#ax.legend(ncol=2, loc="lower right", frameon=True)
#ax.set(xlabel="Top 5 Reasons",
#       ylabel="Fatalities Count")
#ax.set(xlim = (0,200))
#xlabels = ['{:,.0f}'.format(x) + 'K' for x in ax.get_xticks()/1000]
#ax.set_yticklabels(dt.Freq, size=20)
#ax.set_xticklabels(dt.Reason, size=20)
ax.set_ylabel('', size = 15)
ax.set_xlabel('', size = 15)
ax.set_yticklabels(size = 15, labels = dt.Reason)
ax.set_xticklabels(size = 15, labels = ax.get_xticks())

ax.set_xticklabels(ax.get_yticks(), )


# Top reasons by injuries
lst = []
for col in [x for x in df.columns if 'injured' in x]:
    col
    print(col)
    tmp = df[df[col] > 0]
    for col in [x for x in tmp.columns if 'contributing_factor_vehicle' in x]:
        print(col)
        print(tmp[col].value_counts())
        lst.extend(list(tmp[col]))
lst = Counter(lst)
sorted(lst.items(), key = lambda x: x[1], reverse=True)


dt = pd.DataFrame({'Reason': list(lst.keys()),
                  'Freq': list(lst.values())})

# Removing blank and keeping top 10
dt = dt[1:6]

import matplotlib.pyplot as plt
import seaborn as sns
f, ax = plt.subplots(figsize=(15, 15))
sns.set_color_codes("pastel")
sns.barplot(y=dt.Reason, x=dt.Freq,color="r")

#ax.legend(ncol=2, loc="lower right", frameon=True)
#ax.set(xlabel="Top 5 Reasons",
#       ylabel="Fatalities Count")
#ax.set(xlim = (0,200))
xlabels = ['{:,.0f}'.format(x) + 'K' for x in ax.get_xticks()/1000]
#ax.set_yticklabels(dt.Freq, size=20)
#ax.set_xticklabels(dt.Reason, size=20)
ax.set_ylabel('', size = 15)
ax.set_xlabel('', size = 15)
ax.set_yticklabels(size = 15, labels = dt.Reason)
ax.set_xticklabels(size = 15, labels = xlabels)


# Collisions by Year
cols_yr = df.groupby(['Year'])[[x for x in df.columns if 'injured' in x]].size()
    
# Fatalities by Year
fat_yr = df.groupby(['Year'])[[x for x in df.columns if 'killed' in x]].sum()

# Injuries by Year
inj_yr = df.groupby(['Year'])[[x for x in df.columns if 'injured' in x]].sum()


# Collision prone zips
cols_zip = df['zip_code'].value_counts()
cols_zip 

# Treating columns
df['cross_street_name_u'] = df['cross_street_name'].str.upper().str.strip()
df['on_street_name_u'] = df['on_street_name'].str.upper().str.strip()
df['zip_code'] = df['zip_code'].fillna(0)
df['zip_num'] = df['zip_code'].astype(int, errors = 'ignore')


# Collisions prone zips
cols_zip = df['zip_num'].value_counts().reset_index()
cols_zip.rename(columns = {'index': 'zip_num',
                           'zip_num': 'total_collisions'}, inplace=True) 

cols_zip['zip_num'] = cols_zip['zip_num'].astype(str)

cols_zip = cols_zip[0:5]

plt.bar(x = cols_zip.zip_num, height = cols_zip.total_collisions)

# Collisions prone streets under zips
cols_street1 = df.groupby(['zip_num', 'cross_street_name_u']).size().reset_index()


# Fatalaties by zips
cols_zips = df.groupby(['zip_num']).size().reset_index().rename(columns = {0: 'Collisions Count'})

fats_zips = df.groupby(['zip_num'])[[x for x in df.columns if 'killed' in x if 'number_of_persons_killed' not in x]].sum().reset_index()

fats_zips['total_killed'] = fats_zips[[x for x in fats_zips.columns if 'killed' in x]].sum(axis = 1)

# Fatalaties by zips
inj_zips = df.groupby(['zip_num'])[[x for x in df.columns if 'injured' in x if 'number_of_persons_injured' not in x]].sum().reset_index()
inj_zips ['total_injured'] = inj_zips [[x for x in inj_zips.columns if 'injured' in x if 'number_of_persons_injured' not in x]].sum(axis = 1)


# total injuries by killed by zips
fat_inj_zips = fats_zips[['zip_num', 'total_killed']].merge(inj_zips[['zip_num', 'total_injured']], on = 'zip_num')
fat_inj_zips = fat_inj_zips.merge(cols_zips, on = 'zip_num') 

cols_street = df.groupby(['zip_num', 'cross_street_name_u']).size().reset_index()

most_hit_cross_streets = cols_street.groupby('zip_num')[0].max().reset_index().rename(columns = {0: 'Collisions Count'})

most_hit_cross_streets.to_csv('most_hit_cross_streets.csv')

most_hit_cross_streets = most_hit_cross_streets.merge(cols_street, left_on = ['zip_num','Collisions Count'],right_on = ['zip_num',0], how='left')
most_hit_cross_streets = most_hit_cross_streets[['zip_num', 'cross_street_name_u']] 
fat_inj_zips = fat_inj_zips.merge(most_hit_cross_streets, on = 'zip_num')
#fat_inj_zips = fat_inj_zips.merge(cols_zips, on = 'zip_num')
fat_inj_zips.to_csv('Fatality_Injuries_Zip1.csv', index=False) 


# Collisions by day and time of day
df['timsetamp1'] = pd.DatetimeIndex(df['timestamp'])
df['Year'] = df['timsetamp1'].dt.year
df['Month'] = df['timsetamp1'].dt.month
df['Weekday'] = df['timsetamp1'].dt.weekday
df['Hour'] = df['timsetamp1'].dt.hour

# Collisions by weekday
cols_dow = df.groupby('Weekday').size().reset_index()
cols_dow.rename(columns = {0: 'Collisions'}, inplace=True)
cols_dow['Weekday'] = cols_dow['Weekday'].map({0: 'Monday',
                         1: 'Tuesday',
                         2: 'Wednesday',
                         3: 'Thursday',
                         4: 'Friday',
                         5: 'Saturday',
                         6: 'Sunday',
                         })
cols_dow

# Collisions by day of week
f, ax = plt.subplots(figsize=(15, 15))
sns.lineplot(cols_dow['Weekday'], cols_dow['Collisions'], color = 'red')
plt.xlabel('Weekday', size = 25)
plt.ylabel('Collisions Count', size = 25)
ax.set_xticklabels(cols_dow['Weekday'], size = 25)
ax.set_yticklabels(ax.get_yticklabels(), size = 25)



# Collisions by time of day
cols_tod = df.groupby('Hour').size().reset_index()
cols_tod.rename(columns = {0: 'Collisions'}, inplace=True)
cols_tod
sns.set_theme(style = 'whitegrid')
f, ax = plt.subplots(figsize=(15, 15))
sns.lineplot(cols_tod['Hour'], cols_tod['Collisions'], color = 'red', linewidth = 2)
plt.xlabel('Hour', size = 25)
plt.ylabel('Collisions Count', size = 25)
ax.set_xticklabels(ax.get_xticklabels(), size = 25)
ax.set_yticklabels(ax.get_yticklabels(), size = 25)



 
# Analysis by Fatalaties
fat_analysis = df.groupby('Year')[[x for x in df.columns if 'killed' in x]].sum().reset_index()
fat_analysis 
fat_analysis['cyclist_per'] = fat_analysis['number_of_cyclist_killed']/fat_analysis['number_of_persons_killed'] 
fat_analysis['motorist_per'] = fat_analysis['number_of_motorist_killed']/fat_analysis['number_of_persons_killed'] 
fat_analysis['ped_per'] = fat_analysis['number_of_pedestrians_killed']/fat_analysis['number_of_persons_killed'] 
fat_analysis.sum()


fat_analysis.drop('number_of_persons_killed', axis = 'columns', inplace=True)
fat_analysis = fat_analysis.melt('Year')
sns.set_color_codes("pastel")
ax=sns.barplot(x='Year', y='value', hue='variable', data=fat_analysis )
#ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_ylabel('Fatalities', size = 25)
ax.set_xlabel('Year', size = 25)
ax.set_yticklabels(size = 15, labels = ax.get_yticklabels())
ax.set_xticklabels(size = 15, labels = ax.get_xticklabels())
ax.legend(fontsize=20)

