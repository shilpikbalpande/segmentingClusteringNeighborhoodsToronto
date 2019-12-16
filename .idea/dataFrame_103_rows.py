from bs4 import BeautifulSoup
import requests
import pandas as pd
import html5lib
import numpy as np
import geocoder as ge
import folium
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values
from sklearn.cluster import KMeans
import matplotlib.cm as cm
import matplotlib.colors as colors


pd.set_option('display.max_columns', 999)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)

geoData = "http://cocl.us/Geospatial_data"
radius=500
LIMIT=100
CLIENT_ID = 'AO4R21JIVFD5F2AN0IZTNSZIZCFLUEKHPMI5ULDTPDZ5G41P' # your Foursquare ID
CLIENT_SECRET = 'ZLGHMTHWFLNA1CVY53ZOXXU5BGE2PZV3WH5XWMCT0WU1FUNK' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version
listOfCo = pd.read_csv(geoData)
html_doc = "https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M"

listOfCo.columns =  ['Postcode','Latitude','Longitude']
r = requests.get(html_doc)
soup = BeautifulSoup(r.content,'html.parser')
tbl = soup.find("table")
dfs = pd.read_html(str(tbl))
df = pd.DataFrame(dfs[0])
df2 = df

df['Borough'] = df['Borough'].replace('Not assigned',np.NaN)

df = df.dropna(subset=['Borough']).reset_index()

i = len(df)
for i in range(i):
    if df['Borough'][i] != 'Not assigned' and df['Neighbourhood'][i] == 'Not assigned':
        df['Neighbourhood'][i] = df['Borough'][i]
        i = i +1
    else:
        continue

dfs = df.groupby('Postcode')['Neighbourhood'].apply(','.join)

df = df.drop(['Neighbourhood'],axis = 1)

df = df.drop_duplicates(['Postcode','Borough']).reset_index()
df3 = pd.merge(df, dfs, on='Postcode', how='inner')
df3 = df3.drop_duplicates()
df3 = df3.drop(['level_0','index'],axis = 1)
df3 = pd.merge(df3, listOfCo, on='Postcode', how='inner')

print (df3.shape)
print (df3.tail)
