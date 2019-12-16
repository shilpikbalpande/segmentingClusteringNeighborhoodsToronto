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
CLIENT_ID = 'AO4R21JIVFD5F2AN0IZTNSZIZCFLSSUEKHPMI5ULDTPDZ5G41P' # your Foursquare ID, this has been masked 
CLIENT_SECRET = 'ZLGHMTHWFLNA1CVY53ZOXXU5BGE2PZV3WH5XWMCT0WU1FUNK' # your Foursquare Secret, this has been masked 
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

address = 'Toronto, Canada'
geolocator = Nominatim(user_agent="ny_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Toronto, Canada are {}, {}.'.format(latitude, longitude))


map_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)


for lat, lng, borough, neighborhood in zip(df3['Latitude'], df3['Longitude'], df3['Borough'], df3['Neighbourhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)

map_toronto



def getNearbyVenues(names, latitudes, longitudes, radius=500):

    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        # print(name)

        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID,
            CLIENT_SECRET,
            VERSION,
            lat,
            lng,
            radius,
            LIMIT)

        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']

        # return only relevant information for each nearby venue
        venues_list.append([(
                                name,
                                lat,
                                lng,
                                v['venue']['name'],
                                v['venue']['location']['lat'],
                                v['venue']['location']['lng'],
                                v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighbourhood',
                             'Neighbourhood Latitude',
                             'Neighbourhood Longitude',
                             'Venue',
                             'Venue Latitude',
                             'Venue Longitude',
                             'Venue Category']

    return(nearby_venues)

toronto_venues = getNearbyVenues(names = df3['Neighbourhood'],
                                 latitudes = df3['Latitude'],
                                 longitudes =  df3['Longitude']
                                 )


toronto_venues.groupby('Neighbourhood').count()
# one hot encoding
toronto_onehot = pd.get_dummies(toronto_venues[['Venue Category']], prefix="", prefix_sep="")
# add neighborhood column back to dataframe
toronto_onehot['Neighbourhood'] = toronto_venues['Neighbourhood']

toronto_grouped = toronto_onehot.groupby('Neighbourhood').mean().reset_index()
toronto_grouped


# only consider Borough with name containing Toronto
df3 = df3[df3['Borough'].map(lambda x: str(x).__contains__("Toronto"))]

df3 = df3.reset_index()
num_top_venues = 5


for Neighborhood in toronto_grouped['Neighbourhood']:

    temp = toronto_grouped[toronto_grouped['Neighbourhood'] == Neighborhood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})


import numpy as np
num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighbourhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighbourhoods_venues_sorted = pd.DataFrame(columns=columns)
neighbourhoods_venues_sorted['Neighbourhood'] = toronto_grouped['Neighbourhood']


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    return row_categories_sorted.index.values[0:num_top_venues]

for ind in np.arange(toronto_grouped.shape[0]):
    neighbourhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)

neighbourhoods_venues_sorted.head()



# move neighborhood column to the first column
toronto_onehot.head()
kclusters = 3

toronto_grouped_clustering = toronto_grouped.drop('Neighbourhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]


# Let's create a new dataframe that includes the cluster as well as the top 10 venues for each neighborhood.

# add clustering labels
# neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

neighbourhoods_venues_sorted.insert(0, 'Cluster_Labels', kmeans.labels_)


# only consider Borough with name containing Toronto
df3 = df3[df3['Borough'].map(lambda x: str(x).__contains__("Toronto"))]

df3 = df3.reset_index()


toronto_merged = df3

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
toronto_merged = toronto_merged.join(neighbourhoods_venues_sorted.set_index('Neighbourhood'), on='Neighbourhood')
toronto_merged.head() # check the last columns!

toronto_merged=toronto_merged.dropna()


toronto_merged['Cluster_Labels'] = toronto_merged.Cluster_Labels.astype(int)

# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array] 

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'], toronto_merged['Neighbourhood'], toronto_merged['Cluster_Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)

map_clusters

## display clsuter number 1
print(toronto_merged.loc[toronto_merged['Cluster_Labels'] == 1, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]])
