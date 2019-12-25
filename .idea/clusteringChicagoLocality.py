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
import os
import numpy as np

restaurant_cuisine = r"Indian Restaurant"


pd.set_option('display.max_columns', 999)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)

radius=500
LIMIT=100
CLIENT_ID = 'KOZW2334OOYPKS1RF2ZGSFZPRGCRF0WNEQNQPGCKRRZWRQBI' # your Foursquare ID, this has been masked
CLIENT_SECRET = '4L1MG2ZHVPDHICNNVRAVJVEQQKIJRTRWBPSOG1GHYESE5CF4' # youyour Foursquare Secret, this has been masked
VERSION = '20180605' # Foursquare API version

chicago_localities = "https://en.wikipedia.org/wiki/List_of_neighborhoods_in_Chicago"
r = requests.get(chicago_localities)
soup = BeautifulSoup(r.content,'html.parser')
table = soup.find('table')
neighborhoodList = []
dfs = pd.read_html(str(table))


df = pd.DataFrame(dfs[0]).drop(['Community area'],axis=1).reset_index()
df = df[:10]


# function to get coordinates
def get_lat_and_lng(neighborhood):
    lat_lng = None
    # loop until you get the coordinates
    while(lat_lng is None):
        g = ge.arcgis('{}, Chicago, USA'.format(neighborhood))
        lat_lng = g.latlng
    return lat_lng

lat_and_land = [ get_lat_and_lng(neighborhood) for neighborhood in df["Neighborhood"].tolist() ]

df_coordinates = pd.DataFrame(lat_and_land, columns=['Latitude', 'Longitude'])
df['Latitude'] = df_coordinates['Latitude']
df['Longitude'] = df_coordinates['Longitude']


address = 'Chicago, USA'
geolocator = Nominatim(user_agent="ny_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Chicago, USA are {}, {}.'.format(latitude, longitude))

map_chicago = folium.Map(location=[latitude, longitude], zoom_start=10)

for lat, lng, neighborhood in zip(df['Latitude'], df['Longitude'], df['Neighborhood']):
    label = '{}'.format(neighborhood)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_chicago)

map_chicago

map_chicago.save('map_chicago.html')

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
    nearby_venues.columns = ['Neighborhood',
                             'Neighborhood Latitude',
                             'Neighborhood Longitude',
                             'Venue',
                             'Venue Latitude',
                             'Venue Longitude',
                             'Venue Category']

    return(nearby_venues)

chicago_venues = getNearbyVenues(names = df['Neighborhood'],
                                 latitudes = df['Latitude'],
                                 longitudes =  df['Longitude']
                                 )

chicago_restaurant_venues = chicago_venues[chicago_venues['Venue Category'].str.contains("Restaurant")]

## Count venues by each Neighborhood
chicago_venues.groupby('Neighborhood').count()

chicago_restaurant_venues.groupby('Neighborhood').count()

##count unique categories
print('There are {} uniques categories.'.format(len(chicago_venues['Venue Category'].unique())))


# one hot encoding
chicago_onehot = pd.get_dummies(chicago_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
chicago_onehot['Neighborhood'] = chicago_venues['Neighborhood']
chicago_grouped = chicago_onehot.groupby('Neighborhood').mean().reset_index()
chicago_restaurant = chicago_grouped[["Neighborhood",restaurant_cuisine]]


kclusters = 5
chicago_clustering = chicago_grouped.drop('Neighborhood', 1)


# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(chicago_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]
chicago_merged = chicago_restaurant.copy()
chicago_merged["Cluster Labels"] = kmeans.labels_


# merge chicago_merged with chicago to add latitude/longitude for each neighborhood
chicago_merged = chicago_merged.join(df.set_index('Neighborhood'), on='Neighborhood')
chicago_merged.head() # check the last columns!
chicago_merged.sort_values(["Cluster Labels"], inplace=True)


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(chicago_merged['Latitude'], chicago_merged['Longitude'], chicago_merged['Neighborhood'], chicago_merged['Cluster Labels']):
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
print(chicago_merged.loc[chicago_merged['Cluster Labels'] == 0])
print(chicago_merged.loc[chicago_merged['Cluster Labels'] == 1])
print(chicago_merged.loc[chicago_merged['Cluster Labels'] == 2])
print(chicago_merged.loc[chicago_merged['Cluster Labels'] == 3])
print(chicago_merged.loc[chicago_merged['Cluster Labels'] == 4])


