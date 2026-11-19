import pandas as pd #pandas manage data well
from matplotlib import pyplot as plt #for plotting

#opening the file using pandas and printing the first five rows to get a sense of the data
#this file was sitting on my desktop 
file_io = 'file:///Users/mybangpetersen/Desktop/earthquakes.xlsx'
df = pd.read_excel(file_io)
print(df.head())

#plotting a part of the data
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
st = plt.suptitle('Earthquake Data \n A look into where earthquakes occur and how strong they are', fontsize=16)

ax[0,0].plot(df['longitude'], df['latitude'], 'o', markersize=5, color='blue')
ax[0,0].set_xlabel('Longitude [deg]')
ax[0,0].set_ylabel('Latitude [deg]')

ax[0,1].plot(df['depth'], df['rms'], 'o', markersize=5, color='black')
ax[0,1].set_xlabel('Depth [km]')
ax[0,1].set_ylabel('RMS')

ax[1,0].plot(df['depth'], df['rms'], 'o', markersize=5, color='red')
ax[1,0].set_xlabel('Depth [km]')
ax[1,0].set_ylabel('RMS amplitude (strength of the earthquake)')

ax[1,1].plot(df['dmin'], df['rms'], 'o', markersize=5, color='red')
ax[1,1].set_xlabel('Horizontal distance [km]')
ax[1,1].set_ylabel('RMS amplitude (strength of the earthquake)')

plt.show()
