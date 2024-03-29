import matplotlib.pyplot as plt
import numpy as np

lat = np.array([51.46455753, 51.48315348, 51.47946524, 51.47989288, 51.47091165,
                51.46823832, 51.45097352, 51.44236109, 51.44075611, 51.43310493,
                51.42748611, 51.44209359, 51.41860162, 51.41196388, 51.41919040,
                51.45204327, 51.47407462, 51.46455753, 51.41655806, 51.46071441])

lon = np.array([5.441001338, 5.449615716, 5.463520286, 5.478883978, 5.463348625,
                5.47776818, 5.465205594, 5.47979681, 5.471642895, 5.489753169,
                5.465033933, 5.408385679, 5.407870695, 5.409587309, 5.441001338,
                5.517476483, 5.546744746, 5.550177973, 5.572865385, 5.609406608])

fig, ax = plt.subplots()
ax.scatter(lon, lat, s=50)
for i, txt in enumerate(range(len(lat))):
    ax.annotate(txt, (lon[i], lat[i]), fontsize=12)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Locations')
plt.show()