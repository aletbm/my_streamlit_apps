import streamlit as st
import cloudpickle
import pandas as pd
import numpy as np
import requests
import json
import urllib.parse
from astropy.time import Time
from astropy import units as u
from poliastro.bodies import Earth, Sun
from poliastro.twobody import Orbit
from poliastro.plotting.misc import plot_solar_system
from poliastro.frames import Planes
from scipy.spatial.transform import Rotation

def url_encode_json(data):
    json_string = json.dumps(data)
    encoded_string = urllib.parse.quote(json_string)
    return encoded_string

def au2km(x):
    return x*149597870.7

def asteroid_orbit_from_orbital_elements(asteroid, object_center="sun"):
    ma = asteroid.ma   #mean anomaly
    e = asteroid.e    #eccentricity
    a = au2km(asteroid.a)     #semi-major axis
    nu = ma + (2*e - (e**3)/4)*np.sin(ma) + (5/4)*(e**2)*np.sin(2*ma) + (13/12)*(e**3)*np.sin(3*ma)     #true anomaly
    p = a*(1-e**2)  #semi-latus rectum
    if object_center == "sun":
        obc = Sun
        mu = 1.32712E11 #standard gravitational parameter of the Sun
    elif object_center == "earth":
        obc = Earth
        mu = 3.98600E5 #standard gravitational parameter of the Earth
    omega = asteroid.w
    i = asteroid.i
    epoch = Time([asteroid.epoch], format='jd')

    # According to https://downloads.rene-schwarz.com/download/M001-Keplerian_Orbit_Elements_to_Cartesian_State_Vectors.pdf

    # Transform to perifocal frame
    r_w = (p / (1 + e * np.cos(nu))) * np.array((np.cos(nu), np.sin(nu), 0))
    
    v_w = ((mu/p)**(1/2)) * np.array((-np.sin(nu), e + np.cos(nu), 0))

    # Rotate the perifocal frame
    R = Rotation.from_euler("ZXZ", [-omega, -i, -omega])
    r_rot = r_w @ R.as_matrix()
    v_rot = v_w @ R.as_matrix()

    # Change units
    r = r_rot * u.km
    v = v_rot * u.km / u.s

    return r, v, epoch, asteroid.full_name, Orbit.from_vectors(obc, r, v, epoch, plane=Planes.EARTH_ECLIPTIC)

st.set_page_config(layout="wide")

path = "/mount/src/my_streamlit_apps/hazardous_asteroid_classification/"
#path="./"
    
with open(path+'model/HAP_model.bin', 'rb') as f_in:
    pipe, le, sfs, rf = cloudpickle.load(f_in)

st.image("https://i.postimg.cc/QMv4swP3/123.png")    
st.write("""# ☄️ Hazardous Asteroid Classifier by [Alexander D. Rios](https://linktr.ee/aletbm)""")

st.write("## Visualizations with Poliastro")
df = pd.read_parquet(path+"app/full_name.gzip").rename(columns={"full_name":"Asteroid name"})

response = asteroids = None
full_name = []
H = []
i = []
om = []
w = []
ma = []
n = []
e = []
a = []
epoch = []
moid = []
class_option = []

class_orbit = ["AMO - Amor",
                "APO - Apollo",
                "AST - Asteroid",
                "ATE - Aten",
                "CEN - Centaur",
                "HYA - Hyperbolic Asteroid",
                "IEO - Interior Earth Object",
                "IMB - Inner Main-belt Asteroid",
                "MBA - Main-belt Asteroid",
                "MCA - Mars-crossing Asteroid",
                "OMB - Outer Main-belt Asteroid",
                "PAA - Parabolic Asteroid",
                "TJN - Jupiter Trojan",
                "TNO - TransNeptunian Object"]

EPOCH = Time("2000-01-01 12:00:00", scale="tdb")    
frame = plot_solar_system(outer=False, epoch=EPOCH, interactive=True, use_3d=True)    
    
col1, col2 = st.columns([0.3, 0.7], border=True)
with col1:
    st.write("#### List of asteroids")
    st.write("Select one or more asteroids from the list to view their orbits.")
    st_df = st.dataframe(df, on_select="rerun", selection_mode="multi-row", height=550, width=300, hide_index=True)
with col2:
    st.write("#### Orbital visualization")
    if len(st_df["selection"]["rows"]):
        for index in st_df["selection"]["rows"]:
            aux = df.iloc[index].values[0].split("(")[0]
            asteroid = aux if aux.replace(" ", "") != "" else df.iloc[st_df["selection"]["rows"][0]].values[0].split("(")[1].replace(")", "")
            url = f'https://ssd-api.jpl.nasa.gov/sbdb.api?sstr={asteroid}&phys-par=1'
            response = requests.get(url).json()
            
            full_name.append(asteroid)
            H.append(float(response["phys_par"][0]["value"]) if response is not None and len(response["phys_par"]) else 0.0)
            i.append(float(response["orbit"]["elements"][3]["value"]) if response is not None and len(response["orbit"]["elements"]) else 0.0)
            om.append(float(response["orbit"]["elements"][4]["value"]) if response is not None and len(response["orbit"]["elements"]) else 0.0)
            w.append(float(response["orbit"]["elements"][5]["value"]) if response is not None and len(response["orbit"]["elements"]) else 0.0)
            ma.append(float(response["orbit"]["elements"][6]["value"]) if response is not None and len(response["orbit"]["elements"]) else 0.0)
            n.append(float(response["orbit"]["elements"][9]["value"]) if response is not None and len(response["orbit"]["elements"]) else 0.0)
            moid.append(float(response["orbit"]["moid"]) if response is not None and response["orbit"]["moid"] else 0.0)
            class_option.append(response["object"]["orbit_class"]["code"])#class_orbit.index(response["object"]["orbit_class"]["code"] + " - " + response["object"]["orbit_class"]["name"]))
            e.append(float(response["orbit"]["elements"][0]["value"]) if response is not None and len(response["orbit"]["elements"]) else 0.0)
            a.append(float(response["orbit"]["elements"][1]["value"]) if response is not None and len(response["orbit"]["elements"]) else 0.0)
            epoch.append(float(response["orbit"]["epoch"]))
            asteroids = pd.DataFrame({
                        'full_name':full_name,
                        'H': H, 
                        'i': i,
                        'om': om,
                        'w': w,
                        'ma': ma,
                        'n': n,
                        'moid': moid,
                        'e': e,
                        'a': a,
                        'epoch': epoch,
                        'class': class_option,
                        })
            
        for k in range(len(asteroids)):
            _, _, epoch, ast_name, orb = asteroid_orbit_from_orbital_elements(asteroids.iloc[k], object_center="sun")
            frame.plot(orb, label=ast_name)
            
    frame._figure.update_layout(autosize=False, legend=dict(orientation="h", font=dict(size=8)), width=800, height=400, margin=dict(l=20, r=20, b=20, t=20, pad=0), template="plotly_dark")
    st.plotly_chart(frame.show(), use_container_width=True)
        
    if asteroids is not None:
        st.write("#### Parameters of the asteroid")
        ast_df = st.dataframe(asteroids)
        
st.divider()
st.write("## Asteroid classification")
response = None
col1, col2 = st.columns([0.3, 0.7], border=True)
with col1:
    st.write("#### List of asteroids")
    st.write("Select an asteroid from the list to classify it.")
    st_df = st.dataframe(df, on_select="rerun", selection_mode="single-row", height=450, width=300, hide_index=True)
    if len(st_df["selection"]["rows"]):
        aux = df.iloc[st_df["selection"]["rows"][0]].values[0].split("(")[0]
        asteroid = aux if aux.replace(" ", "") != "" else df.iloc[st_df["selection"]["rows"][0]].values[0].split("(")[1].replace(")", "")
        url = f'https://ssd-api.jpl.nasa.gov/sbdb.api?sstr={asteroid}&phys-par=1'
        response = requests.get(url).json()
        
    H_value = float(response["phys_par"][0]["value"]) if response is not None and len(response["phys_par"]) else 0.0
    i_value = float(response["orbit"]["elements"][3]["value"]) if response is not None and len(response["orbit"]["elements"]) else 0.0
    om_value = float(response["orbit"]["elements"][4]["value"]) if response is not None and len(response["orbit"]["elements"]) else 0.0
    w_value = float(response["orbit"]["elements"][5]["value"]) if response is not None and len(response["orbit"]["elements"]) else 0.0
    ma_value = float(response["orbit"]["elements"][6]["value"]) if response is not None and len(response["orbit"]["elements"]) else 0.0
    n_value = float(response["orbit"]["elements"][9]["value"]) if response is not None and len(response["orbit"]["elements"]) else 0.0
    moid_value = float(response["orbit"]["moid"]) if response is not None and response["orbit"]["moid"] else 0.0
    class_option_value = class_orbit.index(response["object"]["orbit_class"]["code"] + " - " + response["object"]["orbit_class"]["name"]) if response is not None else 0
        
with col2:
    st.write("#### Load your asteroid data")
    with st.form("my_form"):
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            H = st.number_input("Absolute magnitude",
                                value=H_value,
                                format="%0.15f",
                                min_value=0.0,
                                help="""# Absolute magnitude
In astronomy, absolute magnitude is a measure 
of the luminosity of a celestial object on an inverse 
logarithmic astronomical magnitude scale; 
the more luminous (intrinsically bright) an object, 
the lower its magnitude number. 

![H](https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Phase_angle_explanation.png/375px-Phase_angle_explanation.png)

An object's absolute magnitude is the magnitude that an observer would see 
if the asteroid were at a distance of 1 AU from the Sun, 
with a phase angle (α) of zero. This angle is formed 
between the Sun and the Earth as seen from the asteroid.
""")
            
            diameter = st.number_input("Object diameter (equivalent to a sphere) [Km]",
                                        format="%0.15f",
                                        min_value=0.0,
                                        help="""# Object diameter
The diameter (D) of an object is the measurement of the distance 
across the object from one side to the other, passing through 
the center. 

![D](https://upload.wikimedia.org/wikipedia/commons/thumb/0/03/Circle-withsegments.svg/480px-Circle-withsegments.svg.png)

""")

            albedo = st.number_input("Geometric albedo",
                                    format="%0.15f",
                                    min_value=0.0,
                                    help="""# Geometric albedo
In astronomy, the geometric albedo of a celestial body is the ratio 
of its actual brightness as seen from the light source (i.e. at zero 
phase angle) to that of an idealized flat, fully reflecting, diffusively 
scattering (Lambertian) disk with the same cross-section. (This phase 
angle refers to the direction of the light paths and is not a phase 
angle in its normal meaning in optics or electronics.)

![albedo](https://upload.wikimedia.org/wikipedia/commons/thumb/2/28/Diffuse_reflector_sphere_disk.png/640px-Diffuse_reflector_sphere_disk.png)

""")

            i = st.number_input("Inclination (Angle relative to the x-y ecliptic plane) [Degrees]", 
                                format="%0.15f",
                                value=i_value,
                                max_value=360.0,
                                min_value=0.0,
                                help="""# Inclination
This variable is the inclination of the asteroid's orbit relative 
to the ecliptic plane.

![Inclination](https://upload.wikimedia.org/wikipedia/commons/thumb/9/95/Orbital_elements.svg/450px-Orbital_elements.svg.png)

In this diagram, we can also observe the variables `om` and `w`,
which we will discuss shortly. The ecliptic plane is the plane 
defined by the Earth's orbit around the Sun, i.e., it is the 
plane that the Earth's orbit defines.

![ecliptic](https://upload.wikimedia.org/wikipedia/commons/f/f1/Ecl%C3%ADptica_diagrama2.png)

""")

            om = st.number_input("Longitude of the ascending node [Degrees]",
                                format="%0.15f",
                                value=om_value,
                                max_value=360.0,
                                min_value=0.0,
                                help="""# Longitude of the ascending node
The longitude of the ascending node is the angle from 
a reference direction, called the origin of longitude, 
to the direction of the ascending node, measured in 
a reference plane, as shown in the adjacent image:

![om](https://upload.wikimedia.org/wikipedia/commons/thumb/e/eb/Orbit1.svg/langes-600px-Orbit1.svg.png)

""")
            
        with col2_2:
            w = st.number_input("Argument of perihelion [Degrees]",
                                format="%0.15f",
                                value=w_value,
                                max_value=360.0,
                                min_value=0.0,
                                help="""# Argument of perihelion
The argument of perihelion `w` is the angle from 
the ascending node to the perihelion, measured in 
the orbital plane of the object and in the direction 
of its motion.

![w](https://upload.wikimedia.org/wikipedia/commons/thumb/e/eb/Orbit1.svg/langes-600px-Orbit1.svg.png)

The point where the periapsis intersects with the body's 
orbit, is the perihelion.

""")

            ma = st.number_input("Mean anomaly [Degrees]",
                                format="%0.15f",
                                value=ma_value,
                                max_value=360.0,
                                min_value=0.0,
                                help="""# Mean anomaly
In celestial mechanics, the mean anomaly is the fraction 
of an elliptical orbit's period that has elapsed since 
the orbiting body passed periapsis, expressed as an angle 
which can be used in calculating the position of that body 
in the classical two-body problem. It is the angular distance 
from the pericenter which a fictitious body would have if it 
moved in a circular orbit, with constant speed, in the same 
orbital period as the actual body in its elliptical orbit.

![ma](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f1/Mean_anomaly_diagram.png/485px-Mean_anomaly_diagram.png)

""")

            n = st.number_input("Mean motion [Degrees/Days]",
                                format="%0.15f",
                                value=n_value,
                                min_value=0.0,
                                help="""# Mean motion
In orbital mechanics, mean motion (represented by n) is the 
angular speed required for a body to complete one orbit, 
assuming constant speed in a circular orbit which completes 
in the same time as the variable speed, elliptical orbit 
of the actual body.

![ma](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f1/Mean_anomaly_diagram.png/485px-Mean_anomaly_diagram.png)

""")

            moid = st.number_input("Minimum orbit intersection distance with Earth [AU]",
                                    format="%0.15f",
                                    value=moid_value,
                                    min_value=0.0,
                                    help="""# Minimum orbit intersection distance with Earth (MOID)
The MOID is the distance between the closest points of two 
objects' orbits. 

![moid](https://www.researchgate.net/profile/Brent-Barbee-2/publication/36174303/figure/fig7/AS:669493323890696@1536631055384/Minimum-Orbital-Intersection-Distance.png)

""")

            class_option = st.selectbox(
                "Orbit classification",
                ["AMO - Amor",
                "APO - Apollo",
                "AST - Asteroid",
                "ATE - Aten",
                "CEN - Centaur",
                "HYA - Hyperbolic Asteroid",
                "IEO - Interior Earth Object",
                "IMB - Inner Main-belt Asteroid",
                "MBA - Main-belt Asteroid",
                "MCA - Mars-crossing Asteroid",
                "OMB - Outer Main-belt Asteroid",
                "PAA - Parabolic Asteroid",
                "TJN - Jupiter Trojan",
                "TNO - TransNeptunian Object"],
                index=class_option_value,
                help="""| Abbreviation | Title | Description |
| ------------- | ------ | ----------- |
|AMO|Amor|Near-Earth asteroid, orbit similar to 1221 Amor (a > 1.0 AU; 1.017 AU < q < 1.3 AU).|
|APO|Apollo|Near-Earth asteroids with orbits that cross Earth's orbit, similar to 1862 Apollo (a > 1.0 AU; q < 1.017 AU).|
|AST|Asteroid|The asteroid's orbit does not match any defined class.|
|ATE|Aten|Near-Earth asteroid, orbit similar to 2062 Aten (a < 1.0 AU; Q > 0.983 AU).|
|CEN|Centaur|Objects with orbits between Jupiter and Neptune (5.5 AU < a < 30.1 AU).|
|HYA|Hyperbolic Asteroid|Asteroids in hyperbolic orbits (e > 1.0).|
|IEO|Interior Earth Object|An asteroid's orbit entirely contained within Earth's orbit (Q < 0.983 AU).|
|IMB|Inner Main-belt Asteroid|Asteroids with orbital elements restricted by (a < 2.0 AU; q > 1.666 AU).|
|MBA|Main-belt Asteroid|Asteroids with orbital elements restricted by (2.0 AU < a < 3.2 AU; q > 1.666 AU).|
|MCA|Mars-crossing Asteroid|Asteroids that cross Mars's orbit, restricted by (1.3 AU < q < 1.666 AU; a < 3.2 AU).|
|OMB|Outer Main-belt Asteroid|Asteroids with orbital elements restricted by (3.2 AU < a < 4.6 AU).|
|PAA|Parabolic Asteroid|Asteroids in parabolic orbits (e = 1.0).|
|TJN|Jupiter Trojan|Asteroids trapped in Jupiter's Lagrange points L4/L5 (4.6 AU < a < 5.5 AU; e < 0.3).|
|TNO|TransNeptunian Object|Objects with orbits beyond Neptune (a > 30.1 AU).|
""")

        submitted = st.form_submit_button(label="Classify it", type="primary", use_container_width=True)

        if submitted:
            error = False
            vars = ["Inclination", "Longitude of the ascending node", "Argument of perihelion", "Mean anomaly"]
            for x, param in enumerate([i, om, w, ma]):
                if not (param >= 0 and param <= 360):
                    st.error(f'The parameter \"{vars[x]}\" must be greater than or equal to 0 and less than or equal to 360.', icon="🚨")
                    error = True

            vars = ["Absolute magnitude", "Mean motion", "Minimum orbit intersection distance with Earth"]
            for x, param in enumerate([H, n, moid]):
                if not param > 0:
                    st.error(f'The parameter \"{vars[x]}\" must be greater than to 0.', icon="🚨")
                    error = True
            
            vars = ["Object diameter", "Geometric albedo"]
            for x, param in enumerate([diameter, albedo]):
                if not param >= 0:
                    st.error(f'The parameter \"{vars[x]}\" must be greater than or equal to 0.', icon="🚨")
                    error = True
                    
            if error == False:       
                asteroid = pd.DataFrame({'spkid': np.nan,
                    'neo': "",
                    'H': [H], 
                    'diametro': diameter,
                    'albedo': np.nan,
                    'epoch': np.nan,
                    'e': np.nan,
                    'a': np.nan,
                    'q': np.nan,
                    'i': [i],
                    'om': [om],
                    'w': [w],
                    'ma': [ma],
                    'ad': np.nan,
                    'n': [n],
                    'tp': np.nan,
                    'per': np.nan,
                    'moid': [moid],
                    'class': str(class_option).split(" ")[0],
                    'rms': np.nan
                    })
                
                if diameter == 0:
                    if albedo == 0:
                        albedo = 0.0615
                    asteroid["diametro"] = (1329/albedo)*np.power(10, -0.4*H)
                    
                X_test = pipe.transform(asteroid)
                X_test = sfs.transform(X_test)
                pred = rf.predict(X_test)
                pred = le.inverse_transform(np.ravel(pred))[0]
                if pred == 'Y':
                    st.warning("This asteroid is potentially dangerous! Get out of here, my friend, run!", icon="😬")
                else:
                    st.success("You're in lucky! This is a friendly asteroid.", icon="😀")