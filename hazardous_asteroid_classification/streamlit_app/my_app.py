import streamlit as st
import cloudpickle
import pandas as pd
import numpy as np

path = "/mount/src/my_streamlit_apps/hazardous_asteroid_classification/"
#path="./"

with open(path+'model/HAP_model.bin', 'rb') as f_in:
    pipe, le, sfs, rf = cloudpickle.load(f_in)

st.set_page_config(layout="wide")

st.image("https://i.postimg.cc/QMv4swP3/123.png")    
st.write("""# ☄️ Hazardous Asteroid Classifier by [Alexander D. Rios](https://linktr.ee/aletbm)""")

with st.form("my_form"):
    st.write("#### Load your asteroid data")

    col1, col2 = st.columns(2)
    with col1:
        H = st.number_input("Absolute magnitude",
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
                            max_value=360.0,
                            min_value=0.0,
                            help="""# Longitude of the ascending node
The longitude of the ascending node is the angle from 
a reference direction, called the origin of longitude, 
to the direction of the ascending node, measured in 
a reference plane, as shown in the adjacent image:

![om](https://upload.wikimedia.org/wikipedia/commons/thumb/e/eb/Orbit1.svg/langes-600px-Orbit1.svg.png)

""")
        
    with col2:
        w = st.number_input("Argument of perihelion [Degrees]",
                            format="%0.15f",
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
                                min_value=0.0,
                                help="""# Minimum orbit intersection distance with Earth (MOID)
The MOID is the distance between the closest points of two 
objects' orbits. 

![moid](https://www.researchgate.net/profile/Brent-Barbee-2/publication/36174303/figure/fig7/AS:669493323890696@1536631055384/Minimum-Orbital-Intersection-Distance.png)

""")

        class_option = st.selectbox(
            "Orbit classification",
            ("AMO - Amor",
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
            "TNO - TransNeptunian Object"),
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