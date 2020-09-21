# VERNE

Voigt fitting code to derive Ew and Rv using Normalised spEctra (VERNE)

---
## CaT version 

The current version is tailor made to fit the 3 Calsium triplet lines and derive the Equivalent Width, Radial velocity, [Fe/H], and [Ca/H].

The current version requires: 
- lmfit 
- specutils
- astropy
- seaborn

---
You will need a input file in .txt 

```
Lambda  Flux_spec  Noise 
8000.756  0.9336  1.0660 
8002.006  1.2451  1.0531 
```
> If you don't have the noise, don't worry, at line 395 you will find an your option.

### Set up the code 

---
In the lines between 400 and 490 you will find some set up options as:

- interactive_mode 
- Magnitude and Distance 
- Bandpass
- RV correction (if necessary) 

> In the interactive_mode you have 3 options True, False or 'Guess' where 'Guess' is rv based guessing system for when you have some ideia about what is the radial velocity. 

> Magnitude and Distance are used to devrive [Fe/H] and [Ca/H] applying the Starkenburg et al. 2010 calibrations.

> For the bandpass you have 3 options: Rutledge et al. 1997a; Carrera et al. 2007; and Vasquez et al. 2018. You can also create your own.

### Run 

```
python3 LMVERNE.py [spectrum.txt]
```

---
