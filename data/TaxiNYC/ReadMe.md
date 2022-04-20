TaxiNYC: In-Flow/Out-Flow at NYC
===========================================================

**If you use the data, please cite the following paper.**

`Fiorini, S., Ciavotta, M., & Maurino, A. (2021). Listening to the city, attentively: A Spatio-Temporal Attention Boosted Autoencoder for the Short-Term Flow Prediction Problem. arXiv preprint arXiv:2103.00983. `

**TaxiNYC** dataset is a `hdf5` file named `NYC10_Taxi_M16x8_T60_InOut.h5`, which includes two subsets:

* `date`: a list of timeslots, which is associated the **data**. 
* `data`: a 4D tensor of shape (number_of_timeslots, 2, 16, 8), of which `data[i]` is a 3D tensor of shape (2, 16, 8) at the timeslot `date[i]`, `data[i][0]` is a `16x8` new-flow matrix and `data[i][1]` is a `16x8` end-flow matrix. 

Note: `*.h5` is `hdf5` file, one can use the follow code to view the data:

```
import h5py
f = h5py.File('NYC10_Taxi_M16x8_T60_InOut.h5')
for ke in f.keys():
    print(ke, f[ke].shape)
```
