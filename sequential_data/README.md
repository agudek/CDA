Sequential data
---

The visualisations in the report were created using ```dataviz.py``` and ```datavis_attacks.py```
```plotting.py``` shows the correlation between columns in the dataset.

---

Simple time series prediction using sliding windows can be found in ```sliding_window.py```

---

ARMA can be found in ```armaSingleDataset.py``` and ```armaJoinedDataset.py```
```armaSingleDataset.py``` contains arma performed on just Batadal Training set 1 which
does not show any interesting results because of the lack of attacks. This was done to
investigate how ARMA will behave in such an instance.
```armaJoinedDataset.py``` is the main solution to the pca assignment.

```plotting.py``` shows autocorrelation plots.


---

Discretisation and anomaly detection using N-grams was performed in ```discrete_models.py```

---

The PCA task can be found in ```pca.py```

---

Requirements to run
---

To run the code, several requirements need to be met. All used code is written in Python3 and use the following libraries: 

- Pandas
- Numpy
- Scikit learn
- Matplotlib
- Seaborn
- Statsmodels
- Plotly

All libraries can be install using ```pip3```.

