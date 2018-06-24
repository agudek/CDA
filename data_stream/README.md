Stream data
---

The datasets used in this assignment are ```capture20110816.pcap.netflow.labeled```, ```capture20110818.pcap.netflow.labeled```, ```scenario_10_filtered.csv``` and ```scenario_10_discretised.csv```, of which the csv files are located in this repository.
The labeled netflow data and more information about its contents can be found at:

[```capture20110816.pcap.netflow.labeled```](https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-47/)
[```capture20110818.pcap.netflow.labeled```](https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-51/)

---

The visualisations in the report were created using ```datavis.py```.

---

Min-wise sampling can be found in ```sampling.py```

---

The count-min sketch implementation is located in ```sketching.py```.

---

Discretisation was performed in ```discrete_models.py```. This creates a csv file containing discretised data and requires ```scenario_10_filtered.csv```, which in turn is obtained by running ```filter.py```.

---

The profiling task can be found in ```profiling.py```

---

Requirements to run
---

To run the code, several requirements need to be met. All used code is written in Python3 and use the following libraries: 

- Murmurhash3
- Pandas
- Numpy
- Scikit learn
- Matplotlib
- Seaborn
- Plotly

All libraries can be install using ```pip3```.

