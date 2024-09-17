import mne 
import matplotlib.pyplot as plt
import os

dirData = "Data/"

#file_path = "Data/SARTINI^DAISY.edf"

list = [f for f in os.listdir(dirData) if "edf" in f]

for file_path in list:

    raw = mne.io.read_raw_edf(dirData+file_path, preload=True)

    print(raw.info)

    raw.plot()
    plt.show()