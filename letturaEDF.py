import mne 
import matplotlib.pyplot as plt
import os

dirData = "Data/"

### metodo per la lettura con file in divisi per cartelle

all_data = []
for dirpath, dirnames, filenames in os.walk(dirData):
    print(f"Directory: {dirpath}")
    for file in filenames:
        if file.endswith('.edf'):  # Controlla se il file ha estensione .edf
            file_path = os.path.join(dirpath, file)
            print(f"\tFile: {file_path}")
            raw = mne.io.read_raw_edf(file_path, preload=True)

            # all_data.append(raw)

            sfreq = raw.info['sfreq']  # Frequenza di campionamento 


            if sfreq != 128:
                print(f"######################Attenzione #########################\nfrequenza di {sfreq} su {file}")

            # print(sfreq)

            # raw.plot()
            # plt.show()


# print(all_data) 

# all_data.clear() #pulisce tutto il vettore

# print(all_data)     



### metodo per la lettura con file in unica cartella


# list = [f for f in os.listdir(dirData) if "edf" in f]

# for file_path in list:

#     raw = mne.io.read_raw_edf(dirData+file_path, preload=True)

#     ann = raw.annotations

#     print(ann)

#     print(raw.info)

#     raw.plot()
#     plt.show()