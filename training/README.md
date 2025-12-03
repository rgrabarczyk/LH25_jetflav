Run the LundNet training like this:

    multilund.py --config /mnt/users/radekgrabarczyk/LH25Lund/config.yaml --signal /mnt/users/radekgrabarczyk/LH25Lund/51886CMP_cjets.bin --background /mnt/users/radekgrabarczyk/LH25Lund/100000C21_ljets.bin --signal-weights /mnt/users/radekgrabarczyk/LH25Lund/weights_sample_A.npy

Where the initial weights were needed as the CMP c and b jet had a significantly different pt shape compared to light.
This will save an npz file with the ROC curve. plotroc.py is a plotting script that reads in these files.

