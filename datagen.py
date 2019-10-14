import csv
import os
from os import listdir

# This script will take the SETI data and turn it into three CSV files
# so that it is easier for tensorflow to read it

dir_ext = ["test", "train", "valid"]
directory_path = "/home/dalton/School/DataMining/primary_small/"
labels = ["brightpixel", "narrowband", "narrowbanddrd", "noise", "squarepulsednarrowband", "squiggle", "squigglesquarepulsednarrowband"]
label_map = {}
i = 0
for label in labels:
    label_map[label] = i
    i += 1

for tset in dir_ext:
    directory_path_prelabel = directory_path + tset + "/"
    with open(tset + '_data.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar="|", quoting=csv.QUOTE_MINIMAL)
        for label in labels:
            directory_path_label = directory_path_prelabel + label
            for file in listdir(directory_path_label):
                filewriter.writerow([directory_path_label + "/" + file, label_map[label]])