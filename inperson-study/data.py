'''
This code reads and prints the data from the pickle files in the U01 to U13 folders.
'''
import numpy as np
import pickle
import os

users = list(range(1, 14))
def process_data():
    for user in users:
        folder = os.path.realpath(os.path.dirname(__file__)) + "/" + "U" + str(user).zfill(2)
        print(folder)
        for data_file in sorted(os.listdir(folder)):
            data = pickle.load(open(folder + "/" + data_file, "rb"))
            print(data)
process_data()