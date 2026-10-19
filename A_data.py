#!/usr/bin/env python
from pathlib import Path
import os

def get_sequences(file="./csv/BIOSCAN_5M_Insect_Dataset_metadata.csv"):
    with open(file, 'r') as inputfile:
        line = inputfile.readline()
        while len(line) != 0:
            print(line[11], "\n")
            line = inputfile.readline().split(',')
        inputfile.close()


def get_line(file="./csv/BIOSCAN_5M_Insect_Dataset_metadata.csv"):
    with open(file, 'r') as inputfile:
        line = inputfile.readline()
        print(line)
        while len(line) != 0:
            # print(line, "\n")
            line = inputfile.readline().split(',')
        inputfile.close()

def write_to_file(output_path, line, overwrite=True):
     if overwrite:
          open(output_path, 'w').close()
     with open(output_path, "a") as outputfile:
          outputfile.write(line)


def convert_BIOSCAN_metadata(file="./csv/BIOSCAN_5M_Insect_Dataset_metadata.csv", output_path="./data/manifest.csv", num_pairs=10):
    counter = 0
    with open(file, 'r') as inputfile:
        line = inputfile.readline().split(',')
        write_to_file(output_path, ','.join([line[0], line[1], line[9], line[11]]) + '\n')
        print(f"Saved headerfile to {output_path}")
        while len(line) != 0:
        # for i in range (0, 100):
            line = inputfile.readline().split(',')
            # print(','.join((line[1], (line[8], line[9]), line[11])), '\n')
            if len(line[9]) != 0:
                counter += 1
                write_to_file(output_path, ','.join([line[0], line[1], line[9], line[11]]) + '\n', False)
            if counter == num_pairs:
                break
        inputfile.close()
            

if __name__ == "__main__":

    # output = "./data/manifest.csv"
    # with open("./csv/BIOSCAN_5M_Insect_Dataset_metadata.csv", 'r') as inputfile:
    #         header = inputfile.readline()
    #         write_to_file(Path(output), header)
    #         print(header)
    #         print(inputfile.readline())

    convert_BIOSCAN_metadata()
    print("completed")

