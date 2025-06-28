import numpy as np
import pandas as pd
import os

folder1, folder2 = "Excel_Messwerte", "Data_Files"
type1, type2 = ".xlsx", ".txt"

parent_dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"
dir1_path = parent_dir_path + folder1 + "/"
dir2_path = parent_dir_path + folder2 + "/"

filelist = [filename[:-len(type1)] for filename in os.listdir(dir1_path)]

def comma_to_dot(string):
    new_string = ""
    for char in string:
        if char == ",":
            new_string = new_string + "."
        else:
            new_string = new_string + char
    return float(new_string)

def xlsx_to_txt(xlsx_file_path, txt_file_path):
    print(xlsx_file_path)
    df = pd.read_excel(xlsx_file_path, header=None, dtype=str)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            print(df[j][i])
            df[j][i] = comma_to_dot(df[j][i])
    df.to_csv(txt_file_path, header=None, index=None, sep=',', mode='w')

for filename in filelist:
    xlsx_to_txt(dir1_path + filename + type1, dir2_path + filename + type2)