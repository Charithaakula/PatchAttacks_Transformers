import subprocess
from ctypes import sizeof
from os import listdir
from os.path import isfile, join
from re import M

path = ""

# last_lines = [subprocess.check_output(['tail', '-1', path + filename]) for filename in [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('.txt')]]
filename = 'combinedfiles.txt'
with open(filename) as f:
    mylist = f.read().splitlines() 

filename_last_lines = [[(filename, subprocess.check_output(['tail', '-1', path + filename]))] for filename in mylist]

fulldict = []
non_finished_list = []

#create a dict from these all keys 
#create a full dict
#create a 

for line in filename_last_lines:     
    localdict = dict()
    mytext = line[0][1]
    mytext = str(mytext, 'UTF-8')

    #print(mytext)
    all_keys = []
    all_vals = []    
    full_path_name = line[0][0]
    parts = full_path_name.split('/')
    exp = parts[2]
    modelname_specs = exp.split('_i')
    modelname = modelname_specs[0]
    specs = modelname_specs[1].split('_')  
    all_keys.append('modelname') 
    all_vals.append(modelname)         
    for k in range(0,10,2):
        all_keys.append(specs[k])
        all_vals.append(specs[k+1])                
    all_keys.append('acc')        
    if 'Robust' in mytext:
        acc = mytext.split(':')                
        all_vals.append(acc[1])
        #print(acc[1])    
    else:
        #non_finished_list.append(line[0][0])
        all_vals.append('-1')
    #print(all_keys)
    #print(all_vals)              
    res = {all_keys[i]: all_vals[i] for i in range(len(all_keys))}      
    fulldict.append(res)
    #print()
    #parts = line[0].split(',')   
    #print(parts[0])
    #print(parts[1])
    #print(line)
# for h in non_finished_list:  
#     print(h)
# print(last_lines)
#print(filename_last_lines)
print(fulldict)
import pandas as pd

# Initialize data of lists
#data = [{'b': 2, 'c': 3}, {'a': 10, 'b': 20, 'c': 30}]
  
# Creates pandas DataFrame by passing
# Lists of dictionaries and row index.
df = pd.DataFrame(fulldict)
  
# Print the data
#print(df)
df.to_csv('out.csv') 

#table1
#ps = 16
table1 = df.loc[df['ps'] == '16']
table1.to_csv('table1.csv') 
print(table1)


#table2
#mp = 5
table2 = df.loc[df['mp'] == '5']
table2.to_csv('table2.csv') 

