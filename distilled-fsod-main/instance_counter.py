import os
import sys

dir_name: str
if sys.argv[1].lower() == 'joab':
    dir_name = 'JOAB'
elif sys.argv[1].lower() == 'public':
    dir_name = 'Public'
else:
    raise ValueError("Environment name was not JOAB or Public")

dir_path = os.path.expanduser('~/OneDrive - Volvo Group/'
        'General - AI sharing of public data/Labels/'
        f'{dir_name}/Truck-Accessories-PascalVOC-export/ImageSets/Main/')

file_names = os.listdir(dir_path)
counts = {}
max_len = 0
for name in file_names:
    file_path = dir_path + name
    with open(file_path, 'r') as file:
        key = name.split('.')[0]
        max_len = max(len(key), max_len)
        counts[key] = 0
        for line in file.readlines():
            counts[key] += 1 if int(line.split(' ')[-1]) == 1 else 0

for k, v in counts.items():
    print(f'{k}{' '*(max_len-len(k))}\t{v:>2}')
