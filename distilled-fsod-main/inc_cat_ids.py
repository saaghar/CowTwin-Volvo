import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('src')

args = parser.parse_args()

if not os.path.isfile(args.src):
    raise FileNotFoundError('src not found')


with open(args.src, 'r') as file:
    data = json.load(file)


for annotation in data['annotations']:
    annotation['category_id'] += 1

for category in data['categories']:
    category['id'] += 1


dst = os.path.splitext(args.src)[0] + "_inc.json"

with open(dst, 'w') as dst_file:
    json.dump(data, dst_file)

