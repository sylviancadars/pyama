import json

filename = 'input.json'
with open(filename) as f:
    input_dict = json.load(f)

print(input_dict)

