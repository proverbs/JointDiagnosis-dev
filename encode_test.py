import base64
import json
import numpy as np

file_path = './mm.jpg'
json_path = './mm.json'


with open(file_path, 'rb') as img:
    byte_content = img.read()
print('Origin Type:', type(byte_content))
base64_bytes = base64.b64encode(byte_content)

base64_string = base64_bytes.decode('utf-8')
print(base64_string)

raw_data = {}
raw_data['name'] = 'mm.jpg'
raw_data['image'] = base64_string

json_data = json.dumps(raw_data, indent=2)

with open(json_path, 'w') as json_file:
    json_file.write(json_data)


######################################################

with open(json_path, 'r') as json_file:
    raw_data = json.load(json_file)

base64_string = raw_data['image']

byte_content = base64.b64decode(base64_string)
print('Type:', type(byte_content))

with open('./mm-m.jpg', 'wb') as img:
    img.write(byte_content)

