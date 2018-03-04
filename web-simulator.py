#!/usr/bin/python

import requests
import base64

payload = {
    'point_x': ['100', ],
    'point_y': ['200'],
    'resolution':['1920x1080', ],
    'user_id':['10000', ],
    'delay': ['2500', ], # >500, activate save func
    'device': ['MACHENIKE']
}

headers = {}
headers['User-Agent'] = 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:53.0) Gecko/20100101 Firefox/53.0'


# loop
file_path = './pic-stream/t1/mid-2.jpg'
with open(file_path, 'rb') as img:
    byte_content = img.read()

base64_bytes = base64.b64encode(byte_content)
base64_string = base64_bytes.decode('utf-8')

payload['base64_image'] = ['base64,' + base64_string, ]

# res = requests.get('https://127.0.0.1:18888/post', verify=True)
res = requests.post("https://127.0.0.1:18888/post", data=payload, verify=False)

print(res.text)