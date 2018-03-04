#!/usr/bin/python

import requests

payload = {
    'point_x': '',
    'point_y': ''
}

headers = {}
headers['User-Agent'] = 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:53.0) Gecko/20100101 Firefox/53.0'

res = requests.post("https://127.0.0.1:18888/post", data=payload)

print(res.text)