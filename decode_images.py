import os
import json
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import glob

if not os.path.exists('output'):
    os.makedirs('output')

for n, fname in enumerate(glob.glob('dataset/valid/*.json')):
    with open(fname) as f:
        data = json.load(f)

    decode_str = base64.b64decode(data['imageData'])
    img = Image.open(BytesIO(decode_str))
    img = img.convert('RGB')
    img.save('output/%d.jpg'%n)
