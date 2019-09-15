import os, sys, random
import xml.etree.ElementTree as ET
from glob import glob
import pandas as pd
from shutil import copyfile

annotations = glob('/home/a1036006/Karthik/Helmet_frcnn/keras-frcnn/Data_set/annots/*.xml')
df = []
cnt = 0

for file in annotations:
    image_names = file.split('/')[-1].split('.')[0] + '.jpg'
    #filename = str(cnt) + '.jpg'
    row = []
    parsedXML = ET.parse(file)
    for node in parsedXML.getroot().iter('object'):
        helmet_color = node.find('name').text
        xmin = int(node.find('bndbox/xmin').text)
        xmax = int(node.find('bndbox/xmax').text)
        ymin = int(node.find('bndbox/ymin').text)
        ymax = int(node.find('bndbox/ymax').text)
        row = [image_names, helmet_color, xmin, xmax,ymin, ymax]
        print(row)
        df.append(row)


data = pd.DataFrame(df, columns=['image_names', 'helmet_color',
'xmin', 'xmax', 'ymin', 'ymax'])

data[['image_names', 'helmet_color', 'xmin', 'xmax', 'ymin', 'ymax']].to_csv('helmet_detection.csv', index=False)