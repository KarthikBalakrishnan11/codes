import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches

# read the csv file using read_csv function of pandas
train = pd.read_csv('train.csv')
train.head()


data = pd.DataFrame()
data['format'] = train['image_names']

# as the images are in train_images folder, add train_images before the image name
for i in range(data.shape[0]):
    data['format'][i] = 'Data_set/train_images/' + data['format'][i]

# add xmin, ymin, xmax, ymax and class as per the format required
for i in range(data.shape[0]):
    data['format'][i] = data['format'][i] + ',' + str(train['xmin'][i]) + ',' + str(train['ymin'][i]) + ',' + str(train['xmax'][i]) + ',' + str(train['ymax'][i]) + ',' + train['helmet_color'][i]

data.to_csv('annotate.txt', header=None, index=None, sep=' ')