# importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches

# read the csv file using read_csv function of pandas
train = pd.read_csv('train.csv')
train.head()

# reading single image using imread function of matplotlib
#image = plt.imread('/home/a1036006/Karthik/Helmet_frcnn/keras-frcnn/Data_set/train_images/00000.jpg')
#plt.imshow(image)

# Number of unique training images
train['image_names'].nunique()

# Number of classes
train['helmet_color'].value_counts()

fig = plt.figure()

#add axes to the image
ax = fig.add_axes([0,0,1,1])

# read and plot the image
image = plt.imread('/home/a1036006/Karthik/Helmet_frcnn/keras-frcnn/Data_set/train_images/00005.jpg')
plt.imshow(image)

# iterating over the image for different objects
for _,row in train[train.image_names == "00005.jpg"].iterrows():
    xmin = row.xmin
    xmax = row.xmax
    ymin = row.ymin
    ymax = row.ymax
    
    width = xmax - xmin
    height = ymax - ymin
    
    # assign different color to different classes of objects
    '''  if row.helmet_color == 'RBC':
        edgecolor = 'r'
        ax.annotate('RBC', xy=(xmax-40,ymin+20))
    elif row.cell_type == 'WBC':
        edgecolor = 'b'
        ax.annotate('WBC', xy=(xmax-40,ymin+20))
    elif row.cell_type == 'Platelets':
        edgecolor = 'g'
        ax.annotate('Platelets', xy=(xmax-40,ymin+20))'''
        
    # add bounding boxes to the image
    rect = patches.Rectangle((xmin,ymin), width, height, edgecolor = 'w', facecolor = 'none')
    
    ax.add_patch(rect)