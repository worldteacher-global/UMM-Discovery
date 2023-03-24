# Copyright 2020 Novartis Institutes for BioMedical Research Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
import numpy as np


class bbbc021_dataset(Dataset):
    def __init__(self, root_dir, metadata, label_header, transform=None):
        #root_dir: directory with the images
        #transform: optional transfornations 
        super(bbbc021_dataset, self).__init__()
        self.df = pd.read_csv(os.path.join(root_dir, metadata), sep=',') # reads in the csv containng the image annotations "image file name, coordiantes, etc"
        # self.df = pd.read_csv(os.path.join(root_dir, metadata), sep=',') # reads in the csv containng the image annotations "image file name, coordiantes, etc"
        self.root_dir = root_dir
        self.transform = transform
        self.label_header = label_header #column in the dataset for unique classes?

        self.add_columns()

        # self.class_to_idx = dict() # creates an empty dictionary
        # self.classes = [] # creates an empty list
        # i = 0
        # for c in self.df[label_header].unique(): # gets unique classes
        #     self.classes.append(c) # creates a list of unique classes
        #     self.class_to_idx[c] = i # creates a value key pair where value is a number and key is a class
        #     i += 1

        # self.imgs = [] # creates an empty list
        # for row in range(self.df.shape[0]//2):
        #     # img_name_ch1 = os.path.join(self.root_dir, self.df['plate'].iloc[row], self.df['filename_dna'].iloc[row]) #path of a series of this channel
        #     # img_name_ch2 = os.path.join(self.root_dir, self.df['plate'].iloc[row], self.df['filename_tubulin'].iloc[row]) #path of a series of this channel
        #     # img_name_ch3 = os.path.join(self.root_dir, self.df['plate'].iloc[row], self.df['filename_actin'].iloc[row]) #path of a series of this channel
        #     img_name_ch1 = os.path.join(self.df['image_root_folder'].iloc[row],self.df[self.df.ChannelID.eq('2')]['URL'].iloc[row])
        #     img_name_ch2 = os.path.join(self.df['image_root_folder'].iloc[row],self.df[self.df.ChannelID.eq('1')]['URL'].iloc[row])

        #     label = self.class_to_idx[self.df[label_header].iloc[row]] #dictionary key (unique class)
        #     # item = (img_name_ch1, img_name_ch2, img_name_ch3, label) #tuple of image channels and the key(unique class)
        #     item = (img_name_ch1, img_name_ch2, label) #tuple of image channels and the key(unique class)
        #     self.imgs.append(item) #add to the image list

    # def __getitem__(self, index): #reads in the images | This is memory efficient because all the images are not stored in the memory at once but read as required
    #     # img_name_ch1, img_name_ch2, img_name_ch3, label = self.imgs[index]
    #     img_name_ch1, img_name_ch2, label = self.imgs[index]
    #     img1 = np.asarray(Image.open(img_name_ch1))
    #     img2 = np.asarray(Image.open(img_name_ch2))
    #     # img3 = np.asarray(Image.open(img_name_ch3))
    #     # image = np.zeros((img1.shape[0], img1.shape[1], 3), dtype='float')
    #     image = np.zeros((img1.shape[0], img1.shape[1], 2), dtype='float')
    #     image[:,:,0] = img1.copy()
    #     image[:,:,1] = img2.copy()
    #     # image[:,:,2] = img3.copy()

    #     if self.transform:
    #         image = self.transform(image)

    #     return image, label

    # def __len__(self):
    #     return len(self.df)

    def add_columns(self):
        rel_dose_list = list()
        for i, row in self.df.iterrows():
            compound = self.df.loc[i, 'compound']
            if (compound == 'POS-000001'):
                rel_dose_list.append(0)
            else:
                d = self.df.loc[i, 'compound_uM'] / max(self.df.loc[self.df['compound'] == compound, 'compound_uM'])
                rel_dose_list.append(d)
        self.df['rel_dose_adjust'] = rel_dose_list
        self.df['compound'] = self.df['compound'].str.replace(',', '.')
        self.df['batch'] = 'Batch_' + self.df['table_nr'].map(str)

    # def get_df(self):
    #     return self.df
