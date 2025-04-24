# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 17:02:07 2025

@author: manid
"""

import torch
import copy
import matplotlib.pyplot as plt


#to see which is the first point that is close enough to the 'all 0' goal
#input: difference between current pixel alignment and map_
def first_occurence(x):
    s=torch.sqrt(torch.sum(x**2, dim=-1))  #-1 is across the features/channels
    return s
    
   
def distance_between_vectors(v1, v2):
    return torch.sum((v1-v2)**2).item()

def mean_shift(image, r):
    
    b,c,h,w=image.size()
    pixel_features=torch.reshape(image, (b,c,h*w)).transpose(dim0=-1, dim1=-2)  #not adding additional features for now
        #shape is (b,h*w,c)
        #VERIFIED LINE
    og_pixel_features=copy.deepcopy(pixel_features)
    
    #VERIFIED FOR LOOP
    window_radius=r
    first_index=0
    for last_index in range(h,h*w+1,h):
        #get height vector
        height_vector=og_pixel_features[:, first_index:last_index]  #takes each subset of h vectors (for each batch, batches are intact)
            #shape is (b,h,c)
        
        
        
        #find distances between every other pixel and height vector
        r_hv=height_vector.reshape(height_vector.size()[0], height_vector.size()[1], 1, height_vector.size()[-1]) #(b,h,1,c)
        all_hv=og_pixel_features.unsqueeze(dim=1) #(b,1,h*w,c)
        distances= torch.sqrt(torch.sum((r_hv-all_hv)**2, dim=-1)) #shape is (b,h,h*w) 
        
        #filter the distances to get only points within window. first create weights (0 for not counted in avg)
        weights=(torch.where(distances<window_radius, 1.0,0.0))
        #get number of pixels counted in average
        num_pix=torch.where(torch.where(distances<window_radius,1.0,-1.0)==1, 1.0,0.0)
        num_pix=(torch.count_nonzero(num_pix,dim=-1)).unsqueeze(dim=-1) #shape is (b,h,1)
        
        #get average
        mass_center=(torch.matmul(weights, og_pixel_features)).div(num_pix)
            # (b,h,h*w) * (b, h*w, c) = (b,h,c)
            # --> mass center for each pixel in the h group.
        
        #move pixel to this center.
        pixel_features[:, first_index:last_index]=mass_center
        first_index=last_index
        
        #VERIFIED EVERYTHING UPTO HERE
    return(og_pixel_features, pixel_features)
    
    
def run_mean_shift(image, window_size):
    epsilon=0.1
    b,c,h,w=image.size()
    pix=None; alignment=None
    for i in range(1): #originally 100
        pix, alignment=mean_shift(img,window_size)
    
    #get the centroids
    alignment=torch.round(alignment, decimals=0)
    torch.unique(alignment)
    
    #assign the centroids to the classes
    map_=torch.randn((b,1,c))*9999 #1 in middle dimension for now, we will add to it.
    #*9999 as not to confuse the rest of program
    classes=torch.randn((b,1))*9999
    map_func=torch.vmap(first_occurence, 0) #apply to each tensor in batch, so dimension=0
    for i in range(h*w):
        #get each pixel alignment
        each_pixel=alignment[:, i].reshape((b,1,c)) 
        #put in hashmap
        map_=torch.cat([map_, each_pixel], dim=1)   
        #find index
        input_to_vmap=torch.abs(map_-each_pixel) #so that it's close to all 0's. this tensor gives the differences
        
        #between the current pixel's alignment and each one in the map
        
        class_lbl=torch.argmax(1-map_func(input_to_vmap), dim=1).unsqueeze(dim=1)
        classes=torch.cat([classes, class_lbl], dim=1)
    
    return classes[:,1:].int() # is (b,h*w)
     
    

# img=torch.ones((1,3,64,64))
# img2=torch.ones((1,3,64,64))+32
# img3=torch.ones((1,3,64,64))+64
# img4=torch.ones((1,3,64,64))+128
# img=(torch.cat([torch.cat([img,img2],dim=2), torch.cat([img3,img4], dim=2)], dim=-1))

img=torch.ones((1,3,64,128))
img2=torch.ones((1,3,64,128))+32
img=(torch.cat([torch.cat([img,img2],dim=2)]))


b,c,h,w=img.size()
window_size=5
res_img=run_mean_shift(img, window_size)

#number of classes
f=torch.unique(res_img)
print(f)

plt.imshow((img.squeeze(dim=0)).transpose(dim0=0, dim1=1).transpose(dim0=1,dim1=2).int()
           )
plt.show()
plt.imshow((res_img.squeeze(dim=0)).reshape(b,h,w)
           .transpose(dim0=0, dim1=1).transpose(dim0=1,dim1=2), cmap='gray', vmin=0, vmax=255)
plt.show()

#am also counting the number of pix to average wrong
