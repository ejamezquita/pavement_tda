# # Compute Betti curves out of 2D images
# 
# Given a 2D image of pavement cells, we can filter it by cell area and compute 
# its Betti curves to topologically describe the shape of the pattern. 
# 
# To compute all Betti curves of all images at once, 
# run this script from the terminal as
#
#      python3 02_betti_curves.py
#
# It assumes the following folder structure:
#
# main
#  |
#  |-- jupyter: jupyter notebooks and py files
#  |
#  |-- data
#  |     |
#  |     |-- <tissue/sample name>
#  |     |         |
#  |     |         |-- <name>_data_membrane.csv : labels and associated measurables
#  |     |         |-- <name>.tiff : TIFF 2D image with cells
#  |     |
#  |     
#  |     
#  |-- results : where betti curves data and diagnostic pictures will be saved
#                subfolders will be created automatically following the same
#                names as in `data`

import glob
import os
import tifffile as tf
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage

import matplotlib.pyplot as plt

# Function to count the number of connected components and holes in a 2D image
#
# INPUT:
#     `img` : a 2D numpy array of an image
#     `structure`: connectivity structure to compute both connected components and holes. 
#                  It can be either 4N or 8N
#                  See https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generate_binary_structure.html
#
# OUTPUT:
#     `betti0`: Number of connected components
#     `betti1`: Number of holes

def betti0_betti1(cimg, structure):
    
    # The image is binarized
    img = cimg.copy()
    img[img > 0] = 1 
    
    # Count the number of connected components
    _ , betti0 = ndimage.label(img, structure)
    
    # Next fill in all the holes.
    # Then compared the original image to the one filled
    # The difference are the holes
    # Count the connected components of the difference ie. count the individual holes
    fimg = img.copy()
    fimg = ndimage.binary_fill_holes(fimg, struc)
    fimg = fimg - img
        
    _ , betti1 = ndimage.label(fimg, structure)
    
    return [betti0,betti1]

fs=20
TT = 32

filesrcs = sorted(glob.glob('../data/*/'))

for sidx in range(len(filesrcs):
    
    src = filesrcs[sidx]
    dst = '../results/' + src.split('/')[-2] + '/'
    print(dst)
    if not os.path.isdir(dst):
        os.makedirs(dst)

    tifffiles = sorted(glob.glob(src + '*.tiff') + glob.glob(src + '*.tif')) 
    for tidx in range(1,len(tifffiles)):
        
        tifffile = tifffiles[tidx]
        _,filename = os.path.split(tifffile)
        bname, _ = os.path.splitext(filename)

        img = tf.imread(tifffile)
        print(filename, img.min(), img.max(), img.shape, sep='\t')

        datafile = src + bname + '_data_membrane.csv'
        data = pd.read_csv(datafile)

        imglabels = np.unique(img)
        datalabels = np.unique(data['Label'])
        missing = set(imglabels) - set(datalabels)
        print(missing)

        for mislab in missing:
            img[img == mislab] = 0


        # Consider the interval `(area_min, area_max)` and split it into `TT = 32` equispaced thresholds.
        # 
        # We will then keep track of the number of connected components and holes at each of these thresholds.

        # In[47]:

        filterval = 'Geometry/Area'
        #filterval = 'Lobeyness/Circularity'
        arearange = np.linspace(np.min(data[filterval]), np.max(data[filterval]), TT)
        digis = np.digitize(data[filterval].values, arearange)

        arange = np.zeros(len(arearange)+1)
        arange[:-1] = arearange
        arange[-1] = arange[-2]
        arange[np.linspace(0,TT,TT//4+1).astype(int)]

        print(len(np.unique(digis)))
        print(digis.max())
        print(digis.min())


        # Make a copy of the TIFF and relabel it based on the interval from above.

        limg = img.copy()
        for i in range(len(data)):
            limg[ limg == data['Label'].iloc[i] ] = digis[i]
        limg[limg == 0] = 1


        print(filename, limg.min(), limg.max(), limg.shape, sep='\t')


        # Make sure we can visualize the TIFF


        fig, ax = plt.subplots(1,1,figsize=(10,10))
        ax.imshow(limg, origin='lower', cmap='plasma');
        ax.axis('off');

        filename = dst + 'pavement_' + filterval.replace('/','_').lower() + '_plasma'.format(i)
        plt.savefig(filename, format='jpg', dpi=300, bbox_inches='tight', pil_kwargs={'optimize':True})
        plt.close()


        # There are two basic ways to filter the image:
        # - **Bottom up**: We start with nothing, then we add the smallest cells, then the small cells, next medium-sized cells, and so on.
        #     - We see a bunch of connected components at first
        #     - These components start to merge and their number decreases
        #     - Holes appear at some point
        #     - These holes are filled in by the end
        # 
        # - **Top bottom**: We start with nothing, then we add the largest cells, then the large cells, next medium-sized cells, and so on.
        #     - We start with few connected components
        #     - These start to increase, and later merge
        #     - Tiny holes start to appear and their number increases as time goes on
        #     
        # For either case, we use 8N-connectivity. That is, each pixel has 8 neighbors. The alternative is 4N, where pixels are only neighbors if they share a full side. See [https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generate_binary_structure.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generate_binary_structure.html)
        # 
        
        
        # TOP BOTTOM
        # Compute Betti numbers

        struc = ndimage.generate_binary_structure(img.ndim, 2)
        betti = np.zeros((TT+1,img.ndim), dtype=np.uint)

        fimg = TT + 1 - limg.copy()

        for i in range(len(betti)):    
            fimg[fimg > TT - i] = 0
            betti[-i-1] = betti0_betti1(fimg, struc)
            print(i, -i-1, len(betti) -i-1, betti[-i-1], sep='\t')
            
        filename = dst + 'betti_' + bname + '_top_down.csv'
        pd.DataFrame(betti, columns=['betti0', 'betti1']).to_csv(filename, index=False)


        # Plot Betti curves

        fig, ax = plt.subplots(1,1,figsize=(15,5))

        ax.plot(betti[1:,0], c='r', lw=5, label='$\\beta_0$')
        ax.plot(betti[1:,1], c='b', lw=5, label='$\\beta_1$')
        ax.legend(fontsize=fs)

        ax.set_ylabel('Cardinality', fontsize=fs)
        ax.set_xlabel(filterval, fontsize=fs)
        ax.set_xticks(np.linspace(0,TT,TT//4+1))
        ax.set_xticklabels(np.round(arange[np.linspace(0,TT,TT//4+1).astype(int)],1)[::-1]);
        ax.set_title(bname + ' : Betti curves for ' + filterval, fontsize=fs);
        ax.tick_params(labelsize=fs-5)

        filename = dst + 'betticurve_top_down_' + bname + '_' + filterval.replace('/','_').lower() + '.jpg'
        plt.savefig(filename, format='jpg', dpi=100, bbox_inches='tight', pil_kwargs={'optimize':True});
        plt.close()


        # # BOTTOM UP

        ## Compute Betti numbers


        struc = ndimage.generate_binary_structure(img.ndim, 2)
        betti = np.zeros((TT+1,img.ndim), dtype=np.uint)

        fimg = limg.copy()

        for i in range(len(betti)):
            fimg[fimg > TT - i] = 0
            betti[-i-1] = betti0_betti1(fimg, struc)
            print(i, -i-1, len(betti) -i-1, betti[-i-1], sep='\t')
            
        filename = dst + 'betti_' + bname + '_bottom_up.csv'
        pd.DataFrame(betti, columns=['betti0', 'betti1']).to_csv(filename, index=False)
        
        
        ## Plot Betti curves

        fig, ax = plt.subplots(1,1,figsize=(15,5))

        ax.plot(betti[1:,0], c='r', lw=5, label='$\\beta_0$')
        ax.plot(betti[1:,1], c='b', lw=5, label='$\\beta_1$')
        ax.legend(fontsize=fs)

        ax.set_ylabel('Cardinality', fontsize=fs)
        ax.set_xlabel(filterval, fontsize=fs)
        ax.set_xticks(np.linspace(0,TT,TT//4+1))
        ax.set_xticklabels(np.round(arange[np.linspace(0,TT,TT//4+1).astype(int)],1));
        ax.set_title(bname + ' : Betti curves for ' + filterval, fontsize=fs);
        ax.tick_params(labelsize=fs-5)

        filename = dst + 'betticurve_bottom_up_' + bname + '_' + filterval.replace('/','_').lower() + '.jpg'
        plt.savefig(filename, format='jpg', dpi=100, bbox_inches='tight', pil_kwargs={'optimize':True});
        plt.close()


