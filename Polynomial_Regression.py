#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 20:15:54 2021

@author: nt4-nani
"""

# Analysis Libraries..
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from PIL import Image
import glob
# Data..
np.random.seed(2)
pageSpeeds = np.random.normal(3.0, 1.0, 100)
purchaseAmount = np.random.normal(50.0, 30.0, 100) / pageSpeeds

# Training and Test Data..
trainX = pageSpeeds[:80]
testX = pageSpeeds[80:]

trainY = purchaseAmount[:80]
testY = purchaseAmount[80:]

# A look at the data..    
plt.scatter(pageSpeeds, purchaseAmount)
#plt.show()

# The aim is to fit polynomial functions, order [1 - 4].
n = np.arange(1, 10, 1)
scores = []

for i in n:

     x = np.array(trainX)
     y = np.array(trainY)    
     p4 = np.poly1d(np.polyfit(x, y, i))
     
    
     xp = np.linspace(0, 7, 100)
     ax = plt.axes()
     ax.set_xlim([0, 7])
     ax.set_ylim([0, 200])
     plt.scatter(x, y, c = 'g') # Training data..
     plt.plot(xp, p4(xp), c = 'r') # Model to be trained..
     plt.xlabel('PageSpeeds', fontsize=10)
     plt.ylabel('PurchaseAmount', fontsize=10)
     r3 = r2_score(np.array(trainY), p4(np.array(trainX)))
     plt.text(3.0, 151.0, 'Train Data: r ='+str(round(r3, 2)), bbox=dict(fill=False, edgecolor='red', linewidth=1))
     plt.text(3.0, 131.0, 'Polynomial Order ='+str(i), bbox=dict(fill=False, edgecolor='red', linewidth=1))
     plt.savefig('poly_'+str(i)+'_train_fit.png')
     plt.close()

     testx = np.array(testX)
     testy = np.array(testY)
     axes = plt.axes()
     axes.set_xlim([0, 7])
     axes.set_ylim([0, 200])
     plt.scatter(testx, testy, c='k') # Test data..
     plt.plot(xp, p4(xp), c = 'r') # Trained Model..
     plt.xlabel('PageSpeeds', fontsize=10)
     plt.ylabel('PurchaseAmount', fontsize=10)
     r2 = r2_score(testy, p4(testx))
     plt.text(3.0, 151.0, 'Test Data: r ='+str(round(r2, 2)), bbox=dict(fill=False, edgecolor='red', linewidth=1))
     plt.text(3.0, 131.0, 'Polynomial Order ='+str(i), bbox=dict(fill=False, edgecolor='red', linewidth=1))
     plt.savefig('poly_'+str(i)+'_test_fit.png')
     plt.close()

     
     scores.append(r2)
     print('Poly_n = ', i, 'Test_r = ',r2, 'Train_r = ',r3)
     print('best fit r =', np.max(scores))

# Convert images into GIF format...

frames = []

imgs = glob.glob("*.png")

for j in imgs:
     new_frame = Image.open(j)
     frames.append(new_frame)

# Save into a GIF that loops forever..

frames[0].save('png_to_gif.gif', format = 'GIF',
               append_images = frames[1:],
               save_all = True,
               duration = 1500,
               Loop = 0)







