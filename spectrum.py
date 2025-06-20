import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize    



filename='ts-2025-06-19.h5'
file = h5py.File(filename,'r')

# read one of the images
image = file.get("/13PICAM1:Pva1:Image/image 0")
image=np.array(image)
print(image.shape)


#also get number of shots
dataset = file["13PICAM1:Pva1:Image"]
shots = len(dataset)


plt.figure(dpi=300)

profile=np.zeros(512, dtype=float)
average_profile=np.zeros(512, dtype=float)

for n in range(0,shots,2):
    bg = file.get(f"/13PICAM1:Pva1:Image/image {n}") #n
    bg = np.asarray(bg)
    bg = bg.astype(float)
    
    image = file.get(f"/13PICAM1:Pva1:Image/image {n+1}") #n+1
    image=np.asarray(image)
    image = image.astype(float)
    
    
    image -= bg
    y1=215
    y2=375
    
    
    for xx in range(512):
        profile[xx] = np.sum(image[y1:y2,xx])   # use flat bg
        
    average_profile+=profile
    
    #profile-=np.mean(profile[440:450]) # remove more bg but not at edge
    
    #intensity[n] = np.sum(profile[200:300])   # y and then x, 450-500

    
    plt.imshow(np.clip(image[:,:],-10,10))
    plt.title(f'shot {n}')
    plt.plot(profile*3E-2+y2,color='white')
    plt.plot([0,512],[y1,y1],color='white')
    plt.plot([0,512],[y2,y2])
    plt.show()
    


file.close()

average_profile/=(shots/4)

#average_profile[220:258]=0

# use old wavelength cal
wavelength = np.arange(512)* 19.80636 / 511 + 522.918


plt.figure(dpi=300)
plt.plot(wavelength, average_profile)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity (counts/bin/shot)')
plt.xticks([524, 526,528, 530, 532, 534,536, 538,540,542])
plt.title(filename)
plt.show()


def gaussian(x, amplitude, mean, stddev, offset):
    return amplitude*np.exp(-(x - mean)**2 / (2*stddev**2))+offset


binned_wavelength=np.arange(256)*0.019*4+523.12  #.4
superbinned_wavelength=np.arange(128)*0.019*8+523.0  #.4
supersuperbinned_wavelength=np.arange(64)*0.019*16+523.2  #.4

binned_profile=np.zeros(256,dtype=float)
for i in range(0,256):
    binned_profile[i]=np.sum(average_profile[i*2:i*2+2])

superbinned_profile=np.zeros(128, dtype=float)
for i in range(0,128):
    superbinned_profile[i]=np.sum(binned_profile[i*2:i*2+2])

supersuperbinned_profile=np.zeros(64, dtype=float)
for i in range(0,64):
    supersuperbinned_profile[i]=np.sum(superbinned_profile[i*2:i*2+2])


plt.plot(binned_wavelength, binned_profile)
plt.show()

plt.plot(superbinned_wavelength, superbinned_profile)
plt.show()



np.savez('spectrum.npz', wavelength=wavelength,intensity=average_profile)
  


