import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize    


#def __array__(self, dtype=None, copy=True):
#    arr = np.asarray(self.data, dtype=dtype)
#    if copy:
#        arr = arr.copy()
#    return arr

filename='1stts500shot-2025-06-25.h5'
filename='hotter500shot-2025-06-25_1.h5'
filename='1kG_delay25ms_500shot-2025-06-25.h5'
file = h5py.File(filename,'r')

# read one of the images
image = file.get("/13PICAM1:Pva1:Image/image 3")
image=np.array(image)
print(image)
print(image.shape)


#also get number of shots
dataset = file["13PICAM1:Pva1:Image"]
shots = len(dataset)


plt.figure(dpi=300)

profile=np.zeros(512, dtype=float)
average_profile=np.zeros(512, dtype=float)

#shots=4

for n in range(0,shots,2):
    print(n)
    bg = file.get(f"/13PICAM1:Pva1:Image/image {n}") #n
    bg = np.asarray(bg)
    bg = bg.astype(float)
    #print(bg)
    
    print(n+1)
    image = file.get(f"/13PICAM1:Pva1:Image/image {n+1}") #n+1
    image=np.asarray(image)
    image = image.astype(float)
    #print(image)
    
    
    image -= bg
    y1=0
    y2=511
    
    #print(image)

    
    
    for xx in range(512):
        profile[xx] = np.sum(image[:,xx])   # use flat bg
        
    average_profile+=profile
    
    #profile-=np.mean(profile[440:450]) # remove more bg but not at edge
    
    #intensity[n] = np.sum(profile[200:300])   # y and then x, 450-500

    
    #plt.imshow(np.clip(image[:,:],-20,20))
    #plt.title(f'shot {n}')
    #plt.plot(profile*3E-2+y2,color='white')
    #plt.plot([0,512],[y1,y1],color='white')
    #plt.plot([0,512],[y2,y2])
    #plt.show()
    


file.close()

average_profile/=(shots/2)


average_profile*=(-1) # flip


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
mostbinned_wavelength=np.arange(32)*0.019*32+523.2  #.4

binned_profile=np.zeros(256,dtype=float)
for i in range(0,256):
    binned_profile[i]=np.sum(average_profile[i*2:i*2+2])

superbinned_profile=np.zeros(128, dtype=float)
for i in range(0,128):
    superbinned_profile[i]=np.sum(binned_profile[i*2:i*2+2])

supersuperbinned_profile=np.zeros(64, dtype=float)
for i in range(0,64):
    supersuperbinned_profile[i]=np.sum(superbinned_profile[i*2:i*2+2])

mostbinned_profile=np.zeros(32, dtype=float)
for i in range(0,32):
    mostbinned_profile[i]=np.sum(supersuperbinned_profile[i*2:i*2+2])



#plt.plot(binned_wavelength, binned_profile)
plt.plot(binned_profile)
plt.show()

#plt.plot(superbinned_wavelength, superbinned_profile)
plt.plot(superbinned_profile)
plt.show()

plt.plot(supersuperbinned_profile)
plt.show()

plt.plot(mostbinned_profile)
plt.show()


plt.plot(supersuperbinned_wavelength, supersuperbinned_profile)
plt.xlabel('wavelength (nm)')

# fit Gaussian
pixel=np.arange(64)
mask = (pixel<(31-4)) | (pixel>(31+4)) 
#popt, _ = optimize.curve_fit(gaussian,supersuperbinned_wavelength[mask],supersuperbinned_profile[mask],p0=[300, 532, 4])
#Gauss = gaussian(supersuperbinned_wavelength, *popt)

#plt.plot(supersuperbinned_wavelength,Gauss,color='red',linewidth=3,label='best fit')

#print('fwhm (nm): ',popt[2]*2.355)
#print('Te in eV = ', 0.4512*popt[2]**2)

mask=np.ones(64,dtype=bool)                                
mask[27:33]=False



popt, cov = optimize.curve_fit(gaussian,supersuperbinned_wavelength[mask],supersuperbinned_profile[mask],p0=[800, 532, 6, 0])
Gauss = gaussian(supersuperbinned_wavelength, *popt)


plt.plot(supersuperbinned_wavelength, Gauss,color='red')
plt.show()


Gauss-=np.mean(Gauss[0:10]) #subract offset for density

sigmaerror=np.sqrt(cov[2][2])
print('sigma: ',popt[2])
print('error sigma: +- ',sigmaerror)
fwhm=popt[2]*2.355
print('fwhm (nm): ',fwhm)


temp=0.903*popt[2]**2


print('Te in eV = ', temp)
print('Tmax in eV = ', 0.903*(popt[2]+sigmaerror)**2)
print('Tmin in eV = ', 0.903*(popt[2]-sigmaerror)**2)
print('area under Gauss: ',np.sum(Gauss))
print('density (1E13 cm-3): ',2.98e8*np.sum(Gauss)/1e13)



  


