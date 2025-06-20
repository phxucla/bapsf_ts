import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy import optimize    

def gauss_constrained(x, amplitude, offset):
    mean=29
    stddev=4
    return amplitude*np.exp(-(x - mean)**2 / (2*stddev**2))+offset

def gaussian(x, amplitude, mean, stddev, offset):
    return amplitude*np.exp(-(x - mean)**2 / (2*stddev**2))+offset


filename='scan_beam-2025-06-19.h5'
file = h5py.File(filename,'r')

# read one of the images
image = file.get("/13PICAM1:Pva1:Image/image 0")
image=np.asarray(image)
print(image.shape)

action_ds = file["actionlist/Motor12:PositionInput"]
z = np.asarray(action_ds)

dataset = file["13PICAM1:Pva1:Image"]
shots = len(dataset)

repetitions=20    # number of iterations
intensity = np.zeros(int(shots/repetitions/2), dtype=float) # save max intensity to find cosmic rays
position = np.zeros(int(shots/repetitions/2), dtype=float) # 

counter = 0
imgsum = np.zeros((511,512),dtype=float)
poscounter=0

for n in range(0,shots,2):   
    print(f"n={n}, poscounter={poscounter}, counter={counter}, z={z[n]:.3f} cm")
 
    image = file.get(f"/13PICAM1:Pva1:Image/image {n}") #n+1
    image=np.asarray(image)
    image = image.astype(float)

    bg = file.get(f"/13PICAM1:Pva1:Image/image {n+1}") #n
    bg = np.asarray(bg)
    bg = bg.astype(float)
   
    subtracted = image-bg
    
    if (counter < 50):
        imgsum += subtracted
        counter+=1 

    if (counter == 20):
        imgsum/=20    # normalize per shot
        intensity[poscounter] = np.sum(imgsum[:,:])
        position[poscounter] = z[n]   # save one of the many z
        
        print('loop done')
        #plt.imshow(np.clip(imgsum,-1,1))
        #plt.show()
        
        profile=np.zeros(64,dtype=float)
        for i in range(64):
            profile[i] = np.sum(imgsum[:,i*8:i*8+8])
        
        #offset1=np.mean(profile[0:10])
        #offset2=np.mean(profile[55:63]) 
        #profile-=np.mean([offset1, offset2])
    
    
        
        pix = np.arange(64)
        mask=np.full(64,True)
        mask[26:32]=False
        plt.plot(profile)
        plt.title("{:.3f} mm , {:.0f} counts".format(z[n], profile.sum()))
        plt.xlabel('wavelength (pixel)')
        plt.ylabel('intensity (counts/bin/shot)')
        #plt.ylim(-1000,3000)
        
        popt, _ = optimize.curve_fit(gauss_constrained,pix[mask],profile[mask],p0=[1000,0])
        Gauss = gauss_constrained(pix, *popt)
        plt.plot(pix, Gauss,color='red')
        
        intensity[poscounter] = np.sum(Gauss)

        
        plt.show()
        
        imgsum = np.zeros((511,512),dtype=float)    # reset image
        counter = 0 # reset loop counter
        poscounter+=1
        

file.close()


plt.figure(figsize=(6,4),dpi=300)
plt.plot(position,intensity,"o",color='black',linewidth=3)
plt.xlabel('z (mm)')
plt.ylabel('intensity (counts)')
plt.title(filename)

popt, _ = optimize.curve_fit(gaussian,position,intensity,p0=[40000, 3.25, 0.2,0])


detailedz=np.arange(1000)/999*4+2
Gauss = gaussian(detailedz, *popt)
plt.plot(detailedz,Gauss,color="red",linewidth=3)


Gauss -= np.average(Gauss[0:200])

print('fwhm: ',2.355*popt[2])
print(f"center: {popt[1]:.3f} cm")
print('area under Gauss /1E6: ', np.sum(Gauss)/1E6)


plt.xlim(min(z),max(z))
plt.plot(position,intensity,color='black',linewidth=3)
plt.show()

