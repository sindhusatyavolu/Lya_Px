import numpy as np
from config import *

def get_skewers(wave_fft_grid,mask_fft_grid,file,verbose=False):
    skewers=[]
    for hdu in file[1:]:
        if len(skewers)%100==0 and verbose:
            print(len(skewers),'read')
        # quasar meta data
        z_qso=hdu.header['Z']
        # angles in radians
        RA=hdu.header['RA']
        Dec=hdu.header['DEC']
        skewer={'z_qso':z_qso, 'RA':RA, 'Dec':Dec}
        # forest data
        wave_data=10.0**(hdu.data['LOGLAM'])
        delta_data=hdu.data['DELTA']
        weight_data=hdu.data['WEIGHT']
        # correct the weights so that they are constant in redshift (for the raw analysis only)
        weight_data *= (wave_data/4500)**3.8
        skewer['wave_data']=wave_data
        skewer['delta_data']=delta_data
        skewer['weight_data']=weight_data

        # Map the observed spectrum to the FFT grid for this particular redshift bin    
        j_min_data=round((wave_data[0]-wave_fft_grid[0])/pw_A)
        j_max_data=round((wave_data[-1]-wave_fft_grid[0])/pw_A)
        #print(j_min_data,j_max_data)

        # map the data deltas and weights into the FFT grid
        delta_fft_grid=np.zeros(N_fft)
        weight_fft_grid=np.zeros(N_fft)
        
        # have to make sure FFT grid is larger than the data grid that falls within the redshift bin
        if j_min_data < 0 and j_max_data >= N_fft:
            print('Data grid is larger than FFT grid, increase N_fft')
            exit(1)

        # figure out whether the spectrum is cut at low-z or at high-z
        loz_cut=False
        hiz_cut=False
        if j_min_data < 0:
            loz_cut=True
            if j_max_data >=0:
                delta_fft_grid[:j_max_data]=delta_data[-j_min_data+1:]
                weight_fft_grid[:j_max_data]=weight_data[-j_min_data+1:]
        if j_max_data >= N_fft:
            hiz_cut=True
            if j_min_data < N_fft:
                delta_fft_grid[j_min_data:]=delta_data[:N_fft-j_max_data-1]
                weight_fft_grid[j_min_data:]=weight_data[:N_fft-j_max_data-1]
        if loz_cut==False and hiz_cut==False:
            delta_fft_grid[j_min_data:j_max_data+1]=delta_data
            weight_fft_grid[j_min_data:j_max_data+1]=weight_data
    
        # limit the data to the pixels within this redshift bin
        weight_fft_grid *= mask_fft_grid
        
        # store relevant information
        skewer['j_min_data']=j_min_data
        skewer['j_max_data']=j_max_data
        skewer['delta_fft_grid']=delta_fft_grid
        skewer['weight_fft_grid']=weight_fft_grid
        skewers.append(skewer)
    return skewers


def get_separations(skewers):
    separation_angles=[]
    for i in range(len(skewers)):
        for j in range(i+1,len(skewers)):
            separation=angular_separation(skewers[i]['RA'],skewers[i]['Dec'],skewers[j]['RA'],skewers[j]['Dec'])
            separation_angles.append(separation)
    return 0


