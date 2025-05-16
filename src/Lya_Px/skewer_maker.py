import numpy as np


class SkewerMaker(object):
    """Object to generate random skewers"""


    def __init__(self,N,L,input_power,SiIII_model=None,seed=1234):
        """Define here FFT grid and input power"""

        # specify FFT grid
        self.N=N
        self.L=L
        # pixel width 
        self.dx = L/N
        # Wavenumbers for the generated Fourier modes, specified by the grid length and resolution
        self.k = np.fft.fftfreq(N)*2*np.pi/self.dx

        # use input power object to specify amplitude of *discrete* Fourier modes
        norm = (N/self.dx)
        self.inP=norm*input_power.get_true_p1d(np.abs(self.k))

        # optionally add SiIII(1207) contamination to the skewers
        self.SiIII_model=SiIII_model
    
        # ratio of PX over P1D (if making pairs of skewers)
        self.f_px=input_power.f_px

        # store input seed in case we need to reset it later
        self.seed=seed
        # setup random number generator using seed
        self.gen = np.random.default_rng(seed)


    def reset_seed(self):
        # setup random number generator using seed
        self.gen = np.random.default_rng(self.seed)
        

    def make_gaussian_modes(self):
        """Generate Gaussian random numbers for the Fourier modes of the grid."""

        N=self.N
        P=self.inP
        # generate random Fourier modes
        modes = np.empty(N, dtype=complex)
        modes[:].real = self.gen.normal(size=N)
        modes[:].imag = self.gen.normal(size=N)
        
        # normalize to desired power and enforce real for i=0, i=(N+1)//2
        modes[0] = modes[0].real * np.sqrt(P[0])
        modes[(N+1)//2] = modes[(N+1)//2].real * np.sqrt(P[(N+1)//2])
        modes[1:(N+1)//2] *= np.sqrt(0.5*P[1:(N+1)//2])    

        # remember that F(-k) = F^*(k)
        modes[(N+1)//2+1:]=np.conj(modes[1:(N+1)//2])[::-1]

        # add contamination by SiIII(1207) to the Fourier modes
        if self.SiIII_model:
            modes *= self.SiIII_model.mode_contamination(self.k)
        
        return modes


    def make_skewer(self):
        """Generate random Gaussian skewer"""

        modes=self.make_gaussian_modes()

        return np.fft.ifft(modes).real


    def make_skewer_pair(self):
        """Generate pair of random Gaussian skewers"""

        # generate a pair of (uncorrelated) set of Fourier modes
        modes_A=self.make_gaussian_modes()
        modes_C=self.make_gaussian_modes()

        # correlate modes, taking into account PX = f_px P1D
        modes_B = self.f_px * modes_A + np.sqrt(1-self.f_px**2) * modes_C

        # iFFT to get skewers
        skewer_A=np.fft.ifft(modes_A).real
        skewer_B=np.fft.ifft(modes_B).real

        return skewer_A, skewer_B
