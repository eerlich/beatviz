import sys # for command line arguments

import numpy as np
import scipy as sp

# IO 
import scipy.io.wavfile
import scipy.ndimage
import imageio
#import skvideo as vid 
#import skvideo.io 

# random offsets for colors
#offsets_preset=np.random.rand(3,2)

# create unit vectors in three directions 
offsets_preset=np.array( [[np.sqrt(3.0)/2.0,0.5], \
        [-np.sqrt(3.0)/2,0.5], [0.0,-1.0]] )

offsets_preset=offsets_preset

def offset_colors(img,amount):
    # random offset for each channel
    #offsets=(np.random.rand(3,2)-0.5)*amount
    offsets=(offsets_preset-0.5)*amount
    #offsets=(offsets_preset)*amount
    offsets=np.floor(offsets)
    out=img.copy()

    offsets=np.int8(offsets)
    (xpad,ypad)=np.max(np.abs(offsets),0)

    out2=np.pad(out,((xpad,xpad),(ypad,ypad),(0,0)),mode='reflect')
    xsize=img.shape[0]
    ysize=img.shape[1]

    #colors=(0,1,2) # leave green out!
    colors=(0,1,2) # leave green out!

    for k in colors:
        tmp=out2[xpad+1+offsets[k,0]:,ypad+1+offsets[k,1]:,k]
        out[:,:,k]=tmp[:xsize,:ysize]

    return out

# read input audio and image 
audiofile=sys.argv[1]
imgfile=sys.argv[2]

# NOTE: this function is a bit picky about the data type; a 16 bit 
# wav-file works
(Fs,audio)=sp.io.wavfile.read(audiofile)
# work in mono, at least for now; also convert to floating point
audio=0.5*audio[:,0]+0.5*audio[:,1]
nsamples=len(audio)

img=sp.ndimage.imread(imgfile)

# initialize video writer object. we will inherit the frame size
# from the input image
#framesize=img[:,:,0].shape
framesize=img[:,:,0].shape
framesize=framesize[::-1]
fname='./output.mp4'
fps=30
writer=imageio.get_writer(fname,fps=fps)

# calculate number of frames 
N=long(np.ceil(1.0*fps*nsamples/Fs))
#N=200 # test with smaller pieces
winsize=1.0*Fs/fps
overlap=0.5*winsize

prev=img

fftsize=36
maxfreq=Fs/2
bassrange=(90,110) #Hz 
bassmin=0
bassmax=100

## generate frames and output them to video file
bass=0.0
bass_prev=0.0
bass_new=0.0

for k in range(N):

    ## extract information from audio data in windows
    # Fourier transform:
    win=audio[max(0,k*winsize-overlap):min(nsamples,(k+1)*winsize+overlap)]
    # take amplitude only
    spectrum=np.abs(np.fft.fft(win,maxfreq))

    # feature 1: bass amplitude
    #spectrum=np.fft.fftshift(spectrum)
    bass_prev=bass_new
    bass_new=0.00001*np.mean(spectrum[range(bassrange[0],bassrange[1])])

    bass_new=max(bass_new,bassmin)
    bass_new=min(bass_new,bassmax)
    bass=bass_new
    bass=bass-bassmin

    # feature 2: RMS level
    rms=np.sqrt(np.mean(win**2))
    rms=0.005*rms # scale
    
    ## apply audio-dependent FX (TODO: put these in a function)
    # 1. color channel offset
    frame=offset_colors(img,bass*0.1)

    # 2. blur (SLOW)
    frame=sp.ndimage.gaussian_filter(frame,(rms*0.05,\
            rms*0.05,0.0))

    # 3. pincushion distortion (probably very slow. could try
    # memoizing mappings?)

    ## TODO: overlay optional "screen layer"

    #prev=frame
    writer.append_data(frame)
    if k%10==0: print "%d/%d"%(k,N)

writer.close()
