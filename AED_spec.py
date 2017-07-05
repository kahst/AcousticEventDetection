import os
import traceback
import operator

import numpy as np
import cv2

import python_speech_features as psf
import scipy.io.wavfile as wave
from scipy import interpolate

######################################################
src_dir = 'dataset/train/wav/'
spec_dir = 'dataset/train/spec/'

SPEC_LENGTH = 3 #seconds
SPEC_OVERLAP = 2 #seconds

######################################################
def getSpecSettings(seconds):

    #recommended settings for spectrogram extraction
    settings = {2:[0.015, 0.0068],
               3:[0.02, 0.00585],
               5:[0.05, 0.0097],
               10:[0.05, 0.0195],
               30:[0.05, 0.0585]}

    winlen = settings[seconds][0]
    winstep = settings[seconds][1]

    nfft = 511

    return winlen, winstep, nfft
    

def changeSampleRate(sig, rate):

    duration = sig.shape[0] / rate

    time_old  = np.linspace(0, duration, sig.shape[0])
    time_new  = np.linspace(0, duration, int(sig.shape[0] * 44100 / rate))

    interpolator = interpolate.interp1d(time_old, sig.T)
    new_audio = interpolator(time_new).T

    sig = np.round(new_audio).astype(sig.dtype)
    
    return sig, 44100

def getSpecFromSignal(sig, rate, seconds=SPEC_LENGTH):

    #get settings
    winlen, winstep, nfft = getSpecSettings(seconds)

    #get frames
    winfunc=lambda x:np.ones((x,))
    frames = psf.sigproc.framesig(sig, winlen*rate, winstep*rate, winfunc)        
    
    #Magnitude Spectrogram
    magspec = np.rot90(psf.sigproc.magspec(frames, nfft))

    #normalize to values from 0 to 1
    magspec -= magspec.min(axis=None)
    magspec /= magspec.max(axis=None)        

    #adjust shape if signal is too short
    magspec = magspec[:256, :512]
    temp = np.zeros((256, 512), dtype="float32")
    temp[:magspec.shape[0], :magspec.shape[1]] = magspec
    magspec = temp.copy()
    magspec = cv2.resize(magspec, (512, 256))
    
    #DEBUG: show
    #cv2.imshow('SPEC', magspec)
    #cv2.waitKey(-1)

    return magspec

def splitSignal(sig, rate, seconds=SPEC_LENGTH, overlap=SPEC_OVERLAP):

    #split signal with ovelap
    sig_splits = []
    for i in xrange(0, len(sig), int((seconds - overlap) * rate)):
        split = sig[i:i + seconds * rate]
        if len(split) >= 1 * rate:
            sig_splits.append(split)

    #is signal too short for segmentation?
    if len(sig_splits) == 0:
        sig_splits.append(sig)

    return sig_splits

def getMultiSpec(path, seconds=SPEC_LENGTH, overlap=SPEC_OVERLAP):

    #open wav file
    (rate, sig) = wave.read(path)
    print "SAMPLE RATE:", rate,

    #adjust to different sample rates
    if rate != 44100:
        sig, rate = changeSampleRate(sig, rate)

    #split signal into chunks
    sig_splits = splitSignal(sig, rate, seconds, overlap)

    #calculate spectrogram for every split
    for sig in sig_splits:
        
        magspec = getSpecFromSignal(sig, rate, seconds)

        yield magspec

######################################################
if __name__ == "__main__":
        
    events = [src_dir + event + '/' for event in sorted(os.listdir(src_dir))]
    print "NUMBER OF EVENTS:", len(events)

    #parse wave files for every event class
    for event in events:
        total_specs = 0
        
        #get wav files for event class
        wav_files = [event + wav for wav in sorted(os.listdir(event))]

        #parse wav files
        for wav in wav_files:

            #stats
            spec_cnt = 0
            print wav,
            
            try:
                #extract specs from wav file
                for spec in getMultiSpec(wav):

                    #output dir for specs
                    dst_dir = spec_dir + event.split("/")[-2] + "/"
                    if not os.path.exists(dst_dir):
                        os.makedirs(dst_dir)

                    #save spec
                    cv2.imwrite(dst_dir + wav.split("/")[-1].rsplit(".")[0] + "_" + str(spec_cnt) + ".png", spec * 255.0)
                    spec_cnt += 1
                    total_specs += 1
                
                print "SPECS:", spec_cnt

            except:
                print spec_cnt, "ERROR"
                traceback.print_exc()
                pass
            
