---
layout: post
title:  "Genre Classifier Part 1: Feature Extraction"
date:   2021-06-02 18:08:11 +0100
categories:
---

## Section A: Data Set and Project Aims

This project is based on the Gtzan dataset. The dataset consists of 1000 audio snipets, each 30 seconds long, across 10 different music genres. The aim of this project was to use a neural network algorithm to train on the dataset and then be able to perform on unseen data given by a user.


Below are two songs from two different genres represented by their respective wave plots and correspond spectorgrams. The wave plot of an audio clip is simply the amplitude or loudness with respect to time. A spectrogram is a visual representation of the spectrum of frequencies of the signal with respect to time.


<audio  controls="controls" >
    Your browser does not support the audio element.
</audio>


```python
#display waveform
x, sr = librosa.load(audio_path)
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)
```

    
![png]({{"/assets/images/output_3_1.png"}})
    



```python
#display Spectrogram
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz') 
#If to pring log of frequencies  
#librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()
```

    
![png]({{"/assets/images/output_4_1.png"}})
    
To get started, I needed to research the properties of the sound waves in music that in which I could extract and create a training set from. The 4 groups of features are Zero Crossing Rate, Spectral Centroid, Spectral Rolloff and MFCC. These will be explained in detail in Section B.




## Section B: Feature Extraction 

# Zero Crossing Rate

The zero crossing rate is the rate at which the signal changes from postive to a negative value in a given interval and vice versa, i.e. how often the signal passes through zero on the y-axis. This can be seen in the two figures below which depict some 100 array columns of the wave plot. As you can see for the rock song, the signal passes through zero 8 times. zero crossing rate tends to be higher for highly percussive genres. Compare this to the zero crossing rate of the classical audio clip.

```python
x, sr = librosa.load(audio_path)
#Plot the signal:
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)
# Zooming in
n0 = 9000
n1 = 9100
plt.figure(figsize=(14, 5))
plt.plot(x[n0:n1])
plt.grid()
```
  
![png]({{"/assets/images/output_6_0.png"}})
    
 
![png]({{"/assets/images/output_6_1.png"}})
    

```python
zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)
print(sum(zero_crossings))
```

    8


# Spectral Centroid

The spectral centroid is the weighted mean of the frequencies present in an audio clip. We can think of this as a sort of 'centre of mass' for sound. In musical terms, this can be translated to be a predictor of the 'brightness' [2] of a sound and therefore is used in processing as a measure of musical timbre. Timbre is the characteristic in music that gives different instruments a different sound, even if both instruments are playing the same notes at the same fundamental pitch.

[2] "https://asa.scitation.org/doi/10.1121/1.3818432".

```python
#spectral centroid -- centre of mass -- weighted mean of the frequencies present in the sound
import sklearn
spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
spectral_centroids.shape# Computing the time variable for visualization
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)
# Normalising the spectral centroid for visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)
#Plotting the Spectral Centroid along the waveform
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='r')
```

    
![png]({{"/assets/images/output_9_1.png"}})
    


# Spectral Rolloff

From the librosa documentation files the The roll-off frequency is defined 'each frame as the center frequency for a spectrogram bin such that at least roll_percent (0.85 by default) of the energy of the spectrum in this frame is contained in this bin and the bins below.'


What this means is that the sepectral rolloff is the frequency below which the 85% of the total spectral energy lies. The default spectral rolloff treshhold is 85% by default.

```python
spectral_rolloff = librosa.feature.spectral_rolloff(x, sr=sr)[0]
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff), color='r')
```

    
![png]({{"/assets/images/output_11_1.png"}})
    


# MFCC — Mel-Frequency Cepstral Coefficients

MFCCs are used to describe the overall shape of a spectral evelope. In this project, I will be extracting a set of 20 MFCCs. MFCCs are coefficients that will collectively make up an MFC (mel-frequency cepstrum). The general process of deriving MFCCs is mulitstep. The overview is [3]:
1. Take the Forier transform of a signal.
2. Map the powers of the spectrum obtained from the fourier transorm onto the mel scale.
3. Take the logs of the powers of the mel frequencies.
4. Calculate the discrete foruier transform of the list of mel frequencies.
5. The ampltiudes of the resulting spectrum are the MFCCS.

```python
mfccs = librosa.feature.mfcc(x, sr=sr)
print(mfccs.shape)#Displaying  the MFCCs:
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
```

    (20, 1293)

    
![png]({{"/assets/images/output_13_2.png"}})
    

[3] https://doi.org/10.1016/j.specom.2011.11.004

