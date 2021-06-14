---
layout: post
title:  "Genre Classifier"
date:   2021-06-02 18:08:11 +0100
categories:
---

## Part 1: Data Set and Project Aims

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
    
To get started, I needed to research the properties of the sound waves in music that in which I could extract and create a training set from. The 4 groups of features are Zero Crossing Rate, Spectral Centroid, Spectral Rolloff and MFCC. These will be explained in detail in Part 2.




## Part 2: Feature Extraction 

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

# Classifier


```python
header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

file = open('data.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
```


```python
def func():
    for g in genres:
        for filename in os.listdir(f'./genre_data/{g}'):
            songname = f'./genre_data/{g}/{filename}'
            y, sr = librosa.load(songname, mono=True, duration=30)
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            rms = librosa.feature.rms(y=y)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            to_append += f' {g}'
            file = open('data.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())
```


```python
func()
data = pd.read_csv('data.csv')

# Dropping unneccesary columns
data = data.drop(['filename'],axis=1)
genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```


```python
#Creating Sequential Model
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train,
                    y_train,
                    epochs=20,
                    batch_size=128)

test_loss, test_acc = model.evaluate(X_test,y_test)
print('This classifier is currently running with an accuracy of: ',test_acc)

model.save('my_model.h5')
```

 
    This classifier is currently running with an accuracy of:  0.6449999809265137


# Test on User Data


```python
#importing model and reading data used for model
data = pd.read_csv('data.csv')
data = data.drop(['filename'],axis=1)
model = load_model('my_model.h5')

#copy of original dataset
X_new = np.array(data.iloc[:, :-1], dtype = float)
```


```python
##Writing fuctions to standardise the dataset. 'means' and 'stdevs' will be used to standardise user data.

# calculate column means
def column_means(dataset):
    means = [0 for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        means[i] = sum(col_values) / float(len(dataset))
        means = np.array(means)
    return means

means = column_means(X_new)

# calculate column standard deviations
def column_stdevs(dataset, means):
    stdevs = [0 for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        variance = [pow(row[i]-means[i], 2) for row in dataset]
        stdevs[i] = sum(variance)
    stdevs = [sqrt(x/(float(len(dataset)-1))) for x in stdevs]
    return stdevs

stdevs = column_stdevs(X_new, means)
```


```python
##accepting a URL as input, downloading the video to a desired directory and converting to a WAV file
user_url = str(input("Please enter a URL for the song of your choice: "))

class MyLogger(object):
    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        print(msg)


def my_hook(d):
    if d['status'] == 'finished':
        print('Done downloading')

ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': '/home/ryan/genres/genre_classifier/test_data/%(title)s.%(ext)s',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
        'preferredquality': '192',
    }],
    'logger': MyLogger(),
    'progress_hooks': [my_hook],
}
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download([user_url])
    
#removing spaces in the file 
directory = '/home/ryan/genres/genre_classifier/test_data'
[os.rename(os.path.join(directory, f), os.path.join(directory, f).replace(' ', '_').lower()) for f in os.listdir(directory)]
```

    Please enter a URL for the song of your choice: https://soundcloud.com/nasirjones/halftime
    Done downloading




```python
#writing a header for the user data
header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

#creating the features of the user file in the same way as for the original dataset
file = open('user_data.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
for filename in os.listdir('test_data'):
    songname = f'./test_data/{filename}'
    print(songname)
    y, sr = librosa.load(songname, mono=True, duration=30)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
    for e in mfcc:
        to_append += f' {np.mean(e)}'
    file = open('user_data.csv', 'a', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(to_append.split())
```

    ./test_data/halftime.wav



```python
#reading and standardising the user data (one observation)
user_data = pd.read_csv('user_data.csv')
user_data = user_data.drop(['filename'],axis=1)
X_user = user_data.drop(labels='label', axis = 1)
X_user = np.array(X_user)

X_user_scaled = (X_user - means)/stdevs

#Predicting on the user data using Convolution Neural Network
user_pred = model.predict(X_user_scaled)
output = np.argmax(user_pred)

this_dict = {
    0 : 'blues',
    1 : 'classical',
    2 : 'country',
    3 : 'disco',
    4 : 'hiphop',
    5 : 'jazz',
    6 : 'metal',
    7 : 'pop',
    8 : 'reggae',
    9 : 'rock' 
}

print("This song should fit into the " + str(this_dict[output]) + " genre.")

#Removing the user file from the directory 
dir = '/home/ryan/genres/genre_classifier/test_data'
for f in os.listdir(dir):
    os.remove(os.path.join(dir, f))
```

    This song should fit into the hiphop genre.
