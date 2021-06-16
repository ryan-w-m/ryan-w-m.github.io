---
layout: post
title:  "Genre Classifier Part 2: Building and Testing the Model"
date:   2021-06-15 
categories:
---


# Classifier

To begin classification, a dataset in a suitable format is needed. Firstly, a CSV file is created with a header as per the features outlined in Part 1. A loop is setup that iterates through each sound file of the original mp3 sound files of the Gtzan dataset and each feature is calculated and then appended to the CSV.

To calculate these features, the librosa package is used. Librosa is a python package for music and audio analysis.

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


## Preprocessing the data
The above function must be called. Then the CSV file is read as a pandas data frame.

```python
func()
data = pd.read_csv('data.csv')
```
Now the data needs to be set up so that the classifier can learn on it. This means creating a set of predictor variables called `X` and a target variable `Y`. In this case the target variable is genres and the predictor variables are the features that have been extracted earlier. 

```python
# Dropping unneccesary columns
data = data.drop(['filename'],axis=1)
genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)
```

In most cases, it is advised to stanardise the data before initiating training because a lot of machine learning algorithms are sensitive to relative scales of features. One method of stanardising is scaling. `The StandardScaler()` function in python scales the features such that the distribtution is centred around 0 with a standard deviation of 1. The formula for this operations is: 

`X = (x - μ)/σ`

Another step in the preprocessing stage is the train/test split. Some data must be kept of of the model training so that it is 'reserved' for testing. This ensures the model can perform on unsseen data. For this project a 80/20 train/test split was used.


```python
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

## Creating the Model

After experimenting with several machine learning alogrithms, the one that perfomed the best was a sequential neural network. 

The chosen topoligy for the neural network is as follows: THe amount of neurons in the input layer should match the amount of features in the dataset. There are two hidden layers in the neural network, using the activation function Relu to account for the vanishing gradient problem. Then, the output layer has 10 nodes, since there are 10 genres in the dataset.

```python
#Creating Sequential Model
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
```

The loss function chosen is sparse catergorical crossentropy because the problem is a classification problem of more than two labels.

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train,
                    y_train,
                    epochs=20,
                    batch_size=128)
```

The model is then tested on the test set, performing with a decent accuracy. However, the model clearly needs further tuning.

```python
test_loss, test_acc = model.evaluate(X_test,y_test)
print('This classifier is currently running with an accuracy of: ',test_acc)

model.save('my_model.h5')
```

 
    This classifier is currently running with an accuracy of:  0.6449999809265137


# Test on User Data

The final part of this project was to get the model to perform on a real world use case. To achieve this, a seprate script was written that reads in the data from the CSV as before, and then imports the model created in the script from Section (1). 

```python
#importing model and reading data used for model
data = pd.read_csv('data.csv')
data = data.drop(['filename'],axis=1)
model = load_model('my_model.h5')

#copy of original dataset
X_new = np.array(data.iloc[:, :-1], dtype = float)
```

The data had to be scaled differently for this script because it is only dealing with one observation, whereas the `Stanardscaler()` fucntion in Python assumes that the data has several rows in the dataset. Thus, the data is scaled manually using the following code snippet[4].

```python

[
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


The program will ask the user for an input in the form of a URL from websites that host video such as Youtube, Soundcloud, etc. For demonstration purposes, the hip-hop song Halftime from Nas' 1994 album Illmatic is chosen.

The program accepts a URL and then then utilises the `youtube_dl` command line script and embeds it as a callable part of the program. The `youtube_dl` script converts a video file into a desired audio format. For this project the target audio format is WAV as this is what the librosa package wants to work with. 

The audio file is sent to the working directory and the file name is formatted in order to be suitable for the CSV file.

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
    'outtmpl': '/path/to/directory/%(title)s.%(ext)s',
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
directory = '/path/to/directory'
[os.rename(os.path.join(directory, f), os.path.join(directory, f).replace(' ', '_').lower()) for f in os.listdir(directory)]
```

    Please enter a URL for the song of your choice: https://soundcloud.com/nasirjones/halftime
    Done downloading


The audio file is broken down in the same way as in Section A. Librosa extracts the relevant features and the data is written to a CSV file.


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
    
The user data is preprocessed in the same way as in Section A, only now the scaling is performed manually using the mean and standard deviation calculated using [4].
    
```python
#reading and standardising the user data (one observation)
user_data = pd.read_csv('user_data.csv')
user_data = user_data.drop(['filename'],axis=1)
X_user = user_data.drop(labels='label', axis = 1)
X_user = np.array(X_user)

X_user_scaled = (X_user - means)/stdevs
```
Then, the model built in Section A is called and performs on the scaled user data. The audio file is correctly classified into it's respective genre. The user data is removed from the  working directory.

```python
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


