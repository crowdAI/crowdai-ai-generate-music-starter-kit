import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, TimeDistributed
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint
import tqdm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# np.set_printoptions(threshold=np.nan)

# import pretty_midi

# m = pretty_midi.PrettyMIDI("sample_submission.mid")
# pr = m.get_piano_roll()
#
# np.save(open("output.npy","wb"), pr)

data = np.load(open("output.npy", "rb")).T
data = data/128

train_percent = 0.6
data_train = data[:int(train_percent*data.shape[0])]
data_test = data[int(train_percent*data.shape[0]):]

MIN_MIDI_NOTE = 24
MAX_MIDI_NOTE = MIN_MIDI_NOTE + 88
timesteps = data.shape[0]
# print data[:,24:24+88].shape

X_WINDOW_SIZE = 1000
Y_WINDOW_SIZE = 1000

nb_samples = data_train.shape[0] - X_WINDOW_SIZE - Y_WINDOW_SIZE

x_train = np.zeros((nb_samples, X_WINDOW_SIZE, 128))
y_train = np.zeros((nb_samples, Y_WINDOW_SIZE, 128))

print "Loading training data..."
for i in tqdm.tqdm(xrange(nb_samples)):
    x_train[i,:,:] = data[i:i+X_WINDOW_SIZE,]
    y_train[i,:,:] = data[i+X_WINDOW_SIZE:i+X_WINDOW_SIZE+Y_WINDOW_SIZE,]

# trials = len(x_train_list)
# features = x_train_list[0].shape[1]
#
# print input_mat.shape
#
#
hidden = [64, 64, 64]
dropouts = [0.2, 0.2, 0.2]
model = Sequential()
model.add(LSTM(
    hidden[0],
    input_shape=(X_WINDOW_SIZE, 128),
    return_sequences=True,
    stateful=False,
    go_backwards=True,
    activation='tanh'
))
model.add(Dropout(dropouts[0]))

model.add(LSTM(
    hidden[1],
    return_sequences=True,
    stateful=False,
    go_backwards=True,
    activation='tanh'
))
model.add(Dropout(dropouts[1]))

model.add(LSTM(
    hidden[-1],
    return_sequences=True,
    stateful=False,
    go_backwards=True,
    activation='tanh'
))
model.add(Dropout(dropouts[-1]))
model.add(TimeDistributed(Dense(128, activation='sigmoid')))

model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())

config = model.get_config()
import pickle
pickle.dump(config, open("checkpoints/model.config.pickle", "wb"))

checkpoint = ModelCheckpoint("checkpoints/epoch-{epoch:02d}.hdf5")
casllbacks_list = [checkpoint]

print "Starting Training....."
history = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=200,
    callbacks=casllbacks_list,
    verbose=1)

print "Writing History...."
pickle.dump(history, open("checkpoints/history.pickle", "wb"))
