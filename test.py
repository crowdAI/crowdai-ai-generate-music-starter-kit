import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, TimeDistributed
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint


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

nb_samples = 2000

x_train_list = [np.expand_dims(
                np.atleast_2d(
                    data[i:i+X_WINDOW_SIZE,]
                ), axis=0) for i in xrange(nb_samples)]

y_train_list = [np.expand_dims(
                np.atleast_2d(
                    data[i+X_WINDOW_SIZE:i+X_WINDOW_SIZE+Y_WINDOW_SIZE,]
                ), axis=0) for i in xrange(nb_samples)]


x_train = np.concatenate(x_train_list, axis=0)
y_train = np.concatenate(y_train_list, axis=0)
print x_train.shape
print y_train.shape

# trials = len(x_train_list)
# features = x_train_list[0].shape[1]
#
# print input_mat.shape
#
#
hidden = [64, 64, 64]
dropouts = [0.5, 0.5, 0.5]
model = Sequential()
model.add(LSTM(
    64,
    input_shape=(X_WINDOW_SIZE, 128),
    return_sequences=True
))
model.add(Dropout(dropouts[0]))

model.add(LSTM(
    64,
    return_sequences=True
))
model.add(Dropout(dropouts[1]))

model.add(LSTM(
    64,
    return_sequences=True
))
model.add(Dropout(dropouts[-1]))
model.add(TimeDistributed(Dense(128)))

model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())

checkpoint = ModelCheckpoint("checkpoints/epoch-{epoch:02d}.hdf5")
casllbacks_list = [checkpoint]

result = model.fit(
    x_train, y_train,
    epochs=150,
    batch_size=10,
    callbacks=casllbacks_list,
    verbose=1)
