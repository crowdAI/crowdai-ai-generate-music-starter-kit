import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, TimeDistributed
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint
import tqdm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# CUDA_VISIBLE_DEVICES=0 python test.py --checkpoint_dir=checkpoints --hidden_layers=64,64 --window_size=600

import argparse
parser = argparse.ArgumentParser(description='Start training')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', action='store', required=True)
parser.add_argument('--hidden_layers', dest='hidden_layers', action='store', required=True)
parser.add_argument('--window_size', dest='window_size', action='store', required=True)
args = parser.parse_args()

# np.set_printoptions(threshold=np.nan)

# import pretty_midi

# m = pretty_midi.PrettyMIDI("sample_submission.mid")
# pr = m.get_piano_roll()
#
# np.save(open("output.npy","wb"), pr)

CHECKPOINT_DIR = args.checkpoint_dir
hidden_layers = [int(x) for x in args.hidden_layers.split(",")]
dropouts = [0.2 for x in hidden_layers]
window_size = int(args.window_size)

data = np.load(open("output.npy", "rb")).T
data = data/128

train_percent = 1
data_train = data[:int(train_percent*data.shape[0])]
data_test = data[int(train_percent*data.shape[0]):]

MIN_MIDI_NOTE = 24
MAX_MIDI_NOTE = MIN_MIDI_NOTE + 88
timesteps = data.shape[0]
# print data[:,24:24+88].shape

X_WINDOW_SIZE = window_size # 600
Y_WINDOW_SIZE = window_size

nb_samples = data_train.shape[0] - X_WINDOW_SIZE - Y_WINDOW_SIZE

stride = 50
nb_samples_corrected = nb_samples/stride + 1
x_train = np.zeros((nb_samples_corrected, X_WINDOW_SIZE, 128))
y_train = np.zeros((nb_samples_corrected, Y_WINDOW_SIZE, 128))

print x_train.shape
print nb_samples, stride, nb_samples_corrected
print "Loading training data..."
for _i in tqdm.tqdm(xrange(nb_samples_corrected)):
    i = _i * stride
    x_train[_i,:,:] = data[i:i+X_WINDOW_SIZE,]
    y_train[_i,:,:] = data[i+X_WINDOW_SIZE:i+X_WINDOW_SIZE+Y_WINDOW_SIZE,]

hidden_layers = [64, 64]
dropouts = [0.2, 0.2]
model = Sequential()

for _idx in range(len(hidden_layers)):
    if _idx == 0:
        model.add(LSTM(
            hidden_layers[_idx],
            input_shape=(X_WINDOW_SIZE, 128),
            return_sequences=True,
            stateful=False,
            go_backwards=True,
            activation='tanh'
        ))
    else:
        model.add(LSTM(
            hidden_layers[_idx],
            return_sequences=True,
            stateful=False,
            go_backwards=True,
            activation='tanh'
        ))
    model.add(Dropout(dropouts[_idx]))

model.add(TimeDistributed(Dense(128, activation='sigmoid')))

model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())

config = model.get_config()
import pickle
pickle.dump(config, open("{}/model.config.pickle".format(CHECKPOINT_DIR), "wb"))

checkpoint = ModelCheckpoint(CHECKPOINT_DIR + "/epoch-{epoch:02d}.hdf5")
casllbacks_list = [checkpoint]

print "Starting Training....."
history = model.fit(
    x_train, y_train,
    epochs=200,
    batch_size=100,
    callbacks=casllbacks_list,
    verbose=1)

print "Writing History...."
pickle.dump(history, open("{}/history.pickle".format(CHECKPOINT_DIR), "wb"))
