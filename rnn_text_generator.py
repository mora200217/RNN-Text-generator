

import numpy as np
import tensorflow as tf


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import LSTM
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.callbacks import ModelCheckpoint
import random
import sys


def loadDocument(path):
    f = open(path,'r')
    doc = f.read()
    f.close()
    return doc.lower()

def applyFilter(doc, chars_ignored='"#$%&()*+-/<=>@[\]^_`{|}~\n\t'):
    for ch in chars_ignored:
        if ch == '\n':
          doc = doc.replace(ch,' ')
        else:
          doc = doc.replace(ch,'')
    return doc
  
def sample(preds, temperature=1.0):

	preds = np.asarray(preds).astype('float64')
	preds = np.log(preds) / temperature
	exp_preds = np.exp(preds)
	preds = exp_preds / np.sum(exp_preds)
	probas = np.random.multinomial(1, preds, 1)
	return np.argmax(probas)

def oneHotArrayToChar(one_hot_array, keys, temperature=0.2, argmax=False):
	if argmax:
		return keys[one_hot_array.argmax()]
	index = sample(one_hot_array, temperature)#one_hot_array.argmax()
	return keys[index]
  
def encodedTextToString(one_hot_text, keys):
	string = ""
	for one_hot_array in one_hot_text:
		char = oneHotArrayToChar(one_hot_array, keys, argmax=True)
		string += char
	return string
  


# -----------------------------
# Loading input
# -----------------------------

doc = loadDocument('database.txt')
doc = applyFilter(doc)

tokenizer = Tokenizer(lower=True, char_level=True) # Por caracter :v 
tokenizer.fit_on_texts(doc)

alphabet_size = len(tokenizer.word_index)

keys = list(tokenizer.word_index.keys())
for key in keys:
	tokenizer.word_index[key] -= 1

print('Alphabet: ', tokenizer.word_index)
sequence_of_int = tokenizer.texts_to_sequences([doc])[0]
text_with_one_hot_encoding = to_categorical(sequence_of_int, alphabet_size)

text_len = len(text_with_one_hot_encoding)
print('Text length: ', text_len)

n= 99  #input length
m= 1 #output length

samples = text_len-n

x = np.zeros((samples, n, alphabet_size), dtype=np.bool)
y = np.zeros((samples, m, alphabet_size), dtype=np.bool)

for i in range(samples):
  x[i]=text_with_one_hot_encoding[i:i+n]
  y[i]=text_with_one_hot_encoding[i+n:i+n+m]
  
y = np.squeeze(y) #delete the axis that have shape 1 (case m=1)



# -----------------------------
# Building LSTM model 
# -----------------------------

model = Sequential()

model.add(LSTM(128, input_shape=(n, alphabet_size), dropout = 0.2))
model.add(Dense(alphabet_size))
model.add(Activation('softmax'))

adam = optimizers.Adam(learning_rate=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])



# -----------------------------
# Building text generator
# -----------------------------

#(will be executed at the end of each epoch)
def on_epoch_end(epoch, logs):
  
  print()
  print('----- Generating text after Epoch: %d' % epoch)
  chars_to_generate = 300

  start_index = random.randint(0, len(x)-1)
  for temperature in [0.2, 0.5, 1.0, 1.2]:
    print('----- temperature:', temperature)

    generated = ''
    one_hot_sentence = x[start_index]
    text_sentence = encodedTextToString(one_hot_sentence, keys)
    generated += text_sentence
    print('----- Seed generator: "' + text_sentence + '"')
    sys.stdout.write(generated)

    for i in range(chars_to_generate):
      preds = model.predict(np.array([one_hot_sentence]), verbose=0)[0]
      next_char = oneHotArrayToChar(preds, keys, temperature=temperature)
      generated += next_char

      next_char_one_hot = np.zeros((alphabet_size))
      next_char_one_hot[tokenizer.word_index[next_char]] = 1
      one_hot_sentence = np.append(one_hot_sentence[1:],[next_char_one_hot],axis=0)

      sys.stdout.write(next_char)
      sys.stdout.flush()
    print()



# -----------------------------
# Entrenamiento del Modelo 
# -----------------------------

weights_path = 'weights.hdf5'
checkpointer = ModelCheckpoint(filepath=weights_path, verbose=1)
generator_callback = LambdaCallback(on_epoch_end=on_epoch_end)

batch_size, epochs = 128, 60
# model.fit(x, y, batch_size=batch_size, epochs=epochs, callbacks=[checkpointer, generator_callback], )


model.load_weights('weights.hdf5')
s = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. In rutrum ornare quam, at cursus mi metus.' 
s = s[0:99]
print(len(s))
main_layer = np.array([[[s]]]) 
T = tf.convert_to_tensor(main_layer)
tf.expand_dims(T, -1)
# model.predict(T)
model.summary()
