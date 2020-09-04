import string
import re
import pickle
from unicodedata import normalize
from numpy.random import shuffle
from numpy import array
from numpy import argmax
from keras.preprocessing import text
from keras.preprocessing import sequence
from keras import utils
from keras.utils import vis_utils
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
import keras
from nltk.translate import bleu_score #import corpus_bleu


# load doc into memory
def loadFromDocument(filename):
    # open the file as read only
    file = open(filename, mode='rt', encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# split a loaded document into sentences
def toWords(doc):
    lines = doc.strip().split('\n')
    pairs = [line.split('\t') for line in lines]
    return pairs


# clean a list of lines
def cleanWords(lines):
    cleaned = list()
    # prepare regex for char filtering
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for pair in lines:
        clean_pair = list()
        for line in pair:
            # normalize unicode characters
            line = normalize('NFD', line).encode('ascii', 'ignore')
            line = line.decode('UTF-8')
            # tokenize on white space
            line = line.split()
            # convert to lowercase
            line = [word.lower() for word in line]
            # remove punctuation from each token
            line = [word.translate(table) for word in line]
            # remove non-printable chars form each token
            line = [re_print.sub('', w) for w in line]
            # remove tokens with numbers in them
            line = [word for word in line if word.isalpha()]
            # store as string
            clean_pair.append(' '.join(line))
        cleaned.append(clean_pair)
    return array(cleaned)


# save a list of clean sentences to file
def saveCleanData(sentences, filename):
    pickle.dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)


# load a clean dataset
def loadCleanData(filename):
	return pickle.load(open(filename, 'rb'))


# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = text.Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# max sentence length
def max_length(lines):
	return max(len(line.split()) for line in lines)


# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
	# integer encode sequences
	X = tokenizer.texts_to_sequences(lines)
	# pad sequences with 0 values
	X = sequence.pad_sequences(X, maxlen=length, padding='post')
	return X


# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None


# generate target given source sequence
def predict_sequence(model, tokenizer, source):
	prediction = model.predict(source, verbose=0)[0]
	integers = [argmax(vector) for vector in prediction]
	target = list()
	for i in integers:
		word = id2Word(i, tokenizer)
		if word is None:
			break
		target.append(word)
	return ' '.join(target)


# one hot encode target sequence
def oneHotEncode(sequences, vocab_size):
	ylist = list()
	for sequence in sequences:
		encoded = utils.to_categorical(sequence, num_classes=vocab_size)
		ylist.append(encoded)
	y = array(ylist)
	y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
	return y


# define NMT model - encoder/decoder pattern
def define_model(srcVocabSize, targetVocabSize, srcSize, targetSize, modelSize):
	model = Sequential()
	model.add(Embedding(srcVocabSize, modelSize, input_length=srcSize, mask_zero=True))
	model.add(LSTM(modelSize))
	model.add(RepeatVector(targetSize))
	model.add(LSTM(modelSize, return_sequences=True))
	model.add(TimeDistributed(Dense(targetVocabSize, activation='softmax')))
	return model


# map an integer to a word
def id2Word(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None


# generate target given source sequence
def predict_sequence(model, tokenizer, source):
	prediction = model.predict(source, verbose=0)[0]
	integers = [argmax(vector) for vector in prediction]
	target = list()
	for i in integers:
		word = id2Word(i, tokenizer)
		if word is None:
			break
		target.append(word)
	return ' '.join(target)


# evaluate the skill of the model
def evaluate_model(model, tokenizer, sources, raw_dataset):
	actual, predicted = list(), list()
	for i, source in enumerate(sources):
		# translate encoded source text
		source = source.reshape((1, source.shape[0]))
		translation = predict_sequence(model, engTokenizer, source)
		raw_target, raw_src, _ = raw_dataset[i]
		if i < 10:
			print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))
		actual.append([raw_target.split()])
		predicted.append(translation.split())
	# calculate BLEU score
	print('BLEU-1: %f' % bleu_score.corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % bleu_score.corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % bleu_score.corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % bleu_score.corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))




# load dataset
modelFileName = 'data/deu.txt'
doc = loadFromDocument(modelFileName)
# split into english-german pairs
pairs = toWords(doc)
# clean sentences
clean_pairs = cleanWords(pairs)
# save clean pairs to file
#save_clean_data(clean_pairs, 'english-german.pkl')
# spot check
for i in range(100):
    print('[%s] => [%s]' % (clean_pairs[i, 0], clean_pairs[i, 1]))

# load dataset
#raw_dataset = load_clean_sentences('english-german.pkl')

# reduce dataset size
n_sentences = 10000
dataset = clean_pairs[:n_sentences] #raw_dataset[:n_sentences, :]
# random shuffle
shuffle(dataset)
# split into train/test
train, test = dataset[:9000], dataset[9000:]
# save
saveCleanData(dataset, 'english-german-both.pkl')
saveCleanData(train, 'english-german-train.pkl')
saveCleanData(test, 'english-german-test.pkl')

# prepare english tokenizer
engTokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size = len(engTokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])
print('English Vocabulary Size: %d' % eng_vocab_size)
print('English Max Length: %d' % (eng_length))
# prepare german tokenizer
germanTokenizer = create_tokenizer(dataset[:, 1])
ger_vocab_size = len(germanTokenizer.word_index) + 1
ger_length = max_length(dataset[:, 1])
print('German Vocabulary Size: %d' % ger_vocab_size)
print('German Max Length: %d' % ger_length)

# prepare training data
trainX = encode_sequences(germanTokenizer, ger_length, train[:, 1])
trainY = encode_sequences(engTokenizer, eng_length, train[:, 0])
trainY = oneHotEncode(trainY, eng_vocab_size)
# prepare validation data
testX = encode_sequences(germanTokenizer, ger_length, test[:, 1])
testY = encode_sequences(engTokenizer, eng_length, test[:, 0])
testY = oneHotEncode(testY, eng_vocab_size)

# define model
model = define_model(ger_vocab_size, eng_vocab_size, ger_length, eng_length, 256)
model.compile(optimizer='adam', loss='categorical_crossentropy')
# summarize defined model
print(model.summary())
vis_utils.plot_model(model, to_file='model.png', show_shapes=True)
# fit model
modelFileName = 'model.h5'
checkpoint = ModelCheckpoint(modelFileName, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.fit(trainX, trainY, epochs=30, batch_size=64, validation_data=(testX, testY), callbacks=[checkpoint], verbose=2)

# load datasets
dataset = loadCleanData('english-german-both.pkl')
train = loadCleanData('english-german-train.pkl')
test = loadCleanData('english-german-test.pkl')
# prepare english tokenizer
engTokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size = len(engTokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])
# prepare german tokenizer
germanTokenizer = create_tokenizer(dataset[:, 1])
ger_vocab_size = len(germanTokenizer.word_index) + 1
ger_length = max_length(dataset[:, 1])
# prepare data
trainX = encode_sequences(germanTokenizer, ger_length, train[:, 1])
testX = encode_sequences(germanTokenizer, ger_length, test[:, 1])

# load model
model = keras.models.load_model('model.h5')
# test on some training sequences
print('train')
evaluate_model(model, engTokenizer, trainX, train)
# test on some test sequences
print('test')
evaluate_model(model, engTokenizer, testX, test)

