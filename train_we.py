# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 15:32:27 2020

@author: Sajeev
"""
# import argparse
# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('', metavar='N', type=int, nargs='+',
#                     help='an integer for the accumulator')
# parser.add_argument('--sum', dest='accumulate', action='store_const',
#                     const=sum, default=max,
#                     help='sum the integers (default: find the max)')
# args = parser.parse_args()
# print(args.accumulate(args.integers))


import numpy as np
import nltk
import sys
import pickle
import glob
from datetime import datetime
from nltk.corpus import words

# Confgurations
concated_file ="data/concat_file.txt"
VOCAB_SIZE = 10000
sents_fobj_initial_pattern = './data/sentences_list'
model_fobj_initial_pattern = './data/word2vec_model'

def remove_new_lines_in_file(infile, outfile):
    with open(infile) as f1, open(outfile, 'ab+') as f2:
        for line in f1:
            f2.write(line.strip().encode('utf-8'))
    return outfile

def load_file(file):
    with open(file, 'rb+') as f:
        contents = f.read()
    contents = contents.decode("utf-8")
    return contents

def get_last_saved_file(pattern):
	filename = ''
	try:
		for filename in glob.glob(pattern):
			pass
	except:
		print('Error...!!!')
		pass
	return filename


#from os import path
#if not path.exists('data/concated_file.txt'):
#	concated_file = remove_new_lines_in_file(sys.argv[1], concated_file)
#	print(f'Newlines have been removed and the new file has been saved at {concated_file}')

filename = get_last_saved_file(sents_fobj_initial_pattern + '_*.pkl')
if filename:
	fp = open(filename, 'rb')
	sents = pickle.load(fp)
	fp.close()
	print('Sentences loaded.')
else:
	contents = load_file(sys.argv[1])
	print(f'The file \'{sys.argv[1]}\' has been loaded ')

	import nltk.data  
	sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')  
	sents = sent_detector.tokenize(contents.strip())
	print('Sentence tokenization done. Here are top 5 results:')
	print(sents[0:5])

	num_sents = len(sents)
	for i,sent in enumerate(sents):
		temp = []
		if i % 10000 == 0:
			print(f'{i}/{num_sents}')
		for word in sent.split():
			if not word.isalnum():
				[temp.append(w) for w in nltk.word_tokenize(word) \
									if w.isalnum()]
			else:
				temp.append(word)
		sents[i] = temp
	print('Sentence split done.') 
	
	# Save sents in pickle file to save time in loading later
	dt = datetime.now()
	fp = open(f'{sents_fobj_initial_pattern}_{dt.year}_{dt.month}_{dt.day}_{dt.hour}_{dt.minute}_{dt.second}.pkl', 'ab')
	pickle.dump(sents, fp)
	fp.close()
	print('Sentences saved...')
	
print('Here are top 5 results:')
print(sents[0:5])

batch_size = 64 #128
embedding_size = 300  # Dimension of the embedding vector.
skip_window = 2       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.


import gensim
import gensim.models


# Check if model is already saved
last_model_file = get_last_saved_file(model_fobj_initial_pattern + '_*.pkl')
if last_model_file:
	model_fp = open(last_model_file, 'rb')
	model = pickle.load(model_fp)
	model_fp.close()
	print('Saved model loaded.')
else:
	model = gensim.models.Word2Vec(sentences=sents, min_count=10, size=200)
	print('Model trained')
	training_loss = model.get_latest_training_loss()
	print(training_loss)
	# Dump model to data/ folder with new timestamp so as not to overwrite any old file.
	dt = datetime.now()
	model_fp = open(f'{model_fobj_initial_pattern}_{dt.year}_{dt.month}_{dt.day}_{dt.hour}_{dt.minute}_{dt.second}.pkl', 'ab') 
	pickle.dump(model, model_fp)
	model_fp.close()
	print('Model saved')

vocabulary_length = len(model.wv.index2word)
print(f'vocab_len = {vocabulary_length}')
	
import pdb
pdb.set_trace()



