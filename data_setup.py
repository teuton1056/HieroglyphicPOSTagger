from unicodedata import category
import torch 
import pickle 

def load_word_list(split='all'):
	if split == 'test':
		with open('test_word_list.pkl', 'rb') as f:
			return pickle.load(f)
	elif split == 'train':
		with open('train_word_list.pkl', 'rb') as f:
			return pickle.load(f)
	else:
		with open('word_list.pkl', 'rb') as f:
			return pickle.load(f)

def convert_word_list_to_category_dict(word_list, data='form'):
	# converts a list of words to a dictionary of categories
	category_dict = {}
	for word in word_list:
		if word['pos'] not in category_dict:
			category_dict[word['pos']] = []
		category_dict[word['pos']].append(word[data])
	return category_dict

def generate_alphabet(word_list):
	# generates an alphabet from a list of words
	alphabet = []
	for word in word_list:
		for c in word['form']:
			if c not in alphabet:
				alphabet.append(c)
	return alphabet

pos_dict = convert_word_list_to_category_dict(load_word_list(split='train'))
# pos_dict = convert_word_list_to_category_dict(load_word_list(split='all'))
test_pos_dict = convert_word_list_to_category_dict(load_word_list(split='test'))
pos_list = list(pos_dict.keys())
alphabet = generate_alphabet(load_word_list())

#print(pos_dict['VERB'][5:10])

def sign2index(sign):
	# converts a sign to an index
	return alphabet.index(sign)

def wordToTensor(word):
	# converts a word to a tensor
	tensor = torch.zeros(len(word), 1, len(alphabet))
	for i, c in enumerate(word):
		tensor[i][0][sign2index(c)] = 1
	return tensor 

#print(wordToTensor(pos_dict['VERB'][6]).size())