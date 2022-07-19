import json 
import os
from dataclasses import dataclass
import pickle
from html import unescape
import random

@dataclass
class Word_Instance:

	word_id: int
	word: str 
	form: str 
	pos: str

	def __dict__(self):
		return {
			'word_id': self.word_id,
			'word': self.word,
			'form': self.form,
			'pos': self.pos
		}

	def __hash__(self):
		return hash(tuple(self.__dict__().values()))

def load_corpus_file(file_name):
	# loads one of the json corpus files from the corpus folder
	with open(file_name, 'r', encoding='utf-8') as f:
		return json.load(f)

def load_corpus_files(directory):
	# loads all the json corpus files from the corpus folder
	return [load_corpus_file(f"{directory}/{file_name}") for file_name in sorted(os.listdir(directory)) if file_name[0] == "_"]

def corpus2words(corpus):
	# converts a corpus to a list of words
	upos_dict = {
	"verb":"VERB",
	"entity_name":"PROPN",
	"adverb":"ADV",
	"preposition":"ADP",
	"interjection":"INTJ",
	"pronoun":"PRON",
	"substantive":"NOUN",
	"numeral":"NUM",
	"epitheton_title":"SYM",
	"particle":"PART",
	"adjective":"ADJ",
	"unknown":"X",
	"undefined":"X"
	}
	for sent_id in corpus:
		for word in corpus[sent_id]["token"]:
			try:
				yield {"word_id": word["lemmaID"], "word":word['lemma_form'],"form": unescape(word["hiero_unicode"]), "pos": upos_dict[word["pos"]]}
				#yield Word_Instance(word_id=word['lemmaID'],word=word['lemma_form'],form=word['hiero_unicode'],pos=upos_dict[word['pos']])
			except:
				pass

def corpus2words_list(corpus):
	# converts a corpus to a list of words
	return [word for word in corpus2words(corpus)]

def word_list_for_corpora(corpora):
	# converts a list of corpora to a list of words
	return [word for corpus in corpora for word in corpus2words_list(corpus)]

def pickle_word_list(word_list, file_name):
	# pickles a list of words to a file
	with open(file_name, 'wb') as f:
		pickle.dump(word_list, f)

def print_statistics(word_list):
	length = len(word_list)
	print(f"Number of words: {length}")
	#print(f"Number of unique words instances: {len(set(word_list))}")

def pickle_train_test_word_list(word_list, train_ratio=0.6):
	# pickles a list of words to a file
	train_size = int(round(len(word_list) * train_ratio))
	random.shuffle(word_list)
	train_word_list = word_list[:train_size]
	test_word_list = word_list[train_size:]
	with open('train_word_list.pkl', 'wb') as f:
		pickle.dump(train_word_list, f)
	with open('test_word_list.pkl', 'wb') as f:
		pickle.dump(test_word_list, f)

def main():
	# loads all the json corpus files from the corpus folder
	corpora = load_corpus_files('aes')
	# converts a list of corpora to a list of words
	word_list = word_list_for_corpora(corpora)
	# print some stats
	print_statistics(word_list)
	# pickles a list of words to a file
	pickle_train_test_word_list(word_list)
	pickle_word_list(word_list, 'word_list.pkl') 

if __name__ == "__main__":
	main()