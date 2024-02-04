import logging
import pprint
import gensim
import nltk
from argparse import ArgumentParser

logging.basicConfig(level=logging.INFO)

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--file',
            dest='file', help='input txt file', required=True)
    parser.add_argument('--algo',
            dest='algo', help='Word2Vec Algorithm', required=False)
    return parser

class TokenizedSentences:
	def __init__(self,file:str):
		self.file = file
	def __iter__(self):
		with open(self.file) as f:
			corpus = f.read()
		raw_sentences = nltk.tokenize.sent_tokenize(corpus)
		for sentence in raw_sentences:
			if len(sentence) > 0:
				yield gensim.utils.simple_preprocess(sentence,min_len=2,max_len=15)



def main():
    parser = build_parser()
    args = parser.parse_args()
    sentences = TokenizedSentences(args.file)
    model = gensim.models.word2vec.Word2Vec(sentences = sentences,
										sg=args.algo, #CBOW MODEL: 0; Skip Gram: 1
										window = 3,
										min_count = 3,
										epochs = 10,
										vector_size = 100
										)
    model.save(args.file+'.model')

if __name__ == '__main__':
	main()