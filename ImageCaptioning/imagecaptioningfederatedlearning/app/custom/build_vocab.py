import nltk
import pickle
import argparse
from collections import Counter
from pycocotools.coco import COCO


class Vocabulary(object):
    """Vocabulary word and index converter."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    def save_to_file(self, filepath):
        """Save the vocabulary to a file."""
        with open(filepath, 'wb') as file:
            pickle.dump((self.word2idx, self.idx2word, self.idx), file)

    def load_from_file(self, filepath):
        """Load the vocabulary from a file."""
        with open(filepath, 'rb') as file:
            self.word2idx, self.idx2word, self.idx = pickle.load(file)

    def __call__(self, word):
        return self.word2idx.get(word, self.word2idx['<unk>'])

    def __len__(self):
        return len(self.word2idx)

def init_vocab(json, threshold):
    """Initialize Vocabulary.
    json = file path to annotations/captions_train2014.json"""
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys() #retrieve the keys from the annotations dictionary
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption']) #access the captions
        tokens = nltk.tokenize.word_tokenize(caption.lower()) #tokenize using nltk
        counter.update(tokens) #count the frequency of each word

        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def build_vocab_main(caption_path, threshold):
    vocab = init_vocab(json=caption_path, threshold=threshold)
    return vocab

def load_vocab_from_another_file(file_path):
    new_vocab = Vocabulary()
    new_vocab.load_from_file(file_path)
    return new_vocab

def main(args):
    vocab = init_vocab(json=args.caption_path, threshold=args.threshold)
    vocab_path = args.vocab_path
    # with open(vocab_path, 'wb') as f:
    #     pickle.dump(vocab, f)
    vocab.save_to_file(vocab_path)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        default='../../../data/annotations/captions_train2014.json', 
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='./vocab_property.pkl', 
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4, 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)