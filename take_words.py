import nltk
from nltk.corpus.reader import TaggedCorpusReader

# Assuming the pos_tag.txt file is located in the current directory
corpus_root = '.'  # Change this if the file is in a different directory
file_pattern = r'data/my_dataset.txt'

# Create a TaggedCorpusReader instance
corpus_reader = TaggedCorpusReader(corpus_root, file_pattern)

# Get words from the tagged sentences
words = corpus_reader.words()
print(len(words))
unduplicate = set(words)
unduplicate.remove(',')
unduplicate.remove('.')
print(len(unduplicate))


# Write words to release.txt, one word per line
with open('release.txt', 'w',encoding='utf-8') as f:
    f.write('\n'.join(unduplicate))

print("Words have been written to release.txt successfully.")
