import nltk

# Define the location of your custom dataset file
custom_dataset_path = 'data/my_dataset.txt'

# Define a subclass of TaggedCorpusReader to handle your custom dataset format
class CustomTaggedCorpusReader(nltk.corpus.reader.TaggedCorpusReader):
    def __init__(self, root, fileids, sep='/'):
        nltk.corpus.reader.TaggedCorpusReader.__init__(self, root, fileids, sep)

# Create an instance of the custom tagged corpus reader
custom_reader = CustomTaggedCorpusReader('.', custom_dataset_path)

# Access the tagged sentences from the custom dataset
tagged_sentences = custom_reader.tagged_sents()
print(tagged_sentences)