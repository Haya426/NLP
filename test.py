import pickle
from word_segmentation import rawang_word_segmentation
from my_crf import sent2features
with open('model/crf_model.pickle', 'rb') as f:
    crf_loaded = pickle.load(f)

# Preprocess the new sentence
new_sentence = 'Ínìgø , àngí Àngcè vlatpè mv̀shøqò dv́ngte àng nv̀ng màshúshì'
print('Input sentence:',new_sentence)
new_sentence = new_sentence.lower()  # Convert to lowercase if needed
new_sentence_tokens = rawang_word_segmentation(new_sentence)

# Tag the new sentence with the loaded model
new_sentence_features = sent2features([(token, '') for token in new_sentence_tokens]) # Assuming the POS tags are not provided
new_sentence_pos_tags = crf_loaded.predict([new_sentence_features])

print('segmentated words : ',new_sentence_tokens)
print('pos tags : ',new_sentence_pos_tags)

