from nltk.tokenize import word_tokenize

def segment_word(word, valid_words):
    segmented_word = []
    word_length = len(word)
    start_index = 0

    while start_index < word_length:
        longest_match = None
        for end_index in range(start_index + 1, word_length + 1):
            current_part = word[start_index:end_index]
            if current_part in valid_words:
                longest_match = current_part

        if longest_match:
            segmented_word.append(longest_match)
            start_index += len(longest_match)
        else:
            # If no valid match found, append the single character
            segmented_word.append(word[start_index])
            start_index += 1

    return segmented_word

def rawang_word_segmentation(input_sentence):
    
    # Step 1: Create a set of valid words from raw.txt
    with open('./data/words_corpus.txt', 'r', encoding='utf-8') as file:
        valid_words = set(file.read().split())

    # Step 2: Tokenize the input sentence using NLTK
    
    tokenized_words = word_tokenize(input_sentence)

    result = []

    for word in tokenized_words:
        if word in valid_words:
            result.append(word)
        else:
            segmented_word = segment_word(word, valid_words)
            result.extend(segmented_word)
    return result

