from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn

def get_sentiment_scores(words):
    sentiment_results = {}

    for word in words:
        synsets = wn.synsets(word)  # Get synsets for the word
        if not synsets:
            sentiment_results[word] = {"pos": 0, "neg": 0, "obj": 1}  # Default neutral
            continue

        # Get the most common synset
        synset = synsets[0]  # Taking the first synset (most common meaning)
        swn_synset = swn.senti_synset(synset.name())

        sentiment_results[word] = {
            "pos": swn_synset.pos_score(),
            "neg": swn_synset.neg_score(),
            "obj": swn_synset.obj_score(),
        }

    return sentiment_results

# Example usage:
words =  ['excellent', 'material', 'passionate', 'student', 'better', 'government', 'future']
sentiment_scores = get_sentiment_scores(words)

# Print results
for word, scores in sentiment_scores.items():
    print(f"{word}: {scores}")