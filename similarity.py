f = "pages_filtered/pages_filtered_0.json"
from nltk.tokenize import word_tokenize
import json
from collections import Counter
sent = "For thousands of years, indigenous peoples were the only inhabitants of the territory that is now Maine."
filtered_words = ["of", ".", ",", "the", "is", "?", "!", ";", "are", "am", "were", "was", "(", ")", "\"", "a", "''", ":", "or", "to", "as", "an"]
filt = lambda sent: list(filter((lambda x: x not in filtered_words), word_tokenize(sent.lower())))
filtered_sent = filt(sent)
filtered_counter = Counter(filtered_sent)
print(filtered_counter['is'])
print(filtered_sent)


from sklearn.feature_extraction.text import CountVectorizer
def sim(target_sent, target_counter, input):
    words = set(target_sent).union(set(input))
    input_counter = Counter(input)
    sum = 0
    for word in words:
        sum += target_counter[word]*input_counter[word]
    # just to make sure
    if len(input) is 0:
        return 0
    return sum / (len(target_sent) * len(input))
    
with open(f) as of:
    rj = json.load(of)
count = 0
for title, content in rj.items():
    count += 1
    print(count)  
    for sentence in content["text"]:
        t = sentence["text"]
        filtered_input = filt(t)
        if sim(filtered_sent, filtered_counter, filtered_input) > 0.4:
            print(filtered_input)
  #     break
  # break
