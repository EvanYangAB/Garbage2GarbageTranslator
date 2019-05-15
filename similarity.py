from nltk.tokenize import word_tokenize, sent_tokenize
import json
from collections import Counter
sent = "This state's largest city endured an 1855 Rum Riot, put down with the help of 1880 Prohibitionist Candidate for President Neal S. Dow."
filtered_words = ["of", ".", ",", "the", "is", "?", "!", ";", "are", "am", "were", "was", "(", ")", "\"", "a", "''", ":", "or", "to", "as", "an", "()", "for", '``', "with"]
filt = lambda sent: list(filter((lambda x: x not in filtered_words), word_tokenize(sent.lower())))
filtered_sent = filt(sent.replace(u'\xa0', u' '))
filtered_counter = Counter(filtered_sent)
print(filtered_counter['is'])
print(filtered_sent)

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
    # return sum
 
largest_sim = 0
largest_sent = ""
for r in range(0, 57):
    print(r)
    f = "pages_filtered/pages_filtered_" + str(0 + 100000*r) + ".json"
    with open(f) as of:
        rj = json.load(of)
    for title, content in rj.items():
        for sentence in content["text"]:
            ts = sent_tokenize(sentence["text"].replace(u'\xa0', u' ').strip())
            for t in ts:
                filtered_input = filt(t)
                if len(filtered_input) < 5:
                    continue
                s = sim(filtered_sent, filtered_counter, filtered_input)
                if s > largest_sim:
                    print(filtered_input)
                    largest_sim = s
                    largest_sent = filtered_input
      #     break
      # break

print("largest sim: ", largest_sim)
print("sent: ", largest_sent)
