import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow
import json
import random
import pickle

with open ("C:/Users/Exa/Desktop/Python/chatbot/intents.json","r") as file:
  data_1 = json.load(file)
with open ("C:/Users/Exa/Desktop/Python/chatbot/intents_new.json","r") as file:
  data_2 = json.load(file)

dunno = ["I don´t understand", "sorry, try asking differently", "I am not following"]


# porovná data v json_file 1 a 2 a vyhodnotí jestli byla ve 2 změněna. Pokud změněna nebyla tak přeskočí training
# a jde rovnou na chat. V případě, že byla změněna tak přepíše původní intents a provede training
if data_1 == data_2:
    print("data in both files are the same, loading old model")
    with open ("data_1.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
else:
    with open ("C:/Users/Exa/Desktop/Python/chatbot/intents.json","w") as file:
        json.dump(data_2, file, indent = 4)
    num_epoch = input("How many epochs to train this bot? ")
    num_epoch = int(num_epoch)
    words = []
    labels = []
    docs_x = []
    docs_y = []


    for i in data_1["intents"]:
        for pattern in i["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(i["tag"])

            if i["tag"] not in labels:
                labels.append(i["tag"])


    #preprocessing
    words = [stemmer.stem(w.lower()) for w in words if w != "?"] #pro každé slovo které najde ve větě udělá lower case w.lower() je již přímo ve stemmeru
    words = sorted(list(set(words))) #setřídí a vyřadí duplikáty (set) dá je do listu a následně je setřídí a->Z

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    # projde slova v docs_x a následně je zařadí do listu bag jako 1 nebo 0. v případě, že se v doc_x slovo vyskytuje přiřadí na patřičnou pozici 1
    for x,doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open ("data_1.pickle", "wb") as f:                 # toto uloží zmíněné listy do pickle formátu, který na začátku zkusí načíst
        pickle.dump((words, labels, training, output), f)   # je to nutné kvůli načtení modelu, které v případě vynechání trainingu musí načíst

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

if data_1 == data_2 :
    model.load("model.tflearn")
else:
    model.fit(training, output, n_epoch= num_epoch, batch_size = 8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words (s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for sent in s_words:
        for i, w in enumerate(words):
            if w == sent:
                bag[i]= 1


    return np.array(bag)

def chat():
    print("Start talking with the bot! (type quit to stop)")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        
        results = model.predict([bag_of_words(inp, words)])[0] #vytvoří výsledky pravděpodobností do kterého tagu patří slovo
        results_index = np.argmax(results) #vrátí index nejvyšší hodnoty results
        tag = labels[results_index]
        
        if results[results_index] > 0.7 : #odpoví v daním tagu pouze pokud je pravděpodobnost větší než zde napsaná
            
            for tg in data_1["intents"]:
                if tg["tag"] == tag:
                    responses = tg["responses"]

            print(random.choice(responses))
        else:
            print(random.choice(dunno))
            question = input("Do you want to add another intent? Y/N: ")
            print(question)
            if question.lower() in ["yes", "y", "yap", "yeah"]:
                print("What should it be called?")
                new_tag = []
                new_patterns = []
                new_responses = []
                new_tag = input("Enter the new tag: ")
                new_patterns = input("Enter the new patterns, separated by commas : ").split(',')
                new_responses = input("Enter the new responses, separated by commas : ").split(',')
                               
                print("Next time you train me, I will know that")
               
                new_data = {
                    'tag': new_tag,
                    'patterns': new_patterns,
                    'responses': new_responses,
                    'context_set': ''
                    }
                
                data_1["intents"].append(new_data)


                # Uloží do nového json filu, který se následně porovnává při dalším spuštění          
                with open('C:/Users/Exa/Desktop/Python/chatbot/intents_new.json', 'w') as file:
                    json.dump(data_1, file, indent = 4)    
            else:        
                continue

                      
chat()

