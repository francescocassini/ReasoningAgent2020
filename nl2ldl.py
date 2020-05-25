#   pip install SpeechRecognition

import speech_recognition as sr
import stanza
stanza.download('en')
nlp = stanza.Pipeline('en')



r = sr.Recognizer()
mic = sr.Microphone()
with mic as source:
    
    r.adjust_for_ambient_noise(source)
    print("What is the goal?")
    audio = r.listen(source)
    
    sentence=r.recognize_google(audio)
print(sentence)

# sentence = "Get the ball, use the key and open the box"

doc = nlp(sentence)
tokens = doc.sentences[0].words

text = []
pos = []
for token in tokens:
    text.append(token.text)
    pos.append(token.xpos)
    

with open("stoplist.txt","r") as stop:
    l = stop.read().splitlines() 
    
indices = []
for i,word in enumerate(text):
    if word in l:
        pos[i] = ''
        text[i] = ''

sentence = ""
for word in text:
    sentence = sentence + " " + word
text = list(filter(lambda a: a != '', text))
pos = list(filter(lambda a: a != '', pos))
        

flag = True;
verb= False;

predicates = [];
# connectives = {'and': '&', 'or': '', 'not':'!'};
for word, tag in zip(text,pos):
    if (word == 'not' or word == 'without'):
        flag = False;
    if (tag == 'VB'):
        verb = True;
    if (verb == True):
        if (tag == 'NN'):
            predicates.append([word, flag]);
            flag = True;
            verb = False;


formula = ""
subformula = ""
if (len(predicates)!=0):
    subformula += '('
    for pred, value in predicates:
        subformula += ' & '
        if (value == True):
            subformula += '!'+pred
        else:
            subformula += pred
    subformula += ')*'
    subformula = subformula[:1]+subformula[4:];
    formula = "<"

    for pred, value in predicates:
        formula += subformula + ';'
        if (value == False):
            formula += '!'
        formula += pred + ';'

    formula = formula[:len(formula)-1]+'>'
    formula+="tt"
    print(formula)



