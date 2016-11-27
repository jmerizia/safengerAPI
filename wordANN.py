import numpy as np
import json
from random import shuffle

alph = "_qQwWaAzZxXsScCdDeErRfFvVtTbDgGhHyYnNjJmMuUkKiIlLoOpP"

insults = []
with open('insults2.txt') as txtfile:
    for row in txtfile:
        message = str(row)
        #message = message.replace("<div>", "").replace("</div>", "").replace("<br>", "").replace("<b>", "").replace("<i>", "")
        parsedMessage = ""
        for char in message:
            if char == " ":
                parsedMessage += "_"
            elif char in alph:
                parsedMessage += char
        insults.append(parsedMessage)

compare = []
for i in range(len(insults)):
    compare.append([1])

compliments = []
with open('compliments.txt') as txtfile:
    for row in txtfile:
        message = str(row)
        parsedMessage = ""
        for char in message:
            if char == " ":
                parsedMessage += "_"
            elif char in alph:
                parsedMessage += char
        compliments.append(parsedMessage)

for i in compliments:
    compare.append([0])

messages = insults + compliments



f = lambda x: 1.0/(1.0 + np.exp(-5*x))
ff = lambda x: x*(1 - x)

def parseMessages(messages):
    indices = []
    for i in range(len(messages)):
        indices.append([])
        for n in xrange(10): # only take the first 144 characters
            if n < len(messages[i]):
                if messages[i][n] in alph:
                    indices[i].append(float(alph.index(messages[i][n])) / 100)
            else:
                indices[i].append(float(alph.index("_")) / 100)
    return indices

#messages = ["You_are_a_prick", "You_are_a_nice_guy", "piece_of_shit"] # get this online/text file
indices = parseMessages(messages)

y = np.array(compare) # expected output

X = np.array(indices) # input

#np.random.seed(1)

# randomly initialize our weights with mean 0
W1 = 2*np.random.randn(10, 8)
W2 = 2*np.random.randn(8, 5)
W3 = 2*np.random.randn(5, 3)
W4 = 2*np.random.randn(3, 1)

def forward(X, W1, W2, W3, W4):
    h1 = f(np.dot(X, W1))
    h2 = f(np.dot(h1, W2))
    h3 = f(np.dot(h2, W3))
    out = f(np.dot(h3, W4))
    return (out, h1, h2, h3)

for j in xrange(6000): 

    out, h1, h2, h3= forward(X, W1, W2, W3, W4)

    Eout = y - out

    if j % 1000 == 0:
        print "Error: ", np.linalg.norm(Eout)
        print "Value: ", np.linalg.norm(out)

    Dout = Eout * ff(out)
    Eh3 = Dout.dot(W4.T)

    Dh3 = Eh3 * ff(h3)
    Eh2 = Dh3.dot(W3.T)

    Dh2 = Eh2 * ff(h2)
    Eh1 = Dh2.dot(W2.T)

    Dh1 = Eh1 * ff(h1)

    W4 += h3.T.dot(Dout)
    W3 += h2.T.dot(Dh3)
    W2 += h1.T.dot(Dh2)
    W1 += X.T.dot(Dh1)

with open("W1.txt", "w") as text_file:
    text_file.write(json.dumps(W1.tolist()))
with open("W2.txt", "w") as text_file:
    text_file.write(json.dumps(W2.tolist()))
with open("W3.txt", "w") as text_file:
    text_file.write(json.dumps(W3.tolist()))
with open("W4.txt", "w") as text_file:
    text_file.write(json.dumps(W4.tolist()))


#test = ["You_are_a_okay_guy_Ithink", "oooooooooooooooooooLiek_ummmmm_You_are_a_really_huge_idiot_prick_you_stupid_fucker_whydontyougofuckyourselffffffff", "piece_of_shit"]
test = ["holy_shit", ]#"great", "bloody", "cunt", "fat", "fuck_you", "hello", "nigger"]
t = parseMessages(test)

print forward(t, W1, W2, W3, W4)[0].tolist()
