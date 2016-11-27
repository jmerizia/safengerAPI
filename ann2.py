import numpy as np

f = lambda x: 1.0/(1.0 + np.exp(-x))
ff = lambda x: x*(1 - x)

alph = "abcdefghijklmnopqrstuvwxyz"
messages = ["bitchass mother fucker",
            "felch",
            "cunt",
            "skullfuck",
            "Are you even?",
            "What the fuck are you doing?"]
words1 = m1.split(" ")
words2 = m2.split(" ")
indices = []
word = "appl"

for n in xrange(4):
    if word[n] in alph:
        indices.append(float(alph.index(word[n]) + 1))

thing = [(x / 100) for x in indices]
print thing

y = np.array([[1]])



X = np.array([[thing[0], thing[1], thing[2], thing[3]]])

np.random.seed(1)

# randomly initialize our weights with mean 0
W1 = 2*np.random.randn(4, 4)
W2 = 2*np.random.randn(4, 4)
W3 = 2*np.random.randn(4, 1)


for j in xrange(100000): 
    h1 = f(np.dot(X, W1))
    h2 = f(np.dot(h1, W2))
    out = f(np.dot(h2, W3))

    Eout = y - out

    if j % 10000 == 0:
        print "Error: ", np.linalg.norm(Eout)
        print "Value: ", np.linalg.norm(out)

    Dout = Eout * ff(out)
    Eh2 = Dout.dot(W3.T)

    Dh2 = Eh2 * ff(h2)
    Eh1 = Dh2.dot(W2.T)

    Dh1 = Eh1 * ff(h1)

    W3 += h2.T.dot(Dout)
    W2 += h1.T.dot(Dh2)
    W1 += X.T.dot(Dh1)


# final_word = ""
# for elem in l2:
#     #print int(round(elem[0]*100)) - 1
#     final_word += alph[int(round(elem[0]*100)) - 1]
# print final_word
