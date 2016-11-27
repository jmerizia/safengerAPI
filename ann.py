import numpy as np

def nonlin(x, deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

alph = "abcdefghijklmnopqrstuvwxyz"
word = "applesauce"
indices = []

for n in xrange(9):
    if word[n] in alph:
        indices.append(float(alph.index(letter) + 1))

thing = [(x / 100) for x in indices]

X = np.array([[.1],
              [.0],
              [.6],
              [.1]])

y = np.array([[thing[0]],
              [thing[1]],
              [thing[2]],
              [thing[3]]
              [thing[4]],
              [thing[5]],
              [thing[6]],
              [thing[7]],
              [thing[8]],
              [thing[9]]])

# X = np.array([[0, 0, 1],
#               [0, 1, 1],
#               [1, 0, 1],
#               [1, 1, 1]])

# y = np.array([[0.5],
#               [0.9],
#               [1],
#               [1]])

np.random.seed(1)

# randomly initialize our weights with mean 0
syn0 = 2*np.random.random((1, 4)) - 1
syn1 = 2*np.random.random((4, 1)) - 1

for j in xrange(100000): 

    # Feed forward through 
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))

    # calculate loss
    l2_error = y - l2

    if (j % 10000) == 0:
        print "Error: " + str(np.mean(np.abs(l2_error)))
        print l2

    # is what direction is target value?
    # and how sure are we?
    l2_delta = l2_error * nonlin(l2, deriv=True)

    # how much did each l1 value contribute to the
    # error in l2
    l1_error = l2_delta.dot(syn1.T)

    # in what direction is the target l1?
    l1_delta = l1_error * nonlin(l1, deriv=True)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

final_word = ""
for elem in l2:
    #print int(round(elem[0]*100)) - 1
    final_word += alph[int(round(elem[0]*100)) - 1]
print final_word
