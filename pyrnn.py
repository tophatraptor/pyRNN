import nltk, sys, itertools, random
import numpy as np
#global config options are all up here
vocabsize = 1000
sentencestogen = 100
numberofepochs=5
evallossevery=20
#how often you should check loss - checks epochnumber%evallossevery
#1 = every epoch, any number>numberofepochs = never
learningrate = 0.005

#tokens for the generator to know when it should end a sentence
#we're going to insert these into every processed sentence so that...
#...after a number of iterations, it can start to get a feel for when it should...
#...be ending sentences
unknown_token="UNKNOWNTOKEN"
linestart="LINESTART"
lineend="LINEEND"

#http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression
#softmax is a supervised logistic regression model
def softmax(x):
    sf = np.exp(x)
    sf = sf/np.sum(sf, axis=0)
    return sf

#our recurrent neural network class
#this is where the magic happens
class RNN:
    def __init__(self,vocablen, hidden=300, bpropt=4):
        self.vocablen=vocablen
        #size of vocabulary
        self.hidden = hidden
        # of nodes in our hidden layer
        #the hidden layer is where all the weighted functions with parameters U,V, and W happen
        self.bpropt=bpropt
        #backpropagation through time truncate value
        #since the neural network 'forgets' quickly, this is fine
        #vanishing gradient problem makes this okay
        self.U = np.random.uniform(-np.sqrt(1./vocablen), np.sqrt(1./vocablen), (hidden, vocablen))
        self.V = np.random.uniform(-np.sqrt(1./hidden), np.sqrt(1./hidden), (vocablen, hidden))
        self.W = np.random.uniform(-np.sqrt(1./hidden), np.sqrt(1./hidden), (hidden, hidden))

    #forward propagation to predict word frequencies
    def fprop(self, x):
        T = len(x)
        s = np.zeros((T+1, self.hidden))
        #s is an array of zeroes
        s[-1]=np.zeros(self.hidden)
        o=np.zeros((T, self.vocablen))
        #this is the output matrix we want to work with
        for t in np.arange(T):
            s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
            o[t] = softmax(self.V.dot(s[t]))
            #update it with the softmax matrix run on s
        return [o, s]

    def predict(self,x):
        o, s = self.fprop(x)
        return np.argmax(o,axis=1)

    #backpropagation through time
    def bprop(self, x, y):
        T = len(y)
        o, s = self.fprop(x)
        #partial derivatives for U/V/W
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        del_o = o
        del_o[np.arange(len(y)), y] -= 1.
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(del_o[t], s[t].T)
            del_t = self.V.T.dot(del_o[t]) * (1 - (s[t] ** 2))
            for bprops in np.arange(max(0, self.bpropt), t+1)[::-1]:
                dLdW += np.outer(del_t, s[bprops-1])
                dLdU[:,x[bprops]] += del_t
                del_t = self.W.T.dot(del_t) * (1 - s[bprops-1]  ** 2)
        return [dLdU, dLdV, dLdW]

    #meet the function that is the secondary bottleneck
    #this iterates from 0 to the number of lines we imported every trftime
    #this means that, for our shakespeare input file,
    #it iterates over it 52,482 times every time it needs to make a loss calculation
    def totalloss(self, x, y):
        loss=0
        for i in np.arange(len(y)):
            o, s = self.fprop(x[i])
            goodpredic = o[np.arange(len(y[i])), y[i]]
            loss-= np.sum(np.log(goodpredic))
        return loss

    #summons the totalloss function and divides it out
    #classification method, see: http://cs231n.github.io/neural-networks-2/#losses
    def loss(self,x,y):
        N = np.sum((len(y_i) for y_i in y))
        return self.totalloss(x,y)/N

    def stochgradstep(self, x, y):
        #stochastic gradient descent stepping function
        #this is probably biggest contributor to the speed (or lack thereof)
        #of this program
        #stochgradstep updates the learning rate of the program in accordance
        #with the updated values from backprop
        dLdU, dLdV, dLdW = self.bprop(x, y)
        self.U -= learningrate * dLdU
        self.V -= learningrate * dLdV
        self.W -= learningrate * dLdW

def trainwithsgd(model, xtrain, ytrain):
    #this is the actual training function
    # calculate loss function and the second for loop are computationally expensive
        losses = []
        #a list of tuples (numiterations, loss_at_num_iterations)
        completeiterations = 0
        print("Training with {} epochs".format(numberofepochs))
        for epoch in range(numberofepochs):
            print("Starting epoch {}".format(epoch))
            if epoch % evallossevery == 0:
                print("Calculating loss for epoch {}".format(epoch))
                loss = model.loss(xtrain, ytrain)
                losses.append((completeiterations, loss))
                print("Calculated loss in epoch {} after {} iterations: {}".format(epoch,completeiterations,loss))
                if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                    learningrate = learningrate / 2
                    #If our loss increases after an epoch, reduce the learning rate
                    #This is currently super aggressive
                print("Iterating over batch for epoch {}".format(epoch))
            for i in range(1000):
                #This should be len(ytrain), but that takes absurdly long for some texts
                #A single epoch would take 8 hours on the shakespeare file with len(ytrain)
                #This is actually the bigger bottleneck between the two bottlenecks
                #stochgradstep is fairly slow - varies from 1 to 4 seconds
                model.stochgradstep(xtrain[i], ytrain[i])
                completeiterations += 1
        print("Finished training after {} epochs with {} examples/iterations".format(numberofepochs,completeiterations))

print("Starting")
#parse in text file
print("Reading in text file")
f = open(sys.argv[1],'rt')
data=f.read()
print("Beginning parsing")
#tokenizes it into sentences
#nltk tokenizes on periods for splitting
lines = nltk.sent_tokenize(data)
lines1 = lines
#adds tokens for start and end, so that training for ending sentences is included
lines = ["{} {} {}".format(linestart, x, lineend) for x in lines]

print("Parsed in {} lines".format(len(lines)))
#split on punctuation within each sentence
tokenized  = [nltk.word_tokenize(sent) for sent in lines]

wordfreq = nltk.FreqDist(itertools.chain(*tokenized))
#the asterisk unpacks tokenized into a single array
print("Found {} unique word tokens".format(len(wordfreq.items())))
print("Using a vocabulary of the first {} most common words".format(vocabsize))
#take in the first vocabsize-1 words and add those to our vocabulary
vocab = wordfreq.most_common(vocabsize-1)

#our functions above are going to be generating input/outputs of numbers
#given that, we need a lookup table and a reverse lookup table
indextoword = [x[0] for x in vocab]
indextoword.append(unknown_token)
wordtoindex = dict([(w,i) for i,w in enumerate(indextoword)])

for i, sent in enumerate(tokenized):
    tokenized[i] = [w if w in wordtoindex else unknown_token for w in sent]

print("\nFirst sentence: '{}'".format(lines1[0]))
print("\nFirst sentence after tokenization and substitution: {}".format(tokenized[0]))

#our two input layers
xtrain = np.asarray([[wordtoindex[w] for w in sent[:-1]] for sent in tokenized])
ytrain = np.asarray([[wordtoindex[w] for w in sent[1:]] for sent in tokenized])

np.random.seed(random.randrange(0,1000))
#seed numpy's random number generator for the multinomial array expression later
model = RNN(vocabsize)
#set up the RNN
model.stochgradstep(xtrain[10], ytrain[10])
#step it once

trainwithsgd(model, xtrain, ytrain)

def gentext(model):
    gensentence = [wordtoindex[linestart]]
    #seed the RNN with a starting value of a new sentence
    count=0
    while not gensentence[-1] == wordtoindex[lineend] and count<100:
        #while we haven't generated an end token, continue to generate words
        #don't generate a sentence longer than 100 words
        nextwordprob = model.fprop(gensentence)
        sample = wordtoindex[unknown_token]
        while sample == wordtoindex[unknown_token]:
            #while we have an unknown word as our current word, resample from current vocab
            samples = np.random.multinomial(1, nextwordprob[0][0])
            #this is a multivariate binomial distribution
            sample = np.argmax(samples)
        if indextoword[sample]!=linestart:
            #note that the outer while loop is conditioned on the end token...
            gensentence.append(sample)
            count+=1
    outtext = [indextoword[x] for x in gensentence[1:-1]]
    #...so instead we just don't add it here
    return outtext

out = open("output.txt",'w')
print("Generating {} sentences into output.txt".format(sentencestogen))
#empty the file
out.seek(0)
out.truncate()

#generate text
for i in range(sentencestogen):
    sent = gentext(model)
    out.write(" ".join(sent))
    out.write("\n")
print("Finished generating {} sentences".format(sentencestogen))
