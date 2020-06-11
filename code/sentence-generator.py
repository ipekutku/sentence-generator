#importing necessary libraries
import time
import collections
import random
import math

start = time.time()

class LanguageModel(object):
    
    #class variables
    list_of_sentences = []
    trigrams = {}
    bigrams = {} 
    unigrams = {}
    unigram_probs = {}
    bigram_probs = {}
    trigram_probs = {}
    V = 0 #Number of words in vocabulary
    
    #takes a folderpath as the argument, returns the list of sentences
    def dataset(self, folderpath):
        
        with open(folderpath) as f:
            for line in f:
                if(line!='\n'):
                    self.list_of_sentences.append(self.process_lines(line))

        return self.list_of_sentences
    
    #takes an integer as the argument, builds the n-gram language model 
    def Ngram(self, n):
        
        ngram_dict = self.callDict(n)
        
        for sentence in self.list_of_sentences:
            words = sentence.split()
            if(n==3):
                words.insert(0, '<s>') #for trigram, one more <s> token is needed
            for i in range(len(words)-n+1):
                string = ' '.join(words[i:i+n])
                try:
                    ngram_dict[string] += 1
                except:
                    ngram_dict[string] = 1

        ngram_dict = self.sortDict(n) #converting the n-gram dictionary to an ordered dictionary
        return ngram_dict
    
    #returns the probability of a given sentence
    def prob(self, sentence, n):
    
        word_list = sentence.split()
        word_list.append('</s>')
        prob_dict = self.callProbDict(n)
        prob_val = 0
    
        for i in range(n-1):
            word_list.insert(0, '<s>')
            
        for i in range(len(word_list)-n+1):
            string = ' '.join(word_list[i:i+n])
            try:
                value = prob_dict[string]
                prob_val += math.log2(value)               
            except:
                prob_val = 0
                break

        return prob_val
    
    #returns the smoothed probability of a given sentence
    def sprob(self, sentence, n):
    
        word_list = sentence.split()
        word_list.append('</s>')
        prob_dict = self.callProbDict(n)
        ngram_dict = self.callDict(n)
        sprob_val = 0
    
        for i in range(n-1):
            word_list.insert(0, '<s>')
        
        for i in range(len(word_list)-n+1):        
            string = ' '.join(word_list[i:i+n])       
            try:
                numerator = ngram_dict[string]
                denominator = round(numerator / prob_dict[string])
                value = (numerator + 1) / (denominator + self.V)        
            except:
                if(n != 1):
                    prev = ' '.join(word_list[i:i+n][:-1])                
                    if(prev != '<s> <s>' and prev in self.callDict(n-1)):
                        value = 1 / (self.callDict(n-1)[prev] + self.V)
                    elif(prev == '<s> <s>'):
                        value = 1 / (self.callDict(1)['<s>'] + self.V)
                    else:
                        value = 1 / self.V
                else:
                    value = 1 / (len(self.callDict(n)) - self.callDict(1)['<s>'] + self.V)        
            sprob_val += math.log2(value)    
        return sprob_val
    
    #returns the perplexity of a given sentence
    def ppl(self, sentence, n):

        N = len(sentence.split())

        if(self.prob(sentence, n) == 0):
            prob_value = 2**(self.sprob(sentence, n))
        else:
            prob_value = 2**(self.prob(sentence, n))

        if(n==1):
            perplexity = prob_value ** (-1/float(N))
        else:
            perplexity = prob_value ** (-1/float(N+1))
    
        return perplexity
    
    #returns the word that is possible to come after the given word argument
    def Next(self, word=None):
        
        List, prob_dict, n = self.returnProbList(word) # list of possible words after the word, their probs, and n
        r = random.random() #random number between 0 and 1
        var = .0
        random.shuffle(list(List))
        
        for element in List:
            if(n==1):
                var += prob_dict[element]
            else:
                var += prob_dict[' '.join([word,element])]
                
            if var >= r:
                return element

        return -1
    
    #generates sentences for all n-gram models
    def generate(self, length, count):
        
        unigram_sentences= self.gen_sentence(1, length, count)
        bigram_sentences = self.gen_sentence(2, length, count, "<s>")
        trigram_sentences = self.gen_sentence(3, length, count, "<s> <s>")
        
        return unigram_sentences, bigram_sentences, trigram_sentences
    
    #function for pre-processing the dataset
    def process_lines(self, line):

        words = line.split()

        if(words[0] == '21'):
            words = words[:-2] # removing unnecessary parts of the line

        words[0] = '<s>' #adding <s> token
        words.append('</s>') #adding </s> token
        line = ' '.join(words).lower() #lowercasing the line
        return line
    
    #function to call a specific model when it's needed
    def callDict(self, n):
        
        if(n==1):
            return self.unigrams
        if(n==2):
            return self.bigrams
        if(n==3):
            return self.trigrams
     
    #function to call a specific probs dictionary when it's needed
    def callProbDict(self, n):
        
        if(n==1):
            return self.unigram_probs
        if(n==2):
            return self.bigram_probs
        if(n==3):
            return self.trigram_probs
    
    #function to calculate the number of the words in the vocabulary
    def calculateV(self):
        return len(self.unigrams)
    
    #function to sort a given dictionary using OrderedDict type
    def sortDict(self,n):
            
        ngram_dict = collections.OrderedDict(sorted(self.callDict(n).items(), key=lambda t: t[1], reverse=True)) 
        return ngram_dict
    
    #function to create probability dictionaries for each ngram model
    def calcProb(self, n):
            
        prob_dict = {}
        ngram_dict = self.callDict(n)
            
        if(n==1):             
            word_count = (sum(ngram_dict.values()) - ngram_dict['<s>'])    
            for key in ngram_dict.keys():
                next_word_prob = ngram_dict[key] / word_count
                prob_dict[key] = next_word_prob             
            del prob_dict['<s>']                
            return prob_dict                
        else:            
            for key in ngram_dict.keys():                
                words = key.split()
                prev_words = words[:-1]
                if(' '.join(prev_words) == "<s> <s>"):
                    prev_count = self.unigrams['<s>']
                else:
                    prev_count = model.callDict(n-1)[' '.join(prev_words)]                    
                key_count = ngram_dict[key]
                next_word_prob = key_count / prev_count
                next_word = words[-1]
                prob_dict[key] = next_word_prob                    
            return prob_dict
        
    #function to find the possible words that can come after the given word   
    def returnProbList(self, word=None):
        
        if word is None:
            n=1
            prob_dict = self.callProbDict(n)
            List = prob_dict.keys()
        else:
            n = len(word.split()) + 1
            prob_dict = self.callProbDict(n)
            List = []
            for key in prob_dict.keys():
                words = key.split()
                if(words[:-1] == word.split()):
                    List.append(words[-1])    
                    
        return List,prob_dict,n
    
    #function to generate sentences of the given model
    def gen_sentence(self, n, length, count, first=None):
        
        sentence_list = []
        
        for i in range(count):            
            first_word = self.Next(first)            
            if(first != None):
                sentence = first + ' ' + first_word
            else:
                while True:
                    if(first_word != '</s>'):
                        sentence = first_word
                        break
                    else:
                        first_word = self.Next(first)                                            
            for j in range(length-1):                
                if(n == 1):
                    prev_words = None
                else:
                    prev_words = ' '.join(sentence.split()[-(n-1):])                
                next_word = self.Next(prev_words)            
                if(next_word == '</s>'):
                    break                
                sentence = ' '.join([sentence, next_word])            
            sentence = ' '.join(list(filter(('<s>').__ne__, sentence.split())))            
            sentence_list.append(sentence)
            
        return sentence_list
    
   #prints the given sentence with its prob, sprob, and ppl values 
    def print_model_sentences(self, sentences, n):

        for i in range(len(sentences)):
            sentence = sentences[i]
            print("Sentence %d: %s \n Probability = %f, Smoothed Probability = %f, Perplexity= %f\n\n" 
                  % (i+1, sentence, self.prob(sentence,n), self.sprob(sentence,n), self.ppl(sentence,n)))
        return
    
    #prints  all generated sentences with their prob, sprob, and ppl values
    def print_all_sentences(self, unigram_sentences, bigram_sentences, trigram_sentences):
        
        print('****************** UNIGRAM Sentences ******************\n')
        self.print_model_sentences(unigram_sentences, 1)
        print('\n****************** BIGRAM Sentences ******************\n')
        self.print_model_sentences(bigram_sentences, 2)
        print('\n****************** TRIGRAM Sentences ******************\n')
        self.print_model_sentences(trigram_sentences, 3)
        
        return

#creating a LanguageModel object
model = LanguageModel()

#reading the dataset and creating the list_of_sentences
model.dataset("assignment1-dataset.txt")

#building n-gram models
model.unigrams = model.Ngram(1)
model.bigrams = model.Ngram(2)
model.trigrams = model.Ngram(3)

model.V = model.calculateV() #calculating the unique word count

#creeating probability dictionaries
model.unigram_probs = model.calcProb(1)
model.bigram_probs = model.calcProb(2)
model.trigram_probs = model.calcProb(3)

#generating and printing sentences
unigram_s, bigram_s, trigram_s = model.generate(50, 3)
model.print_all_sentences(unigram_s, bigram_s, trigram_s)

end = time.time()
print("\n\nTime the program takes is %f minutes" %((end-start) / 60))