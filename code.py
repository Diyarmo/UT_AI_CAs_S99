#!/usr/bin/env python
# coding: utf-8

# ## Genetic Algorithm to Decrypt Substitution Ciphers 

# The substitution cipher replaces every instance of a particular letter in the plain text with a different letter from the cipher text.[1]
# 
# 
# Thus the key to decoded the cipher text is a one-to-one mapping of letters of text's Alphabet. A sample key is shown below.
# 
# 
#     {'l': 'h', 't': 'z', 'w': 'y', 'i': 'i', 'y': 'a', 'u': 'd', 'q': 'l', 'j': 'j', 'p': 'e', 'h': 't', 'o': 'v', 'a': 'o', 'k': 'g', 'c': 'b', 'g': 's', 'm': 'k', 'n': 'n', 'x': 'q', 'd': 'f', 'r': 'p', 'b': 'r', 'z': 'x', 'e': 'w', 'f': 'm', 'v': 'u', 's': 'c'}
# 
# 
# Chromosome is defined as a dictionary mapping each letter of alphabet to another one(like the one above).
# 

#     init_population()
# randomly initiates $population\_size$ chromosomes.
# 
# $population\_size$ is set to 100, and works fine, lowering it cuases decrease in diversity thus increases risk of failure and not finding the key. Setting $population\_size$ to more than 100, only increases computation time.
# 
# Based on the results using $population\_size$ equal to 50, we can find the key so it's a good idea to reduce the $population\_size$ when the $best\_rate$ increases.

# $global\_text$ is used to create a refrence dictionary, but at first it has to be cleaned. I simply erased all non-alphabet characters from the $global\_text$. However you can try to remove all stop-words to gain a better result.
# 

# The idea of fittness function is very simple.
# 
# Let $text\_unique\_len$ be sum of length of unique words of $encoded\_text$ and $correct\_unique\_len$ be sum of length of unique words of $decoded\_text$ which appear in $global\_text$
# 
# Becuase of one-to-one mapping $text\_unique\_len$ remains the same after each decoding, so $Correct\_Decryption\_Rate$ is defined as $\frac{correct\_unique\_len}{text\_unique\_len}.$
# 
# There are also other options for fitness function such as weighting the length of each word by weights related to frequency of words in $global\_text$ so that decoding of more frequent words becomes more valuable. Also using 
# 
#     difflib.get_close_matches()
# we can find semi-decoded words and have a more accurate fitness function, also we should be aware of computation time of this function. 
# Another way of implementing fitness function is to use n-grams and compare frequency of them in $global\_text$ and $decoded\_text$

# After evaluating each generation rates are sorted and based on ranks each chromosome is given a probability, these probabilities are used to randomly choose chromosomes for crossover and choose who survives.
# 
#     mating_pool = np.random.choice(population, new_gen_size, False, probs)
#     elites = population[arg_sorted[-elite_size:]]
#     population = np.random.choice(population, old_gen_size, False, probs)
# 
# to make sure that the best chromosomes of each generation make their way to the new generation, $elites$ are chosen to be added again to population after mutation phase.
# 
#     population = list(map(self.mutate, self.population)) + elites
#     
#     
# OX1 method is used to perform Crossover on population, but crossover is not enough to find the key, because new generations are combinations of their parents and thus nothing new happens. Using mutation there is chance for changes to increase the $best\_rate$. If mutation leads to a better rate mutated chromosome will be in elites and thus makes its way to new generations and replicates rapidly, and if mutation caueses a worse rate in chromosome it's likely that it will be replaced by better chromosomes. 

# There is a chance that algorithm gets stuck in a local maximum. 
# 
# To solve this problem let $iterations\_unchanged$ be number of iterations without a increase in $best\_rate$ and if it reaches a specific value(250) the algorithem starts from the start. 

# Tuning the parameters were done and the best combination of parameters are 
# 
#    
#     self.mutatation_prob = 0.03
#     self.population_size = 100
#     self.new_gen_size = 20
#     self.elite_size = 10
#     self.probs_decayin_rate = 9/10



import numpy as np
from copy import deepcopy
import re
from time import time
class Decoder:
    def __init__(self, text):
        self.corpus = open("global_text.txt").read().lower()
        self.corpus = re.sub('[^a-z]+', ' ', self.corpus)
        self.corpus_words = set(self.corpus.split())
        self.raw_text = text
        self.text = " ".join(set(re.sub('[^a-z]+', ' ', text.lower()).split()))
        self.Alphabet = list('abcdefghijklmnopqrstuvwxyz')
        self.text_unique_len = len(self.text.replace(" ", ""))
        self.mutatation_prob = 0.03
        self.best_chromosome = None
        self.population_size = 100
        self.new_gen_size = 20
        self.elite_size = 10
        self.maximum_generations = 500
        self.probs_decayin_rate = 9/10
        self.best_rate = 0
        
    def init_population(self):
        self.population = []
        values = self.Alphabet
        for i in range(self.population_size):
            np.random.shuffle(values)
            self.population.append(self.create_chromosome(values))
            
    def decode_ch(self, chromosome):
        return "".join(list(map(lambda x: chromosome[x] if x in chromosome else x, list(self.text))))
    
    def decode(self, verbose=False):
        self.run(verbose)
        for i in self.Alphabet:
            self.best_chromosome[i.upper()] = self.best_chromosome[i.lower()].upper()
        return "".join(list(map(lambda x: self.best_chromosome[x] if x in self.best_chromosome else x, list(self.raw_text))))

    def rate(self, chromosome):
        decoded = self.decode_ch(chromosome).split()
        r = np.sum(list(map(lambda x:len(x) if x in self.corpus_words else 0, decoded)))
        return r/self.text_unique_len

    def crossover(self, parentA_v, parentB_v, points):
        child_v = ['']*len(self.Alphabet)
        child_v[points[0]:points[1]] = parentA_v[points[0]:points[1]]
        j = points[1]
        for i in range(1, len(self.Alphabet)+1):
            if parentB_v[(i+points[1])%len(self.Alphabet)] not in child_v:
                child_v[j] = parentB_v[(i+points[1])%len(self.Alphabet)]
                j = (j+1)%len(self.Alphabet)
        return child_v
    
    def create_chromosome(self, values):
        return {a:b for a,b in zip(self.Alphabet, values)}

    def mutate(self, chromosome):
        for i in self.Alphabet:
            if np.random.uniform() < self.mutatation_prob:
                j = np.random.choice(list(self.Alphabet))
                chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
        return chromosome

    def mating(self, parentA, parentB):
        points = np.random.choice(range(len(self.Alphabet)), 2)
        childA_v = self.crossover(list(parentA.values()), list(parentB.values()), points)
        childB_v = self.crossover(list(parentB.values()), list(parentA.values()), points)
        return self.create_chromosome(childA_v), self.create_chromosome(childB_v)
    
    def run(self, verbose=False):
        self.old_gen_size = self.population_size - self.new_gen_size - self.elite_size
        while self.best_rate != 1:
            self.best_rate = 0
            iterations_unchanged = 0
            self.init_population()
            ranks = np.zeros(self.population_size)
            i = 0
            while True:
                rates = np.array(list(map(self.rate, self.population)))
                arg_sorted = np.argsort(rates)
                ranks[arg_sorted] = np.arange(self.population_size, 0, -1)
                probs = (self.probs_decayin_rate)**(ranks)
                probs = probs / np.sum(probs)
                
                if self.best_rate < np.max(rates):
                    self.best_rate = np.max(rates)
                    self.best_chromosome = self.population[np.argmax(rates)]
                    iterations_unchanged = 0
                else:
                    iterations_unchanged += 1
                    
                if iterations_unchanged == 250:
                    if verbose:
                        print(i, ":\t Regenerating population")
                    break
                
                if i%10==0 and verbose:
                    print(i, ":\t Correct Decryption Rate=", self.best_rate)
                
                
                if self.best_rate == 1:
                    if verbose:
                        print(i, ":\t Key Found!")
                    break

                mating_pool = np.random.choice(self.population, self.new_gen_size, False, probs).reshape((-1,2))
                elites = deepcopy(list(np.array(self.population)[arg_sorted[-self.elite_size:]]))
                self.population = list(np.random.choice(self.population, self.old_gen_size, False, probs))
                np.random.shuffle(mating_pool)

                for parents in mating_pool:
                    self.population += self.mating(parents[0], parents[1])
                self.population = list(map(self.mutate, self.population)) + elites

                i += 1





t = time()
decoder = Decoder(open("encoded_text.txt").read())
decoded_text = decoder.decode()
print(time() - t)
print(decoded_text)

