import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator  
import numpy as np 
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

df = pd.read_csv('data/food.csv')

is_noun = lambda pos: pos[:2] == 'NN'

def parse_ingredients(row):
    line = row['Cleaned_Ingredients']
    tokenized = nltk.word_tokenize(line)
    nouns = [word.lower() for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)] 
    return nouns

df['parsed_ingredients'] = df.apply(parse_ingredients, axis=1)
print(df.at[0, 'parsed_ingredients'])
print(df.head())