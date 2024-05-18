import matplotlib.pyplot as plt
from PIL import Image 
import numpy as np 
import pandas as pd
from string import ascii_letters
import torch.nn as nn
import os
import nltk
import torch
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('brown')

# %% [markdown]
# # PREPROCESSING THE INGREDIENTS

# %%
# need to remove these from ingredients
measurements = (['tablespoon', 'tbsp', 'teaspoon', 'tsp', 'cup', 'pint', 'pt', 
                 'quart', 'qt', 'gallon', 'gal', 'ounce', 'ounc', 'ounces', 'oz', 'fluid', 'fl', 'pound', 
                 'lb', 'liter', 'litre', 'l', 'ml', 'gram', 'g', 'inch', 'diameter', 'meter', 'medium',
                 'grill', 'cm', 'handful', 'size', 'firm', 'cupg', 'cupsg', 'cupsml', 'x', 'little', 'divided',
                 'total', 'more', 'package', 'bag', 'bottle', 'tbspg', 'xxinch', 'box', 'instructions', 'info',
                 'ozg', 'lbg', 'kg'])

DATASET_LIMIT = 5000

# %%
is_noun = lambda pos: pos[:2] == 'NN'
stemmer = nltk.stem.PorterStemmer()

all_ingredients = []

def parse_ingredients(row):
    list = eval(row['Cleaned_Ingredients'])
    ingredients = []
    for ingredient in list:
        # removing non-letters
        letter_only = ''.join(l for l in ingredient if l in set(ascii_letters + ' '))
        # tokenizing into words
        tokenized = nltk.word_tokenize(letter_only)
        # remove all except nouns, and remove measurements
        nouns = [word.lower() for (word, pos) in nltk.pos_tag(tokenized) if(pos[:2] == 'NN' or pos[:2] == 'NNS')]
        nouns = [noun for noun in nouns if (noun not in measurements) and (stemmer.stem(noun) not in measurements)]
        if len(tokenized) > 0 and len(nouns) == 0:
            nouns.append(tokenized[-1])
        # add as new row, also add to a full ingredient list as features
        joined = ' '.join(nouns)
        ingredients.append(joined)
        if joined not in all_ingredients:
            all_ingredients.append(joined)
    return ','.join(ingredients)

# %%
df = pd.read_csv('data/food.csv')
df = df[df.Image_Name != '#NAME?']
df = df.iloc[:DATASET_LIMIT]

# add column for ingredients that are cleaned and parsed
df['parsed_ingredients'] = df.apply(parse_ingredients, axis=1)

# %%
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans 

# %%
ingredient_strings = df['parsed_ingredients']
vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(','))
vectorized_documents = vectorizer.fit_transform(ingredient_strings)


# %%
# this is for the dataset of 5000
# if using full dataset, can go between 1500-1750
NUM_COMPONENTS = 900
pca = PCA(n_components=NUM_COMPONENTS)
reduced_ingredients = pca.fit_transform(vectorized_documents.toarray())

# %%
all_ingredients = vectorizer.get_feature_names_out()
ingredient_component_mapping = pd.DataFrame(pca.components_, columns=all_ingredients)
# print(ingredient_component_mapping)

# %%
# get the loadings of each principal component
pca_ingredients = []
for i, pc in enumerate(pca.components_):
    index = np.argsort(np.abs(pc))[::-1]
    
    found = False
    i = 0
    while found == False:
        ing = all_ingredients[index[i]]
        if ing in pca_ingredients:
            i += 1
        else:
            pca_ingredients.append(ing)
            found = True

class CLIPToPCA(nn.Module):
    def __init__(self, clip_input_size, pca_output_size, hidden_size):
        super(CLIPToPCA, self).__init__()
        self.clip_input_size = clip_input_size
        self.pca_output_size = pca_output_size
        self.hidden_size = hidden_size

        self.conv1 = nn.Conv1d(in_channels=clip_input_size, out_channels=hidden_size, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, stride=1, padding=1)
        self.output_fc = nn.Linear(hidden_size, pca_output_size)
        self.activation = nn.ReLU()
        
    def forward(self, clip_encodings):
        clip_encodings = clip_encodings.permute(0, 2, 1)
        
        conv_out = self.conv1(clip_encodings)
        conv_out = self.activation(conv_out)
        conv_out = self.conv2(conv_out)
        conv_out = self.activation(conv_out)
        pooled_out = torch.mean(conv_out, dim=2)
        pca_encodings = self.output_fc(pooled_out)
        return pca_encodings