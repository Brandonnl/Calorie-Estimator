# %%
import matplotlib.pyplot as plt
from PIL import Image 
import numpy as np 
import pandas as pd
from string import ascii_letters
import os
import nltk
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

# %% [markdown]
# # Reducing Dimensions using PCA

# # %%
# #
# # DON'T RUN
# # FINDING THE NUMBER OF COMPONENTS
# #

# pca = PCA()
# pca.fit_transform(vectorized_documents.toarray())

# # get explained variance ratio
# explained_variance_ratio = pca.explained_variance_ratio_

# # plot explained variance ratio
# plt.figure(figsize=(8, 6))
# plt.plot(np.cumsum(explained_variance_ratio), marker='o')
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative Explained Variance Ratio')
# plt.title('Explained Variance Ratio vs. Number of Components')
# plt.grid(True)
# plt.show()

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

# %%
from transformers import CLIPProcessor, CLIPModel
import torch

# %%
model_name = "openai/clip-vit-base-patch32"
clip_processor = CLIPProcessor.from_pretrained(model_name)
clip_model = CLIPModel.from_pretrained(model_name)

# %%
IMAGE_SIZE = (224,224)
clip_model.eval()

def encode_image(image):
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)

    return image_features

def prepare_image (image):
    img = Image.open (image)
    img = img.resize (IMAGE_SIZE)
    #img = np.expand_dims(img,axis = 0 )
    #img = np.array(img)/255.0
    return img

image_dir = 'data/images/images/'
def process_images(row):
    image_path = image_dir + row['Image_Name']
    preprocessed_image = prepare_image(image_path)
    image_features = encode_image(preprocessed_image)
    return image_features

def process_single_image(image_path):
    preprocessed_image = prepare_image(image_path)
    image_features = encode_image(preprocessed_image)
    return image_features

# %%
# create CLIP encodings for every image in the df
df['image_encoding'] = df.apply(process_images, axis=1)

# %%
# print(df.head())
df.to_csv('data/out.csv', index=False)

# %%
def compute_target_pca_encoding(ingredients, pca_df):
    target_pca_encoding = np.zeros(NUM_COMPONENTS)
    for ingredient in ingredients:
        if ingredient in pca_df.index:
            max_component_index = pca_df.loc[ingredient].idxmax()
            target_pca_encoding[max_component_index] += pca_df.loc[ingredient, max_component_index]
    
    return torch.tensor(target_pca_encoding, dtype=torch.float32)

# add a new column with pca encodeding for each image
df['target_PCA_encoding'] = [compute_target_pca_encoding(ingredients.split(','), ingredient_component_mapping.T) for ingredients in df['parsed_ingredients']]

# %% [markdown]
# # Creating the Dataset

# %%
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

# %%
class InputDataset(Dataset):
    def __init__(self, clip_encodings, pca_components):
        self.clip_encodings = clip_encodings
        self.pca_components = pca_components

    def __len__(self):
        return len(self.clip_encodings)

    def __getitem__(self, idx):
        clip_encoding = self.clip_encodings[idx]
        pca_component = self.pca_components[idx]

        return clip_encoding, pca_component

clip_encodings = df.image_encoding.tolist()
target_pca_components = df.target_PCA_encoding.tolist()
dataset = InputDataset(clip_encodings, target_pca_components)

train_size = int(0.9 * DATASET_LIMIT)
val_size = int(0.1 * DATASET_LIMIT)

train_data, val_data = random_split(dataset, [train_size, val_size])

# %%
batch_size = 64

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# %% [markdown]
# # Creating the Model

# %%
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

# %%
clip_input_size = 512
hidden_size = 128
model = CLIPToPCA(clip_input_size, NUM_COMPONENTS, hidden_size)

learning_rate = 0.0001
num_epochs = 20
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for clip_encodings, pca_components in train_loader:
        optimizer.zero_grad()
        outputs = model(clip_encodings)
        loss = loss_function(outputs, pca_components)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * clip_encodings.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    
    # validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for clip_encodings, pca_components in val_loader:
            outputs = model(clip_encodings)
            loss = loss_function(outputs, pca_components)
            val_loss += loss.item() * clip_encodings.size(0)
    val_loss = val_loss / len(val_loader.dataset)

    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss}, Validation Loss: {val_loss}")
torch.save(model.state_dict(), 'data/food_model_weights.pth')

print('Finished Training')

# %%
new_image_path = image_dir + '3-ingredient-blueberry-champagne-granita.jpg'
new_image_encodings = process_single_image(new_image_path).unsqueeze(0)

# %%
model.eval()
with torch.no_grad():
    predicted_encodings = model(new_image_encodings).numpy()

encodings_list = predicted_encodings.flatten().tolist()
encodings_sorted = sorted(range(len(encodings_list)), key=lambda k: encodings_list[k], reverse=True)

top = 20
for i in range(top):
    print(pca_ingredients[encodings_sorted[i]], encodings_list[encodings_sorted[i]])

# %% [markdown]
# # K-MEANS (probably don't need this)

# # %%
# #
# # calculate WCSS for different values of k
# # DON'T RUN
# #
# wcss = []
# low_range = 1
# high_range = 70
# for i in range(low_range, high_range):
#     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=50, n_init=10, random_state=0)
#     kmeans.fit(reduced_ingredients)
#     wcss.append(kmeans.inertia_)

# # plot the elbow
# plt.figure(figsize=(8, 6))
# plt.plot(range(low_range, high_range), wcss, marker='o')
# plt.title('Elbow Method')
# plt.xlabel('Number of Clusters')
# plt.ylabel('WCSS')
# plt.xticks(np.arange(low_range, high_range, 1))
# plt.grid(True)
# plt.show()

# # %%
# kmeans = KMeans(n_clusters=61, max_iter=300) 
# kmeans.fit(reduced_ingredients) 


