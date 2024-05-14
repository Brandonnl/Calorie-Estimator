import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator  
import numpy as np 
import pandas as pd
import nltk

df = pd.read_csv('data/food.csv')
print(df.head())