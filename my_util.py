# -*- coding: utf-8 -*-
class my_util:
    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt
    import seaborn as sns

    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.model_selection import KFold

    from scipy.stats import pearsonr

    import math
    from math import atan, pi

    import tensorflow as tf
    import keras

    from tensorflow.keras.optimizers import Adam
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, Embedding, Input, Flatten, Dropout

    import spm1d
    
    from os import listdir
    from os.path import isfile, join
    