import sys
sys.path.append('../functions')
from model import get_unet
from data import load_data
from keras import backend as K
import tensorflow as tf

x_train, y_train = load_data('train')
x_val, y_val = load_data('val')
model = get_unet(input_shape = x_train[0].shape)
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy']) 
model.fit(x_train, y_train, validation_data = (x_val, y_val))
