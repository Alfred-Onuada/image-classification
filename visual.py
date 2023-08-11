import streamlit as st
import tensorflow as tf
import os
import numpy as np
import cv2

# load the model
model = tf.keras.models.load_model(os.path.join('catdogmodel.h5'))

# define prediction function
def predict(img):
  resize = tf.image.resize(img, (256, 256))
  pred = model.predict(np.expand_dims(resize/255, 0))

  # calculate certainity
  certainty = 0
  if pred < 0.5:
    certainty = (1 - pred) * 100
  else:
     certainty = (pred - 0.5) * 2 * 100

  # the closer the value is to 0.5 the more the likely hood of it been wrong
  if pred > 0.45 and pred < 0.55:
    possible = ''

    if pred > 0.5:
       possible = 'DOG'
    else:
       possible = 'CAT'

    return '<h5>The above image could not be classified with great certainty, model would have gone with <em>{}</em> with a certainity of <em>{:.2f}</em> % </h5>'.format(possible, certainty[0][0])
  elif pred > 0.6:
     return '<h5>The uploaded image is an image of a <em>DOG</em>, certainity <em>{:.2f}</em> %</h5>'.format(certainty[0][0])
  else:
     return '<h5>The uploaded image is an image of a <em>CAT</em>, certainity <em>{:.2f}</em> %</h5>'.format(certainty[0][0])
  
st.header('Hi there 👋')

st.write('This applications is a simple classification model, built using tensorflow and the ideas of a Deep Convulutional Neural Network (CNN)')

st.write("The purpose of this application is a classify images as either cats or dogs, but the code is highly adjustable and can be modified to work with a variety of labels/classes as well before we get into the main gist let's take a look at some metrics")

st.write('This model was trained on a set of <b>4001</b> images of cats and <b>4001</b> images of dogs (not including auto randomized images used), so there is a perfect balance in the training set', unsafe_allow_html=True)

st.write('The validation set also shares the same numbers')

st.write("Let's head over to the jupyter notebook and look at the graphs and metrics representing the accuracy and loss over time of our model")

st.write("Time to get to the fun stuff")

# Create a file uploader component
files = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

for file in files:
   if file is not None:
      with open(os.path.join('visual_test', file.name),'wb') as f:
         f.write(file.read())

      # Load the image with OpenCV
      image = cv2.imread(os.path.join('visual_test', file.name))

      # Display the uploaded image
      st.image(image, channels="BGR")

      # Perform prediction
      prediction = predict(image)

      # Display the prediction result
      st.write(prediction, unsafe_allow_html=True)

      os.unlink(os.path.join('visual_test', file.name))