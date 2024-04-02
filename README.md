# deepfake
## About
 In recent years, the proliferation of deep fake technology has raised significant concerns
regarding the authenticity and trustworthiness of digital media. Deep fake images, synthesized
using sophisticated machine learning algorithms, can convincingly manipulate visual content
to depict events or individuals that never existed or were never present. With the potential to
deceive and manipulate, the emergence of deep fake images poses a formidable challenge to
society, threatening the integrity of media, privacy, and even democracy itself.
In response to this pressing issue, researchers and technologists have been actively developing
methods to detect and mitigate the spread of deep fake content. Among these efforts,
Convolutional Neural Networks (CNNs) have emerged as a powerful tool for identifying
manipulated images by analyzing subtle patterns and inconsistencies that are often
imperceptible to the human eye. By leveraging the deep learning capabilities of CNNs,
researchers can train models to distinguish between authentic and deepfake images with a high
degree of accuracy.
Moreover, the accessibility of machine learning frameworks and web development tools has
paved the way for innovative solutions to combat the proliferation of deepfake content.
Streamlit, a popular open-source framework for building interactive web applications with
Python, offers a convenient platform for deploying deepfake detection models to a broader
audience. By integrating CNN-based image analysis algorithms into a user-friendly web
interface, we can empower individuals to verify the authenticity of visual content and raise
awareness about the prevalence of deepfake manipulation.
## Problem Definition

The proliferation of deepfake technology poses a significant threat to the integrity of visual
media. Deepfake images, which are manipulated or synthesized images often created with the
assistance of artificial intelligence, can be used to spread misinformation, defame individuals,
or even manipulate public opinion. Detecting these fraudulent images has become increasingly
challenging as deepfake techniques advance.
The problem at hand is to develop an effective solution for detecting deepfake images using
Convolutional Neural Networks (CNN) and deploy it through a user-friendly interface using
Streamlit, a web application framework. The primary challenges include:

1. Complexity of Deepfake Generation: Deepfake techniques are continually evolving,
making it difficult to develop robust detection methods. The model needs to be trained
on a diverse dataset encompassing various deepfake generation methods and image
qualities.
2. Performance and Accuracy: The detection model must achieve high accuracy in
distinguishing between authentic and deepfake images while maintaining reasonable
computational efficiency. Balancing accuracy with speed is crucial for real-time
applications.
3. User Interface Design: Creating an intuitive and user-friendly interface using Streamlit
is essential for widespread adoption of the detection tool. The interface should allow
users to upload images easily, view detection results, and understand the confidence
levels of the predictions.
4. Generalization and Robustness: The model should generalize well to detect deepfake
images across different contexts, including various types of manipulation techniques,
resolutions, and compression artifacts. Robustness to adversarial attacks should also be
considered.
5. Ethical Considerations: As deepfake detection technology becomes more
sophisticated, ethical concerns regarding privacy, consent, and potential misuse must
be addressed. Ensuring the responsible deployment and use of the detection tool is
paramount.
Overall, the goal is to develop a deepfake image detection system that is accurate, efficient,
user-friendly, and ethically sound. By leveraging CNNs for deepfake detection and deploying
the solution through a Streamlit website, we aim to empower users to identify and mitigate the
risks associated with the spread of deceptive visual media.

## Flow Diagram:
![image](https://github.com/Kumaravel655/deepfake/assets/75235334/5ee10597-d368-4fdc-a50e-3a1ded1d49b6)

## Program
```python
# -------------------
# IMPORTS
# -------------------
import streamlit as st
import numpy as np
import pickle
import os
from PIL import Image, ImageOps
from streamlit_image_select import image_select
from tensorflow.keras.models import model_from_json
import time

# -------------------
# MAIN
# -------------------
def main():
   
   st.title("Deepfake Detector:")


# function to load and cache pretrained model
@st.cache_resource()
def load_model():
    path = "../dffnetv2B0"
    # Model reconstruction from JSON file
    with open(path + '.json', 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(path + '.h5')
    return model

 
# function to preprocess an image and get a prediction from the model
def get_prediction(model, image):
    
    open_image = Image.open(image)
    resized_image = open_image.resize((256, 256))
    np_image = np.array(resized_image)
    reshaped = np.expand_dims(np_image, axis=0)

    predicted_prob = model.predict(reshaped)[0][0]
    
    if predicted_prob >= 0.5:
        return f"Real, Confidence: {str(predicted_prob)[:4]}"
    else:
        return f"Fake, Confidence: {str(1 - predicted_prob)[:4]}"

# generate selection of sample images 
@st.cache_data()
def load_images():
  real_images = ["images/Real/" + x for x in os.listdir("images/Real/")]
  fake_images = ["images/Fake/" + x for x in os.listdir("images/Fake/")]
  image_library = real_images + fake_images
  image_selection = np.random.choice(image_library, 20, replace=False)

  return image_selection

# load model
classifier = load_model()
images = load_images()

def game_mode():
  st.header("Game Mode")
  st.subheader("Can you beat the model?")
 
  selected_image = image_select(
    "Click on an image below to guess if it is real of fake:", 
    images,
    return_value="index")
  prediction = get_prediction(classifier, images[selected_image])
  true_label = 'Fake' if 'fake' in images[selected_image].lower() else 'Real'

  #st.text(true_label)

  st.subheader("Is this image real or fake?")
  st.image(images[selected_image])
    
  if st.button("It's Real"):
    st.text("You guessed:")
    st.subheader("*Real*")
    st.text("The Deepfake Detector model guessed...")
    time.sleep(1)
    st.subheader(f"*{prediction}*")
    st.text("The truth is...")
    time.sleep(1)
    st.subheader(f"***It's {true_label}!***")

  if st.button("It's Fake"):
    st.text("You guessed:")
    st.subheader("*Fake*")
    st.text("The Deepfake Detector model guessed...")
    time.sleep(1)
    st.subheader(f"*{prediction}*")
    st.text("The truth is...")
    time.sleep(1)
    st.subheader(f"***It's {true_label}!***")


def detector_mode():

  st.header("Detector Mode")
  st.subheader("Upload an Image to Make a Prediction")

  # upload an image
  uploaded_image = st.file_uploader("Upload your own image to test the model:", type=['jpg', 'jpeg'])

  # when an image is uploaded, display image and run inference
  if uploaded_image is not None:
    st.image(uploaded_image)
    st.text(get_prediction(classifier, uploaded_image))


page = st.sidebar.selectbox('Select Mode',['Detector Mode','Game Mode']) 

if page == 'Game Mode':
  game_mode()
else:
  detector_mode()

# -------------------
# SCRIPT/MODULE CHECKER
# -------------------
if __name__ == "__main__":
    main()

```

model url : https://drive.google.com/file/d/16uyhDMZg1B5NiDq5aXEau55VdUF0Rsq4/view?usp=sharing

## Outputs:
### Home 
![image](https://github.com/Kumaravel655/deepfake/assets/75235334/e1486bef-2d48-4004-bbef-1853da80a168)
### Fake image detection
![image](https://github.com/Kumaravel655/deepfake/assets/75235334/076788eb-4cfd-4bfd-ae7e-20da4f932931)
### Real image detection
![image](https://github.com/Kumaravel655/deepfake/assets/75235334/396f06cc-b27a-4102-9b82-5e6d2e399aed)

## conclusion
Deepfake technology poses a significant threat to the integrity of digital content and the trustworthiness of online information.
Our project has demonstrated the effectiveness of using Convolutional Neural Networks (CNNs) for deepfake detection, showcasing promising results in accurately identifying manipulated content.
However, the rapid evolution of deepfake techniques necessitates ongoing research and innovation to stay ahead of malicious actors.
Collaboration across academia, industry, and policymakers is essential to develop robust detection methods and mitigate the risks associated with deepfake manipulation.
Looking forward, continued efforts are needed to enhance the scalability, reliability, and transparency of deepfake detection systems, ensuring a safer digital environment for all users.


