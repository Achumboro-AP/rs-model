import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load your trained model (replace 'model.h5' with the path to your model file)
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('enet89loss46_model.h5')
    return model

model = load_model()

st.title('Reed Sternberg Cell Classification AI Model for Hodgkins Lymphoma')

# File uploader allows user to add their own image
uploaded_file = st.file_uploader(label='Upload an image for analysis', type=['jpg', 'png'])

if uploaded_file is not None:
    # Convert the file to an image
    image = Image.open(uploaded_file).convert('RGB')

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write('')

    # Predicting the class of the image using the loaded model
    with st.spinner('Analyzing the image...'):
        # Resize the image to the target size required by the model
        img = image.resize((290, 290))

        # Convert the image to a numpy array
        img_array = np.array(img)

        # Expand dimensions to represent batch size
        img_array = np.expand_dims(img_array, axis=0)

        # Normalize the image if the model expects normalized inputs
        img_array /= 255.0

        # Predict the class of the image
        prediction = model.predict(img_array)

        # You may need to modify the following lines to suit your model's output
        prediction_score = prediction.max()  # Assuming your model outputs class probabilities
        result = 'Positive' if prediction.argmax() == 1 else 'Negative'

    st.success('Analysis complete')
    st.write(f'Prediction Score: {prediction_score:.2%}')
    st.write(f'Result: {result}')

# Footer
st.markdown('© [Andrews Owusu] 2024')
