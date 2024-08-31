import streamlit as st
import tensorflow as tf
import numpy as np
import google.generativeai as genai
import os
from PIL import Image

# Configure Google Generative AI
genai.configure(api_key="AIzaSyC9ofeMhsLxxB6pw6bENBZUPlveLY_osz0")
os.environ["GOOGLE_API_KEY"] = "AIzaSyC9ofeMhsLxxB6pw6bENBZUPlveLY_osz0"

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

model2 = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    safety_settings=safety_settings,
    generation_config={
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    },
    system_instruction=(
        "You are a helpful personal assistant chatbot"
    ),
)

chat = model2.start_chat()

def chat_with_me(question):
    try:
        response = chat.send_message(question)
        return response.text 
    except Exception as e:
        return f"Error: {str(e)}"

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

# Sidebar
st.sidebar.title("Dashboard")

app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition", "Chat Support"])
st.sidebar.markdown("""
### Connect with Us
<a href="https://github.com/SIT-SIH-2K24" target="_blank">
    <img src="https://img.icons8.com/material-outlined/24/000000/github.png" style="vertical-align: middle;"/>
</a>
<a href="https://www.linkedin.com/in/your-linkedin-profile/" target="_blank">
    <img src="https://img.icons8.com/material-outlined/24/000000/linkedin.png" style="vertical-align: middle;"/>
</a>
<a href="https://www.instagram.com/your-instagram-profile/" target="_blank">
    <img src="https://img.icons8.com/material-outlined/24/000000/instagram-new.png" style="vertical-align: middle;"/>
</a>
""", unsafe_allow_html=True)

# Main Page
if app_mode == "Home":
    st.markdown("""
    <style>
    .typewriter h1 {
        font-family: 'Courier New', Courier, monospace;
        font-size: 3.5em;
        color: white;
        overflow: hidden;
        border-right: .15em solid orange; /* The typewriter cursor */
        white-space: nowrap; /* Keeps the text on a single line */
        margin: 0 auto;
        animation: 
            typing 3.5s steps(40, end),
            blink-caret .75s step-end infinite;
    }
    @keyframes typing {
        from { width: 0; }
        to { width: 100%; }
    }
    @keyframes blink-caret {
        from, to { border-color: black; }
        50% { border-color: orange; }
    }
    </style>
    <div class="typewriter">
        <h1>KRISHI AVARANAM</h1>
    </div>
    """, unsafe_allow_html=True)

    # Background Image
    st.markdown("""
    <style>
    .main {
        background-image: url("https://cdn11.bigcommerce.com/s-tjrce8etun/product_images/uploaded_images/leave-with-fungus.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        padding: 50px;
        color:white;
    }
    </style>
    """, unsafe_allow_html=True)


    st.markdown("""
    Welcome to KRISHI AVARANAM! 🌿🔍
    
    A AI DRIVEN CROP DISEASE PREDICTION AND MANAGEMENT SYSTEM.

    Our mission is to help in identifying plant diseases efficiently.
    Discover the future of plant disease detection! Upload a plant image, and our state-of-the-art system will rapidly evaluate it for any disease signs. 
    Partner with us to enhance crop health and secure a thriving harvest through innovative, precise analysis. Let’s work together for healthier, more resilient plants.


    ### How It Works
    1. Upload Image: Go to the Disease Recognition page and upload an image of a plant with suspected diseases.
    2. Analysis: Our system will process the image using advanced algorithms to identify potential diseases.
    3. Results: View the results and recommendations for further action.

    ### Why Choose Us?
    - Accuracy: Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - User-Friendly: Simple and intuitive interface for seamless user experience.
    - Fast and Efficient: Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Navigate to the Disease Recognition page in the sidebar to upload your plant image and witness the capabilities of our cutting-edge Plant Disease Recognition System. This powerful tool will analyze your image in-depth, providing you with accurate insights and disease detection. Explore the technology that’s transforming plant health management and optimize your crop care with just a few clicks.

    ### About Us
    Learn more about the project, our team, and our goals on the About page.

    ### Recent Work
    - Successfully integrated Google Generative AI for providing chatbot support within the application.
    - Enhanced the machine learning model for better accuracy and faster predictions.
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown(""" #### About the Project
                This project harnesses the power of machine learning to revolutionize plant disease detection through image analysis. By employing TensorFlow for precise model predictions and Google Generative AI for interactive chatbot support, our system is crafted to aid farmers and researchers in diagnosing plant health with unparalleled efficiency.

Dataset :
We use an enhanced dataset derived from an original collection, comprising approximately 87,000 RGB images of both healthy and diseased crop leaves. These images are meticulously categorized into 38 distinct classes, representing a wide array of crops and disease types.

Dataset Breakdown:

Training Set: 70,295 images for model training.
Testing Set: 33 images for evaluating model performance.
Validation Set: 17,572 images to fine-tune and validate model accuracy.
Key Features
State-of-the-Art ML Models: Our system employs advanced machine learning algorithms to achieve high precision in detecting plant diseases.
Instant Chat Support: With Google Generative AI integration, users receive real-time assistance, answering queries and providing support related to plant health.

Achievements : 
Optimized Performance: Significant enhancements in model accuracy and performance through architectural fine-tuning.
Enhanced User Experience: A user-friendly interface designed to facilitate seamless interaction with the system, ensuring ease of use and accessibility.

Future Goals : 

Dataset Expansion: Broaden the dataset to include a wider variety of plant species and disease types, enhancing the model's versatility.
Real-Time Feedback: Implement real-time data processing capabilities to deliver immediate analysis results upon image upload.
Chatbot Enhancement: Further develop the chatbot to provide more tailored advice and personalized support for users, enriching the overall user experience.
                """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    
    if st.button("Show Image"):
        if test_image is not None:
            st.image(test_image, use_column_width=True)
        else:
            st.warning("Please upload an image before attempting to display it.")

    if st.button("Predict"):
        if test_image is not None:
            st.snow()
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            # Reading Labels
            class_name = ['Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
                        'Blueberry__healthy', 'Cherry(including_sour)_Powdery_mildew', 
                        'Cherry_(including_sour)healthy', 'Corn(maize)_Cercospora_leaf_spot Gray_leaf_spot', 
                        'Corn_(maize)Common_rust', 'Corn_(maize)Northern_Leaf_Blight', 'Corn(maize)_healthy', 
                        'Grape__Black_rot', 'Grape_Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 
                        'Grape__healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Peach___Bacterial_spot',
                        'Peach__healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell__healthy', 
                        'Potato__Early_blight', 'Potato_Late_blight', 'Potato__healthy', 
                        'Raspberry__healthy', 'Soybean_healthy', 'Squash__Powdery_mildew', 
                        'Strawberry__Leaf_scorch', 'Strawberry_healthy', 'Tomato__Bacterial_spot', 
                        'Tomato__Early_blight', 'Tomato_Late_blight', 'Tomato__Leaf_Mold', 
                        'Tomato__Septoria_leaf_spot', 'Tomato__Spider_mites Two-spotted_spider_mite', 
                        'Tomato__Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Tomato_mosaic_virus',
                        'Tomato___healthy']
            st.success("Model is Predicting it's a {}".format(class_name[result_index]))
        else:
            st.warning("Please upload an image before attempting to predict.")

# Chat Support Page
elif app_mode == "Chat Support":
    st.header("Chat Support")

    # Initialize session state for chat history if not already present
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Function to display chat history
    def display_chat():
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.write(f"You: {msg['content']}")
            else:
                st.write(f"Bot: {msg['content']}")

    # Display existing chat history
    display_chat()

    # Function to handle sending messages
    def send_message():
        user_message = st.session_state.chat_input
        if user_message:
            st.session_state.messages.append({"role": "user", "content": user_message})
            response = chat_with_me(user_message)
            st.session_state.messages.append({"role": "bot", "content": response})
            # Clear the input field
            st.session_state.chat_input = ""
            # Scroll to bottom
            st.markdown("<script>window.scrollTo(0, document.body.scrollHeight);</script>", unsafe_allow_html=True)

    # User input with Enter key sending message
    user_input = st.text_input("Type your message here:", key="chat_input", on_change=send_message)
    
    # Send button
    st.button("Send", on_click=send_message)
