from pathlib import Path
import streamlit as st
import PIL
import helper
import settings

# Set Streamlit page configuration
st.set_page_config(
    page_title="Ocean Waste Classification using YOLOv8",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    /* Body style */
    body {
        background-color: #2196F3; /* Blue ocean background */
        color: #ffffff; /* White text color */
        font-family: Arial, sans-serif; /* Specify font family */
    }

    /* Button style */
    .stButton>button {
        background-color: #64B5F6; /* Light blue button background */
        color: #ffffff; /* White button text color */
        border-radius: 20px; /* Rounded corners */
        padding: 10px 20px; /* Padding for button */
        font-weight: bold; /* Bold button text */
        transition: background-color 0.3s ease; /* Smooth transition on hover */
    }

    .stButton>button:hover {
        background-color: #42A5F5; /* Darker blue on hover */
    }

    /* Text input style */
    .stTextInput>div>div>input {
        color: #ffffff; /* White text input color */
        background-color: #64B5F6; /* Light blue input background */
        border-radius: 10px; /* Rounded corners */
        padding: 10px; /* Padding for input */
    }

    /* Selectbox style */
    .stSelectbox>div>div>div {
        color: #ffffff; /* White selectbox text color */
        background-color: #64B5F6; /* Light blue selectbox background */
        border-radius: 10px; /* Rounded corners */
        padding: 10px; /* Padding for selectbox */
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Main page heading
st.title("Ocean Waste Classification using YOLOv8")

# Sidebar
st.sidebar.header("ML Model Config")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Detection'])

confidence = 0.4  # Default confidence threshold

# Selecting Detection Or Segmentation
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video ")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

else:
    st.error("Please select a valid source type!")
