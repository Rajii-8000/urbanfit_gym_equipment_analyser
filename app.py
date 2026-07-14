import streamlit as st
import requests
import io
from PIL import Image

st.set_page_config(page_title="Urbanfitt AI Workout Assistant", page_icon="🏋️‍♂️")

st.title("🏋️‍♂️ Urbanfitt Smart Image Classifier")
st.write("Upload a picture of gym equipment to detect its type and receive customized workout insights.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image File", use_container_width=True)
    
    if st.button("Run AI Classification"):
        with st.spinner("Processing image locally through ML pipeline..."):
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format=image.format if image.format else "JPEG")
            img_byte_arr = img_byte_arr.getvalue()
            
            files = {"file": (uploaded_file.name, img_byte_arr, uploaded_file.type)}
            
            try:
                response = requests.post("http://127.0.0.1:8000/analyze", files=files)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("is_gym_equipment"):
                        st.success(f"**Detected Equipment:** {data['equipment_name']} (Confidence: {data['confidence_score']})")
                        st.subheader("💪 Muscles Targeted")
                        st.write(", ".join(data["targeted_muscles"]))
                        st.subheader("🏃‍♂️ Suggested Exercises")
                        for idx, ex in enumerate(data["recommended_exercises"], 1):
                            st.write(f"**{idx}.** {ex}")
                        st.subheader("📅 Recommended Beginner Routine")
                        st.info(data["beginner_workout_plan"])
                    else:
                        st.warning(data.get("error_message"))
                else:
                    st.error("Backend Server returned an error.")
            except requests.exceptions.ConnectionError:
                st.error("Could not reach the backend. Ensure FastAPI is running on port 8000.")