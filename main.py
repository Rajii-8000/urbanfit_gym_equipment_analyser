import io
from typing import Optional
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from PIL import Image
from ultralytics import YOLO

app = FastAPI()

# Load the YOLOv8 model (using the fast, lightweight nano version)
model = YOLO("yolov8n.pt") 

# Define the response structure (perfectly matched to Streamlit's expectations!)
class GymAnalysisResult(BaseModel):
    is_gym_equipment: bool
    equipment_name: Optional[str] = None
    confidence_score: Optional[float] = None  # Corrected key!
    targeted_muscles: list[str] = []
    recommended_exercises: list[str] = []
    beginner_workout_plan: str = ""

# Common gym equipment classes from the COCO dataset that YOLOv8 natively recognizes
GYM_CLASSES = {
    "sports ball", "baseball bat", "baseball glove", "skateboard", 
    "surfboard", "tennis racket", "bottle", "dumbbell"
}

@app.post("/analyze", response_model=GymAnalysisResult)
async def analyze(file: UploadFile = File(...)):
    # 1. Read the uploaded image bytes
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    
    # 2. Run the YOLOv8 ML model on the image
    results = model(image)
    
    is_gym = False
    detected_item = None
    max_conf = 0.0
    
    # 3. Process the model's predictions
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            confidence = float(box.conf[0])
            
            # Check if the detected item matches any of our gym labels
            if label in GYM_CLASSES or "dumbbell" in label or "bench" in label:
                is_gym = True
                if confidence > max_conf:
                    max_conf = confidence
                    detected_item = label

    # 4. Generate the corresponding workout data if gym equipment is found
    if is_gym:
        # Clean up the display name (e.g., make 'dumbbell' look nice as 'Dumbbell')
        equipment_display_name = detected_item.replace("_", " ").title() if detected_item else "Gym Equipment"
        
        targeted_muscles = ["Quads", "Hamstrings", "Glutes", "Lower Back", "Core"]
        recommended_exercises = ["Squats", "Deadlifts", "Barbell Rows", "Lunges"]
        beginner_workout_plan = (
            "Perform 3 sets of 8-10 reps of Squats, "
            "followed by 3 sets of 8 reps of Deadlifts. "
            "Focus on maintaining a straight back and tight core!"
        )
    else:
        equipment_display_name = "Unknown Gym Equipment"
        targeted_muscles = []
        recommended_exercises = []
        beginner_workout_plan = "No recognizable gym equipment detected. Please upload a clearer photo!"

    # 5. Return the response to Streamlit
    return GymAnalysisResult(
        is_gym_equipment=is_gym,
        equipment_name=equipment_display_name,
        confidence_score=round(max_conf, 2) if is_gym else None,
        targeted_muscles=targeted_muscles,
        recommended_exercises=recommended_exercises,
        beginner_workout_plan=beginner_workout_plan
    )

@app.get("/")
def home():
    return {"message": "Urbanfit FastAPI Backend is Running!"}