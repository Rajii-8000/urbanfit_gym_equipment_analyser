from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io
import uvicorn

app = FastAPI()

# 1. Load your model
# Ensure 'best.pt' is in the same folder as main.py or update the pa
model = YOLO('runs/classify/train-3/weights/best.pt')

EXERCISE_MAP = {
    "benchPress": {
        "muscles": ["Chest", "Triceps", "Shoulders"],
        "exercises": ["Flat Bench Press", "Incline Dumbbell Press", "Push-ups"],
        "plan": "Perform 3 sets of 10 repetitions with a moderate weight."
    },
    "dumbell": {
        "muscles": ["Biceps", "Triceps", "Forearms"],
        "exercises": ["Bicep Curls", "Hammer Curls", "Dumbbell Rows"],
        "plan": "3 sets of 12 repetitions per arm."
    },
    "kettleBell": {
        "muscles": ["Core", "Glutes", "Shoulders"],
        "exercises": ["Kettlebell Swings", "Goblet Squats", "Turkish Get-ups"],
        "plan": "3 sets of 15 swings focusing on hip hinge form."
    },
    "pullBar": {
        "muscles": ["Back", "Biceps", "Lats"],
        "exercises": ["Pull-ups", "Chin-ups", "Hanging Leg Raises"],
        "plan": "Aim for 3 sets of as many reps as possible, or use a resistance band for assistance."
    },
    "treadMill": {
        "muscles": ["Legs", "Cardio"],
        "exercises": ["Walking", "Jogging", "Interval Sprints"],
        "plan": "Start with a 5-minute warm-up walk, followed by 10 minutes of light jogging."
    }
}

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # Read the image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    
    # Run model prediction
    results = model.predict(image)
    
    # Get the top prediction
    # This assumes your model is a classifier. If it's a detector, 
    # you might need to adjust based on results[0].boxes
    top_class_index = results[0].probs.top1
    detected_class = results[0].names[top_class_index]
    confidence = float(results[0].probs.top1conf)
    
    # Get info from our map
    info = EXERCISE_MAP.get(detected_class, {
        "muscles": ["N/A"], "exercises": ["N/A"], "plan": "Equipment not recognized."
    })
    
    return {
        "is_gym_equipment": True,
        "equipment_name": detected_class,
        "confidence_score": f"{confidence:.2%}",
        "targeted_muscles": info["muscles"],
        "recommended_exercises": info["exercises"],
        "beginner_workout_plan": info["plan"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)