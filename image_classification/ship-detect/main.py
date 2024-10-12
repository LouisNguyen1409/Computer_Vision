from fastapi import FastAPI
from ultralytics import YOLO
import numpy as np
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

class Image(BaseModel):
	image_url: str

origins = ["*"]

app = FastAPI()
app.add_middleware(
	CORSMiddleware,
	allow_origins=origins,
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


model_path = "./runs/classify/train/weights/best.pt"
model = YOLO(model_path)

@app.post("/classify")
def classify_ship(image: Image):
	image_url = image.image_url

	output = model.predict(image_url)[0]
	probs = output.probs.numpy().data
	names_dict = output.names
	predict = names_dict[np.argmax(probs)]
	score = probs[np.argmax(probs)] * 100
	return {"prediction": predict, 'score': score, 'image_url': image_url}
