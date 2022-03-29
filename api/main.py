from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

MODEL = tf.keras.models.load_model('../models/plantvillage_model_v1.h5')
CLASS_NAMES = ['Tomato_Bacterial_spot',
               'Tomato_Early_blight',
               'Tomato_Late_blight',
               'Tomato_Leaf_Mold',
               'Tomato_Septoria_leaf_spot',
               'Tomato_Spider_mites_Two_spotted_spider_mite',
               'Tomato__Target_Spot',
               'Tomato__Tomato_YellowLeaf__Curl_Virus',
               'Tomato__Tomato_mosaic_virus',
               'Tomato_healthy']

def read_file_as_image(data):
    file_bytes = np.array(Image.open(BytesIO(data)))
    return file_bytes

@app.get("/ping")
async def ping():
    return "pong"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = read_file_as_image(await file.read())
    img_batch = np.expand_dims(img_bytes, axis=0)
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)