from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf

# โหลดโมเดลจากไฟล์ .h5
model = tf.keras.models.load_model('models/modelVGG16.h5')

app = Flask(__name__)
# สร้าง mapping สำหรับคลาส
class_mapping = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D'
}

def preprocess_image(image):
    image = image.resize((224, 224))  # ปรับขนาดให้ตรงกับ input ของโมเดล
    image = np.array(image)
    image = image / 255.0  # normalize
    image = np.expand_dims(image, axis=0)  # เพิ่มมิติให้ตรงกับโมเดล
    return image

@app.route('/predict_indigo_grade', methods=['POST'])
def predict_herb_grade():
    try:
        data = request.get_json()
        base64str = data['base64str']
        image_data = base64.b64decode(base64str)
        image = Image.open(BytesIO(image_data))

        # เตรียมภาพก่อนส่งเข้าโมเดล
        processed_image = preprocess_image(image)

        # ใช้โมเดลในการทำนาย
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions, axis=-1)[0]

        # แทนที่ตัวเลขคลาสด้วยชื่อที่แมปไว้
        predicted_class_name = class_mapping.get(predicted_class, "Unknown")

        return jsonify({'predicted_class': predicted_class_name})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
