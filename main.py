import tempfile
import os
from flask import Flask, request, redirect, send_file, render_template
from tensorflow.keras.models import load_model
from io import BytesIO
from skimage import io
import base64
import glob
import numpy as np
from PIL import Image
from skimage.transform import resize
import base64
import numpy as np

model= load_model('hiragana_predictor.keras', compile=False)
app = Flask(__name__)

@app.route("/")
def main():
    return render_template("index.html")


@app.route('/upload', methods=['POST'])
def upload():
    try:
        # check if the post request has the file part
        img_data = request.form.get('myImage').replace("data:image/png;base64,","")
        aleatorio = request.form.get('numero')
        print(aleatorio)
        with tempfile.NamedTemporaryFile(delete = False, mode = "w+b", suffix='.png', dir=str(aleatorio)) as fh:
            fh.write(base64.b64decode(img_data))
        #file = request.files['myImage']
        print("Image uploaded")
    except Exception as err:
        print("Error occurred")
        print(err)

    return redirect("/", code=302)


@app.route('/prepare', methods=['GET'])
def prepare_dataset():
    images = []
    d = ["あ","え","い","お","う"]
    digits = []
    for digit in d:
      filelist = glob.glob('{}/*.png'.format(digit))
      images_read = io.concatenate_images(io.imread_collection(filelist))
      images_read = images_read[:, :, :, 3]
      digits_read = np.array([digit] * images_read.shape[0])
      images.append(images_read)
      digits.append(digits_read)
    images = np.vstack(images)
    digits = np.concatenate(digits)
    np.save('X.npy', images)
    np.save('y.npy', digits)
    return render_template("prepare.html")

@app.route('/X.npy', methods=['GET'])
def download_X():
    return send_file('./X.npy')
@app.route('/y.npy', methods=['GET'])
def download_y():
    return send_file('./y.npy')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        """
        img_data = request.form.get('myImage').replace("data:image/png;base64,","")
        img_binary = base64.b64decode(img_data)

        # Convert binary data to PIL image
        image = Image.open(BytesIO(img_binary))

        # Convert the image to grayscale
        img_gray = image.convert('L')

        # Resize the image and convert it to a NumPy array
        img_resized = np.array(resize(np.array(img_gray), (28, 28)))

        # Convert the image data to 8-bit integer format
        img_resized = (img_resized * 255).astype(np.uint8)

        # Save the image
        Image.fromarray(img_resized, 'L').save('predict.png')

        # Reshape the array
        img_array = img_resized.reshape(1, 28, 28, 1)

        """

        img_data = request.form.get('myImage').replace("data:image/png;base64,","")
        img_binary = base64.b64decode(img_data)

        # Convert binary data to PIL image
        image = Image.open(BytesIO(img_binary))

        # Keep the image in RGBA format
        img_rgba = np.array(image)

        # Resize the image
        img_resized = np.array(resize(img_rgba, (28, 28)))

        # Extract the alpha channel and convert the image data to 8-bit integer format
        img_resized = (img_resized[:, :, 3] * 255).astype(np.uint8)

        # Normalize the image data
        img_resized = img_resized.astype('float32') / 255.0

        # Reshape the array
        img_array = img_resized.reshape(1, 28, 28)

        # Add an extra dimension if needed
        if img_array.ndim == 3:
            img_array = img_array[..., None]

        # Predict
        prediction = model.predict(img_array)
        print(prediction)
        etiquetas = {0: 'あ', 1: 'い', 2: 'う', 3: 'え', 4: 'お'}

        valor = np.argmax(prediction)

        print(f"Valor predicho: {valor}")

        if valor in etiquetas:
            kind = etiquetas[valor]
            print(f"Kind: {kind}")
        else:
            print("El valor predicho no tiene una etiqueta asociada.")
     
        print("Image charged")
        return render_template('predict.html',value=kind)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    digits = ["あ","え","い","お","う"]
    for d in digits:
        if not os.path.exists(str(d)):
            os.mkdir(str(d))
    app.run()
