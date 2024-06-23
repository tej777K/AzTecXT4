import os
import logging
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from configparser import ConfigParser

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

config = ConfigParser()
config.read("config.ini")
endpoint = config['CREDENTIALS']['ENDPOINT']
key = config['CREDENTIALS']['KEY']

if not endpoint or not key:
    raise ValueError("Set ENDPOINT and KEY in config.ini file.")


client = ImageAnalysisClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    try:
        os.makedirs(app.config['UPLOAD_FOLDER'])
        logging.info(f"Created directory {app.config['UPLOAD_FOLDER']}")
    except Exception as e:
        logging.error(f"Failed to create directory {app.config['UPLOAD_FOLDER']}: {e}")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'file' not in request.files:
            logging.info("No file part in the request.")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            logging.info("No file selected.")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(file_path)
                logging.info(f"File saved to {file_path}")

                with open(file_path, "rb") as image_stream:
                    img = image_stream.read()
                    result = client.analyze(
                        image_data = img,
                        visual_features=[VisualFeatures.CAPTION, VisualFeatures.READ],
                        gender_neutral_caption=True,
                    )
                caption = result.caption.text if result.caption else "No caption found"
                os.remove(file_path)
                return render_template("index.html", caption=caption)
            except Exception as e:
                logging.error(f"Error processing file {filename}: {e}")
                return redirect(request.url)
    return render_template("index.html", caption="")

if __name__ == "__main__":
    app.run()
