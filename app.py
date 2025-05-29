import matplotlib
matplotlib.use('Agg')  # non-GUI backend to prevent tkinter errors

from flask import Flask, request, render_template, redirect, url_for
import os
import uuid
from model_utils import process_file, process_predict_file

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'
CSV_FOLDER = os.path.join('static', 'csv')
PLOT_FOLDER = os.path.join('static', 'images')

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(CSV_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/noisemaps")
def noisemaps():
    return render_template("noise_maps.html")

@app.route("/results", methods=["GET", "POST"])
def results():
    error = None
    models = None
    if request.method == "POST":
        f = request.files.get("file")
        if f and f.filename.endswith(".xlsx"):
            path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{f.filename}")
            f.save(path)
            models = process_file(path, PLOT_FOLDER)
        else:
            error = "Please upload a valid .xlsx file."
    return render_template("results.html", models=models, error=error)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    error = None
    predictions = None
    csv_file = None
    if request.method == "POST":
        f = request.files.get("file")
        if f and f.filename.endswith(".xlsx"):
            path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{f.filename}")
            f.save(path)
            df, csv_file = process_predict_file(path, CSV_FOLDER)
            predictions = df.to_dict(orient='records')
        else:
            error = "Please upload a valid .xlsx file."
    return render_template("predict.html", predictions=predictions, csv_file=csv_file, error=error)

if __name__ == "__main__":
    app.run(debug=True)