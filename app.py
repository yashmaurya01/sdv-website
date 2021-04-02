import os
from werkzeug.utils import secure_filename
from flask import Flask,flash,request,redirect,send_file,render_template, url_for, abort, session
import numpy as np
import pandas as pd
from sdv.demo import load_tabular_demo
from sdv.tabular import CTGAN

import warnings
warnings.filterwarnings("ignore")

#app = Flask(__name__)
app = Flask(__name__)
app.config['UPLOAD_PATH'] = 'uploads'
app.config['UPLOAD_EXTENSIONS'] = ['.txt', '.csv', '.data', '.names']
app.secret_key = 'abc123'

cur_dir = os.getcwd()


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            abort(400)
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
        session['file'] = filename
        headers = pd.read_csv(os.path.join(app.config['UPLOAD_PATH'],filename)).columns
        headers = list(headers)
        print(headers)
        session['columns'] = headers
        flash('File Uploaded!', 'info')
    return render_template('new_index.html', headers=session['columns'])

@app.route('/categories')
def categorize():
    for col in session['columns']:
        option = request.form[col]
        print(option)
    return redirect('upload_files')

@app.route('/predict',methods=['POST'])
def predict():
    data = load_tabular_demo('student_placements')
    model = CTGAN()
    model.fit(data)
    new_data = model.sample(5)
    new_data.to_csv('new_data.csv')
    flash('Dataset Generation Complete!', 'info')
    # return render_template('index.html',  tables=[new_data.to_html(classes='data', header="true")])
    return render_template('index.html')

# Download API
@app.route("/download", methods = ['GET'])
def download_file():
    output = 'new_data.csv'
    file_path = os.path.join(cur_dir,output)
    flash('Dataset Downloading...', 'info')
    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)



