import os
from werkzeug.utils import secure_filename
from flask import Flask,flash,request,redirect,send_file,render_template, url_for, abort, session
import numpy as np
import pandas as pd
from sdv.demo import load_tabular_demo
from sdv.tabular import CTGAN
from parse import parse, split
from generate import sample
import string
import random

import warnings
warnings.filterwarnings("ignore")

#app = Flask(__name__)
app = Flask(__name__)
app.config['UPLOAD_PATH'] = 'datasets'
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
        input_file, file_ext = os.path.splitext(filename)

        #Generating random alphanumeric string to avoid repetition of folder names
        letters = string.ascii_uppercase + string.digits
        folder = ''.join(random.choice(letters) for i in range(10))
        print(folder)

        path = os.path.join(app.config['UPLOAD_PATH'],folder)
        
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            flash('Unsupported file type', 'error')
            return redirect(url_for('home'))

        try:
            os.mkdir(path)
        except OSError as error:
            flash(error, 'error')
            print(error)
            return redirect(url_for('home'))
 
        uploaded_file.save(os.path.join(path, filename))
        session['file'] = filename
        df = pd.read_csv(os.path.join(path,filename))
        headers = list(df.columns)
        session['columns'] = headers
        df = df.head(5)
        dataset = df.to_html()
        session['dataset'] = dataset

        flash('File Uploaded!', 'info')

        output_file = input_file + '_parsed'
        ### FIX THIS
        target = 'Hazard' ### FIX THIS
        ### FIX THIS

        parse.parse_csv(path, input_file, output_file, target)
        split.split_csv(path, output_file)

        return render_template('index.html', headers=session['columns'], data = dataset, preview = True, categorize=True)
    
    else:
        flash('No file specified', 'error')
        return redirect(url_for('upload_files'))

@app.route('/categories', methods=['POST'])
def categorize():
    categories = {i:[] for i in ['categorical', 'ordinal', 'PII']}
    for col in session['columns']:
        option = request.form[col]
        print(option)
        if option == "1":
            categories['categorical'].append(col)
        elif option == "2":
            categories['ordinal'].append(col)
        elif option == "3":
            categories['PII'].append(col)

    print(categories)
    flash('Categories assigned', 'info')
    return redirect(url_for('upload_files'))

@app.route('/predict',methods=['POST'])
def predict():
    data = load_tabular_demo('student_placements')
    model = CTGAN()
    model.fit(data)
    new_data = model.sample(5)
    new_data.to_csv('new_data.csv')

    # sample.sample_tablegan("Hazards", "LibertyMutualHazard", "./datasets", output=f"datasets/Hazards/LibertyMutualHazard_train_output{i}.csv", sample_synthetic_rows=41600, preprocess_table=preprocess_hazards)


    flash('Dataset Generation Complete!', 'info')
    return render_template('index.html', data = session['dataset'], gendata=new_data.to_html(), generated=True, preview=True)

# Download API
@app.route("/download", methods = ['GET'])
def download_file():
    output = 'new_data.csv'
    file_path = os.path.join(cur_dir,output)
    flash('Dataset Downloading...', 'info')
    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)



