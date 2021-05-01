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
# from flask_ngrok import run_with_ngrok


import warnings
warnings.filterwarnings("ignore")

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
        session['folder'] = folder

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
        
        session['path'] = path
        session['input_file'] = input_file
        session['output_file'] = input_file + '_parsed'

        return render_template('index.html', headers=session['columns'], data = dataset, preview = True, categorize=True)
    
    else:
        flash('No file specified', 'error')
        return redirect(url_for('upload_files'))

@app.route('/categories', methods=['POST'])
def categorize():

    target = request.form['target']
        
    print(target)

    session['target'] = target

    parse.parse_csv(session['path'], session['input_file'], session['output_file'], session['target'])
    split.split_csv(session['path'], session['input_file'])

    flash('Target Column assigned', 'info')
    return redirect(url_for('upload_files'))

@app.route('/predict',methods=['POST'])
def predict():

    syn_data_files = []

    for i in range(1, 5):
        syn_data = sample.sample_tablegan(session['folder'], session['input_file']+'_parsed', "./datasets", sample_synthetic_rows=1000)
        syn_data_files.append(syn_data)

    new_data = pd.concat(syn_data_files)
    new_data.to_csv('new_data.csv')
    new_data = new_data.head(10)
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
