#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
from flask import Flask, render_template, request, jsonify
from utils import process_speaker, get_speakers_list


# In[ ]:


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    speakers = get_speakers_list()
    return render_template('index.html', speakers=speakers)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part", "success": False})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file", "success": False})
    
    if file:
        # Save the file
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        # Get mode from form
        mode = request.form.get('mode', 'check')
        
        # Process the file using utils
        try:
            result = process_speaker(filename, mode)
            result["success"] = True
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e), "success": False})
        finally:
            # Clean up uploaded file
            if os.path.exists(filename):
                os.remove(filename)

if __name__ == '__main__':
    app.run(debug=True)

