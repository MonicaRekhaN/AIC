from flask import Flask, render_template, request
import numpy as np



app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/after', methods=['GET', 'POST'])
def after():

   
    return render_template('after.html', data="Result")

if __name__ == '__main__':
    app.run(debug=True)


