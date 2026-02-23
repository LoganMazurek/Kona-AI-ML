"""
Simple Flask app for routing between subdomains.
Serves as the landing page for loganmazurek.com
"""

from flask import Flask, render_template
import os

# Get the path to the templates directory (one level up from source/)
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
app = Flask(__name__, template_folder=template_dir)

@app.route('/')
def home():
    """Render the landing page router"""
    return render_template('router.html')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8000)
