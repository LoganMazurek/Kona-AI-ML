"""
Simple Flask app for routing between subdomains.
Serves as the landing page for loganmazurek.com
"""

from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    """Render the landing page router"""
    return render_template('router.html')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8000)
