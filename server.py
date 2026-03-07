#!/usr/bin/env python3
"""
Dolphin Tinder Server

Simple static file server for the labeling frontend.
"""

from flask import Flask, send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')


@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_from_directory(BASE_DIR, 'index.html')


@app.route('/data/<path:filename>')
def serve_data(filename):
    """Serve files from the data directory."""
    return send_from_directory(DATA_DIR, filename)


@app.route('/pairs/<path:filename>')
def serve_pairs(filename):
    """Serve sampling pairs JSON files."""
    return send_from_directory(BASE_DIR, filename)


if __name__ == '__main__':
    app.run(debug=True, port=5001)
