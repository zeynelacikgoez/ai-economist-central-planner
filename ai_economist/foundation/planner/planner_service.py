from flask import Flask, request, jsonify
from ai_economist.planner.central_planner import CentralPlanner
from ai_economist.foundations.engine import EconomyEnvironment
from functools import wraps
import os
from dotenv import load_dotenv
import logging

# Laden der Umgebungsvariablen
load_dotenv()

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialisierung der Umgebung und des zentralen Planers
environment = EconomyEnvironment()
central_planner = CentralPlanner(environment)
central_planner.model.model.summary()  # Optional: Modellzusammenfassung ausgeben

# Einfaches Token für Authentifizierung (für Demo-Zwecke)
API_TOKEN = os.getenv('API_TOKEN', 'securetoken123')

def authenticate(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if token != f"Bearer {API_TOKEN}":
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/generate_plan', methods=['POST'])
@authenticate
def generate_plan():
    try:
        data = request.json
        # Optional: Aktualisiere die Umgebung basierend auf den erhaltenen Daten
        central_planner.generate_plan()
        return jsonify(central_planner.plan), 200
    except Exception as e:
        logger.error(f"Fehler bei der Planerstellung: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/train_model', methods=['POST'])
@authenticate
def train_model():
    try:
        experiences = request.json
        central_planner.train_model(experiences)
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        logger.error(f"Fehler beim Trainieren des Modells: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/reset', methods=['POST'])
@authenticate
def reset():
    try:
        central_planner.reset()
        environment.reset()
        return jsonify({'status': 'reset'}), 200
    except Exception as e:
        logger.error(f"Fehler beim Zurücksetzen: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
