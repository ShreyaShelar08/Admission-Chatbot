from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os
import json
from models.chatbot_model import ChatbotModel

training_bp = Blueprint('training', __name__)

# Initialize chatbot model for training
trainer = ChatbotModel()

ALLOWED_EXTENSIONS = {'json'}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@training_bp.route('/upload-train', methods=['POST'])
def upload_and_train():
    """Upload dataset and train the model"""
    try:
        # Check if file is present
        if 'dataset' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['dataset']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Only JSON files are allowed'}), 400
        
        # Save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load training data
        trainer.load_training_data(filepath)
        
        # Train the model
        training_result = trainer.train()
        
        # Save the model
        save_result = trainer.save_model()
        
        return jsonify({
            'success': True,
            'message': 'Model trained and saved successfully',
            'training': training_result,
            'save': save_result
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@training_bp.route('/train', methods=['POST'])
def train():
    """Train model with JSON data"""
    try:
        data = request.get_json()
        
        if not data or 'intents' not in data:
            return jsonify({'error': 'Invalid training data format. Expected {"intents": [...]}'}), 400
        
        intents = data['intents']
        
        if not isinstance(intents, list):
            return jsonify({'error': 'Intents must be an array'}), 400
        
        # Validate intent structure
        for intent in intents:
            if 'tag' not in intent or 'patterns' not in intent or 'responses' not in intent:
                return jsonify({
                    'error': 'Each intent must have "tag", "patterns", and "responses" fields'
                }), 400
        
        # Set intents and train
        trainer.intents = intents
        training_result = trainer.train()
        
        # Save the model
        save_result = trainer.save_model()
        
        return jsonify({
            'success': True,
            'message': 'Model trained and saved successfully',
            'training': training_result,
            'save': save_result
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@training_bp.route('/load-model', methods=['GET'])
def load_model():
    """Load existing trained model"""
    try:
        result = trainer.load_model()
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@training_bp.route('/stats', methods=['GET'])
def get_stats():
    """Get model statistics"""
    try:
        stats = trainer.get_stats()
        return jsonify({
            'success': True,
            **stats
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@training_bp.route('/add-intent', methods=['POST'])
def add_intent():
    """Add new intent and retrain model"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['tag', 'patterns', 'responses']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        tag = data['tag']
        patterns = data['patterns']
        responses = data['responses']
        
        # Validate data types
        if not isinstance(patterns, list) or not isinstance(responses, list):
            return jsonify({'error': 'Patterns and responses must be arrays'}), 400
        
        if not patterns or not responses:
            return jsonify({'error': 'Patterns and responses cannot be empty'}), 400
        
        # Add intent
        add_result = trainer.add_intent(tag, patterns, responses)
        
        # Retrain the model
        training_result = trainer.train()
        
        # Save the model
        save_result = trainer.save_model()
        
        return jsonify({
            'success': True,
            'message': 'Intent added and model retrained successfully',
            'intent': add_result['intent'],
            'training': training_result
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@training_bp.route('/export-intents', methods=['GET'])
def export_intents():
    """Export current intents as JSON"""
    try:
        return jsonify({
            'success': True,
            'intents': trainer.intents
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500