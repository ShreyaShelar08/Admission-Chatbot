from flask import Blueprint, request, jsonify
from models.chatbot_model import ChatbotModel
from datetime import datetime

chatbot_bp = Blueprint('chatbot', __name__)

# Initialize chatbot model globally at module level
chatbot_instance = None

def get_chatbot():
    """Get or create chatbot instance"""
    global chatbot_instance
    if chatbot_instance is None:
        chatbot_instance = ChatbotModel()
        # Try to load existing model on first access
        try:
            chatbot_instance.load_model()
            print("✓ Pre-trained model loaded successfully")
        except Exception as e:
            print(f"⚠ No pre-trained model found: {e}")
            print("Please train the model first using /api/training/train endpoint")
    return chatbot_instance


@chatbot_bp.route('/query', methods=['POST'])
def query():
    """Get chatbot response for user query"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400
        
        user_message = data['message']
        
        # Get chatbot instance
        chatbot = get_chatbot()
        
        # Get response from chatbot
        result = chatbot.predict(user_message)
        
        return jsonify({
            'success': True,
            'user_message': user_message,
            'intent': result['intent'],
            'response': result['response'],
            'confidence': result['confidence']
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'response': "I'm having trouble processing your request. Please try again."
        }), 500


@chatbot_bp.route('/message', methods=['POST'])
def handle_message():
    """Alternative endpoint with simpler response (for compatibility)"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400
        
        user_message = data['message']
        
        # Get chatbot instance
        chatbot = get_chatbot()
        
        # Get response from chatbot
        result = chatbot.predict(user_message)
        
        return jsonify({
            'response': result['response'],
            'confidence': result.get('confidence', 0.0),
            'intent': result.get('intent', 'unknown'),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'response': "I'm having trouble processing your request. Please ensure the model is trained."
        }), 500


@chatbot_bp.route('/intents', methods=['GET'])
def get_intents():
    """Get all available intents"""
    try:
        chatbot = get_chatbot()
        return jsonify({
            'success': True,
            'intents': chatbot.intents
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@chatbot_bp.route('/test', methods=['GET'])
def test():
    """Test endpoint to check if chatbot is working"""
    try:
        chatbot = get_chatbot()
        is_trained = chatbot.pipeline is not None
        
        return jsonify({
            'success': True,
            'message': 'Chatbot endpoint is working',
            'model_loaded': is_trained,
            'intents_count': len(chatbot.intents) if chatbot.intents else 0
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'message': 'Chatbot endpoint error',
            'error': str(e),
            'model_loaded': False
        }), 500


@chatbot_bp.route('/stats', methods=['GET'])
def get_stats():
    """Get chatbot statistics"""
    try:
        chatbot = get_chatbot()
        
        if not chatbot.intents:
            return jsonify({
                'success': False,
                'message': 'Model not trained yet'
            }), 400
        
        stats = chatbot.get_stats()
        return jsonify({
            'success': True,
            'stats': stats
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500