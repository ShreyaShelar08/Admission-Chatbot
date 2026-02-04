import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.chatbot_model import ChatbotModel


def train_chatbot():
    """Train the chatbot model with the training data"""
    print("=" * 50)
    print("Starting Chatbot Training")
    print("=" * 50)
    
    trainer = ChatbotModel()
    
    try:
        # Load training data
        print("\n1. Loading training data...")
        data_path = 'data/training_data.json'
        trainer.load_training_data(data_path)
        print(f"✓ Training data loaded successfully")
        print(f"  - Total intents: {len(trainer.intents)}")
        
        # Train the model
        print("\n2. Training the model...")
        result = trainer.train()
        print(f"✓ {result['message']}")
        print(f"  - Intent count: {result['intent_count']}")
        print(f"  - Pattern count: {result['pattern_count']}")
        
        # saved the model
        print("\n3. Saving the model...")
        saved_result = trainer.saved_models()   
        print(f"✓ {saved_result['message']}")
        print(f"  - Model path: {trainer.model_path}")
        
        # Get and display statistics
        print("\n4. Model Statistics:")
        stats = trainer.get_stats()
        print(f"  - Total Intents: {stats['total_intents']}")
        print(f"  - Total Patterns: {stats['total_patterns']}")
        print(f"  - Total Responses: {stats['total_responses']}")
        
        print("\n" + "=" * 50)
        print("✅ Training completed successfully!")
        print("=" * 50)
        print("\nYou can now start the server with: python app.py")
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    train_chatbot()