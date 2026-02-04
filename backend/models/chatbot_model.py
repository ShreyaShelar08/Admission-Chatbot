import json
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import os
from difflib import SequenceMatcher
from collections import Counter
import random

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('omw-1.4', quiet=True)


class ChatbotModel:
    """
    A robust chatbot with high accuracy and intelligent suggestions
    """
    
    def __init__(self, confidence_threshold=0.4, suggestion_count=3):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.intents = []
        self.pipeline = None
        self.tag_to_intent = {}
        self.all_patterns = {}
        self.response_keywords = {}
        self.pattern_to_tag = {}  # Map patterns to tags for suggestions
        self.model_path = 'saved_models/robust_chatbot_model.pkl'
        self.intents_path = 'saved_models/intents.json'
        self.confidence_threshold = confidence_threshold
        self.suggestion_count = suggestion_count
        
        os.makedirs('saved_models', exist_ok=True)
    
    def extract_keywords(self, text, top_n=15):
        """Extract important keywords from text"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = nltk.word_tokenize(text)
        keywords = [
            self.lemmatizer.lemmatize(word) 
            for word in tokens 
            if word not in self.stop_words and len(word) > 2
        ]
        word_freq = Counter(keywords)
        return [word for word, _ in word_freq.most_common(top_n)]
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s\?\.]', '', text)
        tokens = nltk.word_tokenize(text)
        tokens = [
            self.lemmatizer.lemmatize(word) 
            for word in tokens 
            if word not in self.stop_words and len(word) > 1
        ]
        return ' '.join(tokens)
    
    def load_training_data(self, file_path):
        """Load training data from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.intents = json.load(f)
            return True
        except Exception as e:
            raise Exception(f"Error loading training data: {str(e)}")
    
    def prepare_training_data(self):
        """Prepare training data"""
        X = []
        y = []
        
        for intent in self.intents:
            tag = intent['tag']
            self.tag_to_intent[tag] = intent
            self.all_patterns[tag] = intent['patterns']
            
            # Extract keywords from responses
            all_responses_text = ' '.join(intent['responses'])
            self.response_keywords[tag] = self.extract_keywords(all_responses_text, top_n=20)
            
            for pattern in intent['patterns']:
                processed = self.preprocess_text(pattern)
                X.append(processed)
                y.append(tag)
                # Store pattern mapping for suggestions
                self.pattern_to_tag[pattern.lower()] = tag
        
        return X, y
    
    def train(self, model_type='ensemble'):
        """
        Train the chatbot with ensemble or single model
        """
        try:
            if not self.intents:
                raise Exception("No training data loaded")
            
            X, y = self.prepare_training_data()
            
            # Use ensemble for better accuracy
            if model_type == 'ensemble':
                # Combine Random Forest and Logistic Regression
                from sklearn.ensemble import VotingClassifier
                
                rf_clf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=20)
                lr_clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
                
                self.pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(
                        max_features=2000,
                        ngram_range=(1, 3),
                        min_df=1,
                        sublinear_tf=True,
                        use_idf=True
                    )),
                    ('clf', VotingClassifier(
                        estimators=[('rf', rf_clf), ('lr', lr_clf)],
                        voting='soft',
                        weights=[1.2, 1.0]
                    ))
                ])
            else:
                # Single model (faster)
                classifier = RandomForestClassifier(n_estimators=200, random_state=42)
                self.pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(
                        max_features=2000,
                        ngram_range=(1, 3),
                        min_df=1,
                        sublinear_tf=True
                    )),
                    ('clf', classifier)
                ])
            
            self.pipeline.fit(X, y)
            
            return {
                'success': True,
                'message': f'Robust model trained with {model_type}',
                'intent_count': len(self.intents),
                'pattern_count': len(X)
            }
        except Exception as e:
            raise Exception(f"Error training model: {str(e)}")
    
    def keyword_match(self, user_input):
        """Match based on keywords from responses"""
        user_keywords = set(self.extract_keywords(user_input, top_n=10))
        
        if not user_keywords:
            return None
        
        scores = []
        for tag, response_kws in self.response_keywords.items():
            response_kw_set = set(response_kws)
            common_keywords = user_keywords.intersection(response_kw_set)
            
            if common_keywords:
                score = len(common_keywords) / len(user_keywords.union(response_kw_set))
                if len(common_keywords) > 1:
                    score *= (1 + 0.3 * len(common_keywords))
                scores.append({
                    'tag': tag,
                    'score': min(score, 1.0),
                    'matched_keywords': list(common_keywords)
                })
        
        if scores:
            scores.sort(key=lambda x: x['score'], reverse=True)
            return scores[0] if scores[0]['score'] >= 0.25 else None
        
        return None
    
    def fuzzy_match(self, user_input):
        """Fuzzy string matching with multiple candidates"""
        processed_input = self.preprocess_text(user_input)
        matches = []
        
        for tag, patterns in self.all_patterns.items():
            for pattern in patterns:
                processed_pattern = self.preprocess_text(pattern)
                ratio = SequenceMatcher(None, processed_input, processed_pattern).ratio()
                
                if ratio >= 0.5:  # Lower threshold to get more candidates
                    matches.append({
                        'tag': tag,
                        'score': ratio,
                        'pattern': pattern
                    })
        
        if matches:
            matches.sort(key=lambda x: x['score'], reverse=True)
            return matches[0] if matches[0]['score'] >= 0.6 else None
        
        return None
    
    def cosine_similarity_match(self, user_input):
        """Cosine similarity matching"""
        if self.pipeline is None:
            return None
        
        processed_input = self.preprocess_text(user_input)
        tfidf_vectorizer = self.pipeline.named_steps['tfidf']
        input_vector = tfidf_vectorizer.transform([processed_input])
        
        similarities = []
        for tag, patterns in self.all_patterns.items():
            processed_patterns = [self.preprocess_text(p) for p in patterns]
            pattern_vectors = tfidf_vectorizer.transform(processed_patterns)
            sims = cosine_similarity(input_vector, pattern_vectors)
            max_similarity = np.max(sims)
            
            if max_similarity >= 0.25:
                similarities.append({
                    'tag': tag,
                    'score': float(max_similarity)
                })
        
        if similarities:
            similarities.sort(key=lambda x: x['score'], reverse=True)
            return similarities[0] if similarities[0]['score'] >= 0.4 else None
        
        return None
    
    def get_similar_questions(self, user_input, count=3):
        """
        Get similar question suggestions based on user input
        """
        processed_input = self.preprocess_text(user_input)
        user_keywords = set(self.extract_keywords(user_input, top_n=10))
        
        suggestions = []
        
        # Get suggestions from all patterns
        for pattern, tag in self.pattern_to_tag.items():
            processed_pattern = self.preprocess_text(pattern)
            
            # Calculate multiple similarity scores
            fuzzy_score = SequenceMatcher(None, processed_input, processed_pattern).ratio()
            
            # Keyword overlap score
            pattern_keywords = set(self.extract_keywords(pattern, top_n=10))
            if user_keywords and pattern_keywords:
                keyword_score = len(user_keywords.intersection(pattern_keywords)) / len(user_keywords.union(pattern_keywords))
            else:
                keyword_score = 0
            
            # Combined score
            combined_score = (fuzzy_score * 0.6) + (keyword_score * 0.4)
            
            if combined_score > 0.2:  # Minimum threshold
                suggestions.append({
                    'question': pattern,
                    'tag': tag,
                    'score': combined_score
                })
        
        # Sort by score and return top N
        suggestions.sort(key=lambda x: x['score'], reverse=True)
        
        # Remove duplicates and return diverse suggestions
        seen_tags = set()
        unique_suggestions = []
        
        for sug in suggestions:
            if len(unique_suggestions) >= count:
                break
            # Prefer diverse intents
            if sug['tag'] not in seen_tags or len(unique_suggestions) < count // 2:
                unique_suggestions.append(sug['question'])
                seen_tags.add(sug['tag'])
        
        # If not enough, add some random popular questions
        if len(unique_suggestions) < count:
            popular_questions = [
                "What courses do you offer?",
                "How do I apply?",
                "What are the admission requirements?",
                "What are the tuition fees?",
                "Do you offer scholarships?"
            ]
            for q in popular_questions:
                if q not in unique_suggestions and len(unique_suggestions) < count:
                    unique_suggestions.append(q)
        
        return unique_suggestions[:count]
    
    def predict(self, user_input, return_details=False):
        """
        Robust prediction with multiple strategies and smart fallback
        """
        try:
            if self.pipeline is None:
                raise Exception("Model not loaded")
            
            if not user_input or len(user_input.strip()) < 2:
                return {
                    'intent': 'unknown',
                    'response': "Could you please provide more details?",
                    'confidence': 0.0,
                    'method': 'invalid_input'
                }
            
            processed_input = self.preprocess_text(user_input)
            
            # Collect all matching results
            all_results = []
            
            # Strategy 1: ML Model
            try:
                predicted_tag = self.pipeline.predict([processed_input])[0]
                probabilities = self.pipeline.predict_proba([processed_input])[0]
                ml_confidence = float(np.max(probabilities))
                
                all_results.append({
                    'tag': predicted_tag,
                    'confidence': ml_confidence,
                    'method': 'ml_model'
                })
            except:
                pass
            
            # Strategy 2: Keyword Matching
            keyword_result = self.keyword_match(user_input)
            if keyword_result:
                all_results.append({
                    'tag': keyword_result['tag'],
                    'confidence': keyword_result['score'],
                    'method': 'keyword_matching',
                    'matched_keywords': keyword_result['matched_keywords']
                })
            
            # Strategy 3: Cosine Similarity
            cosine_result = self.cosine_similarity_match(user_input)
            if cosine_result:
                all_results.append({
                    'tag': cosine_result['tag'],
                    'confidence': cosine_result['score'],
                    'method': 'cosine_similarity'
                })
            
            # Strategy 4: Fuzzy Matching
            fuzzy_result = self.fuzzy_match(user_input)
            if fuzzy_result:
                all_results.append({
                    'tag': fuzzy_result['tag'],
                    'confidence': fuzzy_result['score'],
                    'method': 'fuzzy_matching',
                    'matched_pattern': fuzzy_result['pattern']
                })
            
            # Choose best result
            if all_results:
                # Sort by confidence
                all_results.sort(key=lambda x: x['confidence'], reverse=True)
                best_result = all_results[0]
                
                # Check if confidence is acceptable
                if best_result['confidence'] >= self.confidence_threshold:
                    intent = self.tag_to_intent[best_result['tag']]
                    response = random.choice(intent['responses'])
                    
                    result = {
                        'intent': best_result['tag'],
                        'response': response,
                        'confidence': best_result['confidence'],
                        'method': best_result['method'],
                        'has_suggestions': False
                    }
                    
                    if return_details:
                        result['all_results'] = all_results
                        if 'matched_keywords' in best_result:
                            result['matched_keywords'] = best_result['matched_keywords']
                        if 'matched_pattern' in best_result:
                            result['matched_pattern'] = best_result['matched_pattern']
                    
                    return result
            
            # If no good match found, provide suggestions
            suggestions = self.get_similar_questions(user_input, count=self.suggestion_count)
            
            fallback_responses = [
                "I'm not quite sure I understood that. Here are some questions I can help you with:",
                "I didn't catch that. Perhaps you're looking for one of these topics:",
                "Hmm, I'm not certain about that. Here are some related questions:",
                "I want to make sure I give you the right information. Did you mean to ask about:"
            ]
            
            return {
                'intent': 'unknown',
                'response': random.choice(fallback_responses),
                'confidence': max([r['confidence'] for r in all_results]) if all_results else 0.0,
                'method': 'fallback_with_suggestions',
                'suggestions': suggestions,
                'has_suggestions': True
            }
            
        except Exception as e:
            return {
                'intent': 'error',
                'response': "I encountered an error. Please try rephrasing your question.",
                'confidence': 0.0,
                'method': 'error',
                'error': str(e)
            }
    
    def saved_models(self):
        """Save the trained model"""
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'pipeline': self.pipeline,
                    'tag_to_intent': self.tag_to_intent,
                    'all_patterns': self.all_patterns,
                    'response_keywords': self.response_keywords,
                    'pattern_to_tag': self.pattern_to_tag,
                    'confidence_threshold': self.confidence_threshold,
                    'suggestion_count': self.suggestion_count
                }, f)
            
            with open(self.intents_path, 'w', encoding='utf-8') as f:
                json.dump(self.intents, f, indent=2)
            
            return {'success': True, 'message': 'Model saved successfully'}
        except Exception as e:
            raise Exception(f"Error saving model: {str(e)}")
    
    def load_model(self):
        """Load a previously trained model"""
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.pipeline = data['pipeline']
                self.tag_to_intent = data['tag_to_intent']
                self.all_patterns = data['all_patterns']
                self.response_keywords = data.get('response_keywords', {})
                self.pattern_to_tag = data.get('pattern_to_tag', {})
                self.confidence_threshold = data.get('confidence_threshold', 0.4)
                self.suggestion_count = data.get('suggestion_count', 3)
            
            with open(self.intents_path, 'r', encoding='utf-8') as f:
                self.intents = json.load(f)
            
            return {'success': True, 'message': 'Model loaded successfully'}
        except FileNotFoundError:
            raise Exception("No trained model found")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def get_stats(self):
        """Get model statistics"""
        total_patterns = sum(len(intent['patterns']) for intent in self.intents)
        total_responses = sum(len(intent['responses']) for intent in self.intents)
        
        return {
            'total_intents': len(self.intents),
            'total_patterns': total_patterns,
            'total_responses': total_responses,
            'confidence_threshold': self.confidence_threshold,
            'suggestion_count': self.suggestion_count,
            'intents': [
                {
                    'tag': intent['tag'],
                    'pattern_count': len(intent['patterns']),
                    'response_count': len(intent['responses'])
                }
                for intent in self.intents
            ]
        }
    
    def chat(self):
        """Interactive chat mode with suggestions"""
        print("="*70)
        print("ROBUST CHATBOT - Type 'quit' to exit")
        print("="*70)
        print()
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nChatbot: Goodbye! Have a great day!")
                break
            
            if not user_input:
                continue
            
            result = self.predict(user_input, return_details=True)
            
            # Show response
            print(f"\nChatbot: {result['response']}")
            
            # Show suggestions if available
            if result.get('has_suggestions') and result.get('suggestions'):
                print("\n📌 Suggested questions:")
                for i, suggestion in enumerate(result['suggestions'], 1):
                    print(f"   {i}. {suggestion}")
                
                # Allow user to select a suggestion
                print("\n💡 Type a number to ask that question, or ask something else")
            
            # Show debug info
            if result.get('confidence'):
                confidence_bar = "█" * int(result['confidence'] * 20)
                print(f"\n[{result['method']} | Confidence: {result['confidence']:.0%} {confidence_bar}]")
            
            print()