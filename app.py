from flask import Flask, render_template, request, jsonify, session
import nltk
import pickle
import numpy as np
import json
import random
import os
import logging
import sys
from pathlib import Path

# Configure logging with better formatting
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Global variables for model and data
model = None
words = []
classes = []
intents = {"intents": []}

def initialize_nltk():
    """Download and initialize NLTK data with better error handling."""
    try:
        import ssl
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        nltk.download("punkt", quiet=True)
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)  # Additional wordnet data
        logger.info("NLTK data downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to download NLTK data: {str(e)}")
        return False

def load_model_and_data():
    """Load all required model files with comprehensive error handling."""
    global model, words, classes, intents
    
    try:
        # Check if all required files exist
        required_files = ["chatbot_model.h5", "words.pkl", "classes.pkl", "intents.json"]
        missing_files = []
        
        for file in required_files:
            if not Path(file).exists():
                missing_files.append(file)
        
        if missing_files:
            logger.error(f"Missing required files: {missing_files}")
            return False
        
        # Load TensorFlow model
        try:
            from tensorflow.keras.models import load_model
            model = load_model("chatbot_model.h5")
            logger.info("TensorFlow model loaded successfully")
        except ImportError:
            logger.error("TensorFlow not installed. Please install: pip install tensorflow")
            return False
        except Exception as e:
            logger.error(f"Failed to load TensorFlow model: {str(e)}")
            return False
        
        # Load pickle files
        try:
            with open("words.pkl", "rb") as f:
                words = pickle.load(f)
            logger.info(f"Loaded {len(words)} words")
            
            with open("classes.pkl", "rb") as f:
                classes = pickle.load(f)
            logger.info(f"Loaded {len(classes)} classes")
        except Exception as e:
            logger.error(f"Failed to load pickle files: {str(e)}")
            return False
        
        # Load intents with proper encoding
        try:
            with open("intents.json", "r", encoding="utf-8") as f:
                intents = json.load(f)
            logger.info(f"Loaded {len(intents.get('intents', []))} intents")
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error in intents.json: {str(e)}")
            # Try with different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open("intents.json", "r", encoding=encoding) as f:
                        intents = json.load(f)
                    logger.info(f"Successfully loaded intents with {encoding} encoding")
                    break
                except:
                    continue
            else:
                logger.error("Could not load intents.json with any encoding")
                return False
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in intents.json: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Failed to load intents.json: {str(e)}")
            return False
        
        logger.info("All model files loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Unexpected error loading model and data: {str(e)}")
        return False

def initialize_lemmatizer():
    """Initialize lemmatizer with error handling."""
    try:
        from nltk.stem import WordNetLemmatizer
        return WordNetLemmatizer()
    except Exception as e:
        logger.error(f"Failed to initialize lemmatizer: {str(e)}")
        return None

# Initialize components
logger.info("Starting chatbot initialization...")

# Initialize NLTK
nltk_success = initialize_nltk()

# Initialize lemmatizer
lemmatizer = initialize_lemmatizer()

# Load model and data
model_success = load_model_and_data()

if not (nltk_success and model_success and lemmatizer):
    logger.warning("Some components failed to initialize. Chatbot may not work properly.")

# Create Flask app
app = Flask(__name__, static_folder="static")
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))

def preprocess_input(sentence):
    """Convert user input into a bag-of-words representation with error handling."""
    try:
        if not sentence or not sentence.strip():
            return np.array([[0] * len(words)])
        
        if not lemmatizer:
            logger.error("Lemmatizer not initialized")
            return np.array([[0] * len(words)])
        
        # Import here to avoid issues if NLTK not properly installed
        from nltk.tokenize import word_tokenize
        
        # Tokenize and lemmatize
        tokens = word_tokenize(sentence.lower())
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
        
        # Create bag of words
        bag = [0] * len(words)
        for token in tokens:
            if token in words:
                bag[words.index(token)] = 1
        
        return np.array([bag])
        
    except Exception as e:
        logger.error(f"Error in preprocessing input '{sentence}': {str(e)}")
        return np.array([[0] * len(words)]) if words else np.array([[]])

def get_response(user_input):
    """Predict intent and return corresponding response with improved error handling."""
    try:
        # Handle empty or invalid input
        if not user_input or not user_input.strip():
            return "Please say something so I can help you!"
        
        # Check if components are loaded
        if not model:
            return "I'm sorry, the chatbot is not properly initialized. Please check the server logs."
        
        if not words or not classes:
            return "I'm sorry, the chatbot data is not properly loaded. Please check the server logs."
        
        # Process input and get prediction
        input_data = preprocess_input(user_input)
        
        if input_data.size == 0:
            return "I'm sorry, I couldn't process your input. Please try again."
        
        # Get prediction
        prediction = model.predict(input_data, verbose=0)[0]
        
        # Get confidence score
        confidence = float(np.max(prediction))
        predicted_class_index = np.argmax(prediction)
        
        # Log prediction details for debugging
        logger.info(f"Input: '{user_input}', Confidence: {confidence:.3f}, Predicted class: {predicted_class_index}")
        
        # Check confidence threshold
        confidence_threshold = 0.5  # Lowered threshold for better responsiveness
        if confidence < confidence_threshold:
            return "I'm not quite sure what you're asking about. Could you please rephrase your question or ask about admissions, courses, fees, or faculty?"
        
        # Get predicted tag
        if predicted_class_index >= len(classes):
            logger.error(f"Predicted class index {predicted_class_index} out of range")
            return "I'm sorry, I encountered an error processing your request."
        
        tag = classes[predicted_class_index]
        
        # Find matching intent and return response
        for intent in intents.get("intents", []):
            if intent.get("tag") == tag:
                responses = intent.get("responses", [])
                if responses:
                    response = random.choice(responses)
                    
                    # Handle context if present
                    if "context_set" in intent:
                        session["context"] = intent["context_set"]
                    
                    logger.info(f"Response selected: '{response}' for tag: '{tag}'")
                    return response
        
        # Fallback response
        return "I understand you're asking something, but I'm not sure how to help with that specific query. Try asking about our courses, admissions, fees, or faculty!"
        
    except Exception as e:
        logger.error(f"Error generating response for '{user_input}': {str(e)}")
        return "I'm sorry, I encountered an unexpected error. Please try again or contact support if the problem persists."

@app.route("/")
def home():
    """Render chatbot HTML page."""
    try:
        return render_template("index.html")
    except Exception as e:
        logger.error(f"Error rendering home page: {str(e)}")
        return f"Error loading chatbot interface: {str(e)}", 500

@app.route("/get_response", methods=["POST"])
def chatbot_response():
    """Handle user input and return chatbot response with comprehensive error handling."""
    try:
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "Missing 'message' field in request"}), 400
        
        user_message = data["message"]
        
        # Validate message
        if not isinstance(user_message, str):
            return jsonify({"error": "Message must be a string"}), 400
        
        if len(user_message.strip()) == 0:
            return jsonify({"response": "Please type something so I can help you!"})
        
        if len(user_message) > 500:  # Prevent very long inputs
            return jsonify({"response": "Your message is too long. Please keep it under 500 characters."})
        
        # Get context
        current_context = session.get("context", "")
        
        # Log interaction
        logger.info(f"User: '{user_message}' | Context: '{current_context}'")
        
        # Get response
        response = get_response(user_message)
        
        # Return successful response
        return jsonify({
            "response": response,
            "timestamp": random.random(),  # Prevent caching
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Error in chatbot_response: {str(e)}")
        return jsonify({
            "response": "I'm sorry, I encountered an error. Please try again.",
            "status": "error"
        }), 500

@app.route("/reset", methods=["POST"])
def reset_conversation():
    """Reset conversation context."""
    try:
        session.clear()
        logger.info("Conversation context reset")
        return jsonify({"status": "success", "message": "Conversation reset"})
    except Exception as e:
        logger.error(f"Error resetting conversation: {str(e)}")
        return jsonify({"status": "error", "message": "Failed to reset"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Comprehensive health check endpoint."""
    try:
        health_status = {
            "status": "ok",
            "model_loaded": model is not None,
            "words_loaded": len(words) > 0,
            "classes_loaded": len(classes) > 0,
            "intents_loaded": len(intents.get("intents", [])) > 0,
            "lemmatizer_ready": lemmatizer is not None,
            "components": {
                "tensorflow": model is not None,
                "nltk": lemmatizer is not None,
                "data_files": len(words) > 0 and len(classes) > 0,
                "intents": len(intents.get("intents", [])) > 0
            }
        }
        
        # Determine overall health
        all_ready = all([
            health_status["model_loaded"],
            health_status["words_loaded"],
            health_status["classes_loaded"],
            health_status["intents_loaded"],
            health_status["lemmatizer_ready"]
        ])
        
        health_status["ready"] = all_ready
        status_code = 200 if all_ready else 503
        
        return jsonify(health_status), status_code
        
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

# Run Flask app
if __name__ == "__main__":
    # Configuration from environment variables
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"Starting Flask app on {host}:{port} (debug={debug_mode})")
    
    # Final health check before starting
    if model and words and classes and intents.get("intents"):
        logger.info("✅ Chatbot is ready to serve requests!")
    else:
        logger.warning("⚠️  Chatbot may not work properly - some components failed to load")
    
    try:
        app.run(host=host, port=port, debug=debug_mode)
    except Exception as e:
        logger.error(f"Failed to start Flask app: {str(e)}")
        sys.exit(1)