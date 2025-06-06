import os
import sqlite3
import json
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import requests
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Hugging Face API configuration
HF_API_KEY = os.getenv('HUGGINGFACE_API_KEY', '')
HF_API_BASE = "https://api-inference.huggingface.co/models/"

# Database configuration
DATABASE_PATH = 'ai_memory.db'

# Model configurations with speed categories and power ratings
MODELS_CONFIG = {
    # Lightning Fast (1-2 stars)
    "deepseek-ai/deepseek-coder-1.3b-instruct": {
        "name": "DeepSeek Coder 1.3B",
        "category": "lightning",
        "power": 1,
        "description": "Ultra-fast lightweight model for quick code suggestions",
        "specs": "1.3B params • Instruct-tuned • Multi-language"
    },
    "Salesforce/codegen2-1B-multi": {
        "name": "CodeGen2 1B Multi",
        "category": "lightning",
        "power": 1,
        "description": "Salesforce's efficient multi-language code generator",
        "specs": "1B params • Multi-language • Fast inference"
    },
    "monsterapi/llama2-7b-tiny-codes-code-generation": {
        "name": "Llama2 Tiny Codes",
        "category": "lightning",
        "power": 2,
        "description": "Optimized Llama2 variant for code generation tasks",
        "specs": "7B params • Code-focused • Fast generation"
    },
    
    # Balanced Performance (2 stars)
    "bigcode/starcoder2-3b": {
        "name": "StarCoder2 3B",
        "category": "balanced",
        "power": 2,
        "description": "BigCode's balanced model with excellent code understanding",
        "specs": "3B params • State-of-the-art • 80+ languages"
    },
    "replit/replit-code-v1-3b": {
        "name": "Replit Code v1 3B",
        "category": "balanced",
        "power": 2,
        "description": "Replit's code-specialized model with IDE integration focus",
        "specs": "3B params • IDE-optimized • Multi-language"
    },
    "THUDM/codegeex4-all-9b": {
        "name": "CodeGeeX4 All 9B",
        "category": "balanced",
        "power": 2,
        "description": "Comprehensive code model with strong reasoning capabilities",
        "specs": "9B params • All-purpose • Advanced reasoning"
    },
    "lightblue/kurage-multilingual": {
        "name": "Kurage Multilingual",
        "category": "balanced",
        "power": 2,
        "description": "Specialized multilingual coding assistant",
        "specs": "Multilingual • Code-focused • Balanced speed"
    },
    
    # Maximum Power (3 stars)
    "codellama/CodeLlama-13b-Instruct-hf": {
        "name": "Code Llama 13B Instruct",
        "category": "power",
        "power": 3,
        "description": "Meta's powerful instruction-tuned coding model",
        "specs": "13B params • Instruct-tuned • High accuracy"
    },
    "bigcode/starcoder2-15b": {
        "name": "StarCoder2 15B",
        "category": "power",
        "power": 3,
        "description": "Most powerful StarCoder variant with exceptional capabilities",
        "specs": "15B params • Top performance • 80+ languages"
    },
    "deepseek-ai/deepseek-coder-33b-instruct": {
        "name": "DeepSeek Coder 33B",
        "category": "power",
        "power": 3,
        "description": "Most powerful model with exceptional reasoning and code quality",
        "specs": "33B params • Maximum power • Enterprise-grade"
    }
}

def init_database():
    """Initialize SQLite database for AI memory"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Create chat_sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id TEXT PRIMARY KEY,
            model TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create chat_messages table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            model TEXT NOT NULL,
            user_message TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chat_id) REFERENCES chat_sessions (id)
        )
    ''')
    
    # Create indexes for better performance
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_chat_messages_chat_id ON chat_messages(chat_id)
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_chat_messages_timestamp ON chat_messages(timestamp)
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")

def get_chat_memory(chat_id, limit=10):
    """Retrieve recent chat history for context"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT user_message, bot_response, timestamp 
        FROM chat_messages 
        WHERE chat_id = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
    ''', (chat_id, limit))
    
    messages = cursor.fetchall()
    conn.close()
    
    # Reverse to get chronological order
    return list(reversed(messages))

def save_chat_message(chat_id, model, user_message, bot_response):
    """Save chat message to database"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Create or update chat session
    cursor.execute('''
        INSERT OR REPLACE INTO chat_sessions (id, model, last_active)
        VALUES (?, ?, CURRENT_TIMESTAMP)
    ''', (chat_id, model))
    
    # Save message
    cursor.execute('''
        INSERT INTO chat_messages (chat_id, model, user_message, bot_response)
        VALUES (?, ?, ?, ?)
    ''', (chat_id, model, user_message, bot_response))
    
    conn.commit()
    conn.close()

def clear_chat_memory(chat_id):
    """Clear memory for a specific chat"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM chat_messages WHERE chat_id = ?', (chat_id,))
    cursor.execute('DELETE FROM chat_sessions WHERE id = ?', (chat_id,))
    
    conn.commit()
    conn.close()

def call_huggingface_api(model_name, prompt, chat_history=None):
    """Call Hugging Face API with retry logic"""
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Build context from chat history
    context = ""
    if chat_history:
        for user_msg, bot_msg, _ in chat_history[-3:]:  # Last 3 exchanges for context
            context += f"User: {user_msg}\nAssistant: {bot_msg}\n\n"
    
    # Format prompt based on model type
    if "instruct" in model_name.lower() or "chat" in model_name.lower():
        full_prompt = f"{context}User: {prompt}\nAssistant:"
    else:
        full_prompt = f"{context}{prompt}"
    
    payload = {
        "inputs": full_prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "return_full_text": False
        }
    }
    
    url = f"{HF_API_BASE}{model_name}"
    
    # Retry logic for API calls
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get('generated_text', '').strip()
                    if generated_text:
                        return generated_text
                    else:
                        return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
                else:
                    return "I received an unexpected response format. Please try again."
                    
            elif response.status_code == 503:
                # Model loading
                if attempt < max_retries - 1:
                    wait_time = 5 * (attempt + 1)
                    logger.info(f"Model loading, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    return "The AI model is currently loading. Please try again in a few moments."
                    
            elif response.status_code == 429:
                # Rate limit
                if attempt < max_retries - 1:
                    wait_time = 10 * (attempt + 1)
                    logger.info(f"Rate limited, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    return "I'm currently experiencing high traffic. Please try again in a moment."
                    
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return f"I encountered an error (code {response.status_code}). Please try again."
                
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                logger.info("Request timeout, retrying...")
                time.sleep(2)
                continue
            else:
                return "The request timed out. Please try again."
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            return "I'm having trouble connecting to the AI service. Please try again."
    
    return "I'm unable to process your request right now. Please try again later."

# Routes
@app.route('/')
def index():
    """Serve the main HTML page"""
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError:
        return """
        <html>
        <head><title>AI Platform Error</title></head>
        <body>
            <h1>AI Platform</h1>
            <p>Error: index.html file not found. Please create the frontend file.</p>
            <p>API is running at: <a href="/api/health">/api/health</a></p>
        </body>
        </html>
        """

@app.route('/api/models')
def get_models():
    """Get available models configuration"""
    return jsonify(MODELS_CONFIG)

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.json
        chat_id = data.get('chat_id')
        model = data.get('model')
        message = data.get('message')
        
        if not all([chat_id, model, message]):
            return jsonify({'error': 'Missing required fields'}), 400
            
        if model not in MODELS_CONFIG:
            return jsonify({'error': 'Invalid model'}), 400
        
        # Get chat history for context
        chat_history = get_chat_memory(chat_id)
        
        # Call AI model
        response = call_huggingface_api(model, message, chat_history)
        
        # Save to database
        save_chat_message(chat_id, model, message, response)
        
        return jsonify({
            'response': response,
            'model': model,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/chat/<chat_id>/clear', methods=['DELETE'])
def clear_chat(chat_id):
    """Clear chat memory"""
    try:
        clear_chat_memory(chat_id)
        return jsonify({'message': 'Chat memory cleared successfully'})
    except Exception as e:
        logger.error(f"Clear chat error: {str(e)}")
        return jsonify({'error': 'Failed to clear chat memory'}), 500

@app.route('/api/chat/<chat_id>/history')
def get_chat_history(chat_id):
    """Get chat history"""
    try:
        history = get_chat_memory(chat_id, limit=50)
        return jsonify({
            'chat_id': chat_id,
            'messages': [
                {
                    'user_message': msg[0],
                    'bot_response': msg[1],
                    'timestamp': msg[2]
                } for msg in history
            ]
        })
    except Exception as e:
        logger.error(f"Get history error: {str(e)}")
        return jsonify({'error': 'Failed to retrieve chat history'}), 500

@app.route('/api/test/<model_name>')
def test_model(model_name):
    """Test endpoint for AI models"""
    try:
        if model_name not in MODELS_CONFIG:
            return jsonify({'error': f'Model {model_name} not found'}), 404
        
        test_prompt = "Write a simple Python function that adds two numbers and returns the result."
        
        logger.info(f"Testing model: {model_name}")
        start_time = time.time()
        
        response = call_huggingface_api(model_name, test_prompt)
        
        end_time = time.time()
        response_time = round(end_time - start_time, 2)
        
        model_info = MODELS_CONFIG[model_name]
        
        return jsonify({
            'model': model_name,
            'model_info': model_info,
            'test_prompt': test_prompt,
            'response': response,
            'response_time_seconds': response_time,
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Test model error: {str(e)}")
        return jsonify({
            'model': model_name,
            'error': str(e),
            'status': 'failed',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/test/all')
def test_all_models():
    """Test all available models"""
    results = {}
    test_prompt = "Write a simple hello world function in Python."
    
    for model_name, model_info in MODELS_CONFIG.items():
        try:
            logger.info(f"Testing model: {model_name}")
            start_time = time.time()
            
            response = call_huggingface_api(model_name, test_prompt)
            
            end_time = time.time()
            response_time = round(end_time - start_time, 2)
            
            results[model_name] = {
                'model_info': model_info,
                'response': response,
                'response_time_seconds': response_time,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error testing {model_name}: {str(e)}")
            results[model_name] = {
                'model_info': model_info,
                'error': str(e),
                'status': 'failed'
            }
        
        # Small delay between tests to avoid rate limiting
        time.sleep(1)
    
    return jsonify({
        'test_prompt': test_prompt,
        'results': results,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_count': len(MODELS_CONFIG),
        'database_path': DATABASE_PATH
    })

@app.route('/api/stats')
def get_stats():
    """Get platform statistics"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Get total chats
        cursor.execute('SELECT COUNT(*) FROM chat_sessions')
        total_chats = cursor.fetchone()[0]
        
        # Get total messages
        cursor.execute('SELECT COUNT(*) FROM chat_messages')
        total_messages = cursor.fetchone()[0]
        
        # Get most used models
        cursor.execute('''
            SELECT model, COUNT(*) as usage_count 
            FROM chat_messages 
            GROUP BY model 
            ORDER BY usage_count DESC 
            LIMIT 5
        ''')
        popular_models = cursor.fetchall()
        
        conn.close()
        
        return jsonify({
            'total_chats': total_chats,
            'total_messages': total_messages,
            'popular_models': [{'model': model, 'usage_count': count} for model, count in popular_models],
            'available_models': len(MODELS_CONFIG),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        return jsonify({'error': 'Failed to retrieve stats'}), 500

if __name__ == '__main__':
    # Initialize database on startup
    init_database()
    
    # Get port from environment variable (for Render.com deployment)
    port = int(os.environ.get('PORT', 5000))
    
    logger.info(f"Starting AI Code Assistant Platform on port {port}")
    logger.info(f"Available models: {len(MODELS_CONFIG)}")
    
    # Run the application
    app.run(
        host='0.0.0.0',
        port=port,
        debug=os.environ.get('FLASK_ENV') == 'development'
    )
