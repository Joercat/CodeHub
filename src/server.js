const express = require('express');
const path = require('path');
const sqlite3 = require('sqlite3').verbose();
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Database setup
const dbPath = path.join(__dirname, 'ai_chat.db');
let db;

function initializeDatabase() {
    return new Promise((resolve, reject) => {
        db = new sqlite3.Database(dbPath, (err) => {
            if (err) {
                console.error('Error opening database:', err);
                reject(err);
                return;
            }
            
            console.log('Connected to SQLite database');
            
            // Create tables
            const createTables = `
                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY,
                    bot_id TEXT NOT NULL,
                    bot_model TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chat_id) REFERENCES chats (id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_chat_messages ON messages(chat_id);
                CREATE INDEX IF NOT EXISTS idx_message_timestamp ON messages(timestamp);
            `;
            
            db.exec(createTables, (err) => {
                if (err) {
                    console.error('Error creating tables:', err);
                    reject(err);
                } else {
                    console.log('Database tables initialized');
                    resolve();
                }
            });
        });
    });
}

// Hugging Face API configuration
const HF_API_KEY = process.env.HF_API_KEY;
const HF_API_URL = 'https://api-inference.huggingface.co/models/';

// Bot configurations with their actual Hugging Face model paths
const botConfigs = {
    'deepseek-coder-1.3b': {
        model: 'deepseek-ai/deepseek-coder-1.3b-instruct',
        maxTokens: 512,
        temperature: 0.3
    },
    'codegen2-1b': {
        model: 'Salesforce/codegen2-1B-multi',
        maxTokens: 512,
        temperature: 0.2
    },
    'starcoder2-3b': {
        model: 'bigcode/starcoder2-3b',
        maxTokens: 1024,
        temperature: 0.3
    },
    'replit-code-v1': {
        model: 'replit/replit-code-v1-3b',
        maxTokens: 512,
        temperature: 0.3
    },
    'llama2-7b-codes': {
        model: 'monsterapi/llama2-7b-tiny-codes-code-generation',
        maxTokens: 1024,
        temperature: 0.4
    },
    'codegeex4-9b': {
        model: 'THUDM/codegeex4-all-9b',
        maxTokens: 1024,
        temperature: 0.3
    },
    'kurage-multilingual': {
        model: 'lightblue/kurage-multilingual',
        maxTokens: 1024,
        temperature: 0.3
    },
    'starcoder2-15b': {
        model: 'bigcode/starcoder2-15b',
        maxTokens: 1024,
        temperature: 0.3
    },
    'codellama-13b': {
        model: 'codellama/CodeLlama-13b-Instruct-hf',
        maxTokens: 2048,
        temperature: 0.3
    },
    'deepseek-coder-33b': {
        model: 'deepseek-ai/deepseek-coder-33b-instruct',
        maxTokens: 2048,
        temperature: 0.3
    }
};

// Hugging Face API interaction
async function queryHuggingFace(model, prompt, config) {
    if (!HF_API_KEY) {
        throw new Error('Hugging Face API key not configured');
    }

    const response = await fetch(`${HF_API_URL}${model}`, {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${HF_API_KEY}`,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            inputs: prompt,
            parameters: {
                max_new_tokens: config.maxTokens,
                temperature: config.temperature,
                return_full_text: false,
                do_sample: true
            }
        })
    });

    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HuggingFace API error: ${response.status} - ${errorText}`);
    }

    const result = await response.json();
    
    if (Array.isArray(result) && result.length > 0) {
        return result[0].generated_text || result[0].text || 'No response generated';
    } else if (result.generated_text) {
        return result.generated_text;
    } else {
        return 'Unable to generate response';
    }
}

// Routes
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// Initialize chat session
app.post('/api/init-chat', async (req, res) => {
    const { chatId, botId, botModel } = req.body;
    
    if (!chatId || !botId || !botModel) {
        return res.status(400).json({ error: 'Missing required fields' });
    }

    try {
        const stmt = db.prepare('INSERT OR REPLACE INTO chats (id, bot_id, bot_model) VALUES (?, ?, ?)');
        stmt.run([chatId, botId, botModel], function(err) {
            if (err) {
                console.error('Database error:', err);
                return res.status(500).json({ error: 'Database error' });
            }
            
            res.json({ success: true, chatId });
        });
        stmt.finalize();
    } catch (error) {
        console.error('Error initializing chat:', error);
        res.status(500).json({ error: 'Server error' });
    }
});

// Handle chat messages
app.post('/api/chat', async (req, res) => {
    const { chatId, message, botId, botModel } = req.body;
    
    if (!chatId || !message || !botId || !botModel) {
        return res.status(400).json({ error: 'Missing required fields' });
    }

    try {
        // Verify chat exists and bot hasn't changed
        const chat = await new Promise((resolve, reject) => {
            db.get('SELECT * FROM chats WHERE id = ?', [chatId], (err, row) => {
                if (err) reject(err);
                else resolve(row);
            });
        });

        if (!chat) {
            return res.status(404).json({ error: 'Chat not found' });
        }

        if (chat.bot_id !== botId) {
            return res.status(400).json({ error: 'Cannot switch bots mid-chat. Please start a new chat.' });
        }

        // Store user message
        const userStmt = db.prepare('INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)');
        userStmt.run([chatId, 'user', message]);
        userStmt.finalize();

        // Get chat history for context
        const history = await new Promise((resolve, reject) => {
            db.all(
                'SELECT role, content FROM messages WHERE chat_id = ? ORDER BY timestamp ASC LIMIT 10',
                [chatId],
                (err, rows) => {
                    if (err) reject(err);
                    else resolve(rows);
                }
            );
        });

        // Build context prompt
        let contextPrompt = "You are a helpful coding assistant. Help the user with their programming questions.\n\n";
        
        for (const msg of history.slice(-5)) { // Last 5 messages for context
            if (msg.role === 'user') {
                contextPrompt += `Human: ${msg.content}\n`;
            } else {
                contextPrompt += `Assistant: ${msg.content}\n`;
            }
        }
        
        contextPrompt += `\nHuman: ${message}\nAssistant:`;

        // Get bot configuration
        const config = botConfigs[botId];
        if (!config) {
            return res.status(400).json({ error: 'Invalid bot configuration' });
        }

        try {
            // Query the AI model
            const aiResponse = await queryHuggingFace(config.model, contextPrompt, config);
            
            // Clean up the response
            let cleanResponse = aiResponse.trim();
            
            // Remove any repeated context or prompts
            if (cleanResponse.includes('Human:') || cleanResponse.includes('Assistant:')) {
                const parts = cleanResponse.split(/(?:Human:|Assistant:)/);
                cleanResponse = parts[parts.length - 1].trim();
            }

            // Store AI response
            const botStmt = db.prepare('INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)');
            botStmt.run([chatId, 'assistant', cleanResponse]);
            botStmt.finalize();

            res.json({ 
                success: true, 
                response: cleanResponse,
                botId: botId 
            });

        } catch (aiError) {
            console.error('AI API error:', aiError);
            
            // Fallback response
            const fallbackResponse = "I'm experiencing some technical difficulties right now. Please try again in a moment, or rephrase your question.";
            
            const botStmt = db.prepare('INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)');
            botStmt.run([chatId, 'assistant', fallbackResponse]);
            botStmt.finalize();
            
            res.json({ 
                success: true, 
                response: fallbackResponse,
                botId: botId,
                warning: 'Using fallback response due to API issues'
            });
        }

    } catch (error) {
        console.error('Error handling chat:', error);
        res.status(500).json({ error: 'Server error' });
    }
});

// Get chat history
app.get('/api/chat/:chatId/history', (req, res) => {
    const { chatId } = req.params;
    
    db.all(
        'SELECT role, content, timestamp FROM messages WHERE chat_id = ? ORDER BY timestamp ASC',
        [chatId],
        (err, rows) => {
            if (err) {
                console.error('Database error:', err);
                return res.status(500).json({ error: 'Database error' });
            }
