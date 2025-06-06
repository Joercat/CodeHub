// Load environment variables
require('dotenv').config();

const express = require('express');
const path = require('path');
const sqlite3 = require('sqlite3').verbose();
const fs = require('fs');
const fetch = require('node-fetch');

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

// Updated bot configurations with verified working Hugging Face models
const botConfigs = {
    'deepseek-coder-1.3b': {
        model: 'deepseek-ai/deepseek-coder-1.3b-instruct',
        maxTokens: 512,
        temperature: 0.3
    },
    'codegen-350m': {
        model: 'Salesforce/codegen-350M-mono',
        maxTokens: 512,
        temperature: 0.3
    },
    'starcoder2-3b': {
        model: 'bigcode/starcoder2-3b',
        maxTokens: 1024,
        temperature: 0.3
    },
    'tiny-starcoder': {
        model: 'bigcode/tiny_starcoder_py',
        maxTokens: 512,
        temperature: 0.3
    },
    'phi-1.5': {
        model: 'microsoft/phi-1_5',
        maxTokens: 512,
        temperature: 0.4
    },
    'santacoder': {
        model: 'bigcode/santacoder',
        maxTokens: 512,
        temperature: 0.3
    },
    'gpt2-medium': {
        model: 'gpt2-medium',
        maxTokens: 512,
        temperature: 0.7
    },
    'distilgpt2': {
        model: 'distilgpt2',
        maxTokens: 256,
        temperature: 0.7
    },
    'flan-t5-base': {
        model: 'google/flan-t5-base',
        maxTokens: 512,
        temperature: 0.4
    },
    'code-t5-small': {
        model: 'Salesforce/codet5-small',
        maxTokens: 512,
        temperature: 0.3
    }
};

// Enhanced Hugging Face API interaction with better error handling
async function queryHuggingFace(model, prompt, config) {
    if (!HF_API_KEY) {
        throw new Error('Hugging Face API key not configured');
    }

    const maxRetries = 3;
    let lastError = null;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            console.log(`Attempting to query model: ${model} (attempt ${attempt})`);
            
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
                        do_sample: true,
                        top_p: 0.95,
                        top_k: 50
                    },
                    options: {
                        wait_for_model: true,
                        use_cache: false
                    }
                })
            });

            if (!response.ok) {
                const errorText = await response.text();
                console.error(`API Error for ${model}: ${response.status} - ${errorText}`);
                
                // Handle specific error cases
                if (response.status === 404) {
                    throw new Error(`Model '${model}' not found on Hugging Face`);
                } else if (response.status === 503) {
                    // Model is loading, wait and retry
                    if (attempt < maxRetries) {
                        console.log(`Model ${model} is loading, waiting 5 seconds...`);
                        await new Promise(resolve => setTimeout(resolve, 5000));
                        continue;
                    }
                    throw new Error(`Model '${model}' is currently loading. Please try again later.`);
                } else if (response.status === 429) {
                    // Rate limited, wait and retry
                    if (attempt < maxRetries) {
                        console.log(`Rate limited for ${model}, waiting 3 seconds...`);
                        await new Promise(resolve => setTimeout(resolve, 3000));
                        continue;
                    }
                    throw new Error(`Rate limit exceeded for model '${model}'`);
                }
                
                throw new Error(`HuggingFace API error: ${response.status} - ${errorText}`);
            }

            const result = await response.json();
            console.log(`Successful response from ${model}:`, typeof result, Array.isArray(result));
            
            // Handle different response formats
            if (Array.isArray(result) && result.length > 0) {
                const firstResult = result[0];
                return firstResult.generated_text || firstResult.text || 'No response generated';
            } else if (result.generated_text !== undefined) {
                return result.generated_text;
            } else if (result[0] && result[0].generated_text !== undefined) {
                return result[0].generated_text;
            } else if (typeof result === 'string') {
                return result;
            } else {
                console.warn('Unexpected response format:', result);
                return 'Unable to parse response from AI model';
            }

        } catch (error) {
            lastError = error;
            console.error(`Attempt ${attempt} failed for ${model}:`, error.message);
            
            // Don't retry for certain errors
            if (error.message.includes('not found') || error.message.includes('404')) {
                throw error;
            }
            
            // Wait before retrying (except on last attempt)
            if (attempt < maxRetries) {
                await new Promise(resolve => setTimeout(resolve, 2000));
            }
        }
    }

    throw lastError || new Error(`Failed to query ${model} after ${maxRetries} attempts`);
}

// Test individual model function with better error handling
async function testModel(botId, config) {
    const testPrompt = "def add_numbers(a, b):\n    # Write a function to add two numbers\n    return";
    const startTime = Date.now();
    
    try {
        const response = await queryHuggingFace(config.model, testPrompt, config);
        const responseTime = Date.now() - startTime;
        
        return {
            botId,
            model: config.model,
            status: 'working',
            responseTime: `${responseTime}ms`,
            response: response.substring(0, 200) + (response.length > 200 ? '...' : ''),
            error: null
        };
    } catch (error) {
        const responseTime = Date.now() - startTime;
        
        return {
            botId,
            model: config.model,
            status: 'error',
            responseTime: `${responseTime}ms`,
            response: null,
            error: error.message
        };
    }
}

// Routes
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// Test endpoint to check which AIs are working
app.get('/test', async (req, res) => {
    const testSingleBot = req.query.bot;
    
    // Check if API key is configured
    if (!HF_API_KEY) {
        return res.status(500).json({
            error: 'Hugging Face API key not configured',
            message: 'Please set HF_API_KEY in your environment variables'
        });
    }

    try {
        let results = [];
        
        if (testSingleBot) {
            // Test single bot
            if (!botConfigs[testSingleBot]) {
                return res.status(400).json({
                    error: `Bot '${testSingleBot}' not found`,
                    availableBots: Object.keys(botConfigs)
                });
            }
            
            console.log(`Testing single bot: ${testSingleBot}`);
            results.push(await testModel(testSingleBot, botConfigs[testSingleBot]));
        } else {
            // Test all bots with sequential testing to avoid rate limits
            console.log('Testing all AI models sequentially...');
            
            for (const [botId, config] of Object.entries(botConfigs)) {
                console.log(`Testing ${botId}...`);
                try {
                    const result = await testModel(botId, config);
                    results.push(result);
                    // Small delay between tests to avoid rate limits
                    await new Promise(resolve => setTimeout(resolve, 1000));
                } catch (error) {
                    results.push({
                        botId,
                        model: config.model,
                        status: 'error',
                        responseTime: '0ms',
                        response: null,
                        error: error.message
                    });
                }
            }
        }

        // Generate summary
        const working = results.filter(r => r.status === 'working');
        const errors = results.filter(r => r.status === 'error');
        
        const summary = {
            totalModels: results.length,
            working: working.length,
            errors: errors.length,
            workingModels: working.map(r => r.botId),
            errorModels: errors.map(r => r.botId)
        };

        // HTML response for browser viewing
        if (req.headers.accept && req.headers.accept.includes('text/html')) {
            let html = `
            <!DOCTYPE html>
            <html>
            <head>
                <title>AI Models Test Results</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .summary { background: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
                    .model { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }
                    .working { border-left: 5px solid #4CAF50; }
                    .error { border-left: 5px solid #f44336; }
                    .response { background: #f9f9f9; padding: 10px; margin-top: 10px; border-radius: 3px; font-family: monospace; white-space: pre-wrap; }
                    .error-text { color: #f44336; }
                    .success-text { color: #4CAF50; }
                    .refresh-btn { background: #2196F3; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin-bottom: 20px; }
                    .test-single { margin-bottom: 20px; }
                    .test-single input { padding: 8px; margin-right: 10px; width: 300px; }
                    .test-single button { padding: 8px 15px; background: #FF9800; color: white; border: none; border-radius: 3px; cursor: pointer; }
                    .model-list { background: #f9f9f9; padding: 10px; border-radius: 3px; margin: 10px 0; }
                </style>
            </head>
            <body>
                <h1>AI Models Test Results</h1>
                
                <div class="test-single">
                    <h3>Test Single Model:</h3>
                    <input type="text" id="botId" placeholder="Enter bot ID (e.g., deepseek-coder-1.3b)" list="botList">
                    <datalist id="botList">
                        ${Object.keys(botConfigs).map(id => `<option value="${id}">`).join('')}
                    </datalist>
                    <button onclick="testSingle()">Test Single Bot</button>
                    
                    <div class="model-list">
                        <strong>Available models:</strong> ${Object.keys(botConfigs).join(', ')}
                    </div>
                </div>
                
                <button class="refresh-btn" onclick="location.reload()">Refresh All Tests</button>
                
                <div class="summary">
                    <h2>Summary</h2>
                    <p><strong>Total Models:</strong> ${summary.totalModels}</p>
                    <p><strong class="success-text">Working:</strong> ${summary.working}</p>
                    <p><strong class="error-text">Errors:</strong> ${summary.errors}</p>
                    <p><strong>Working Models:</strong> ${summary.workingModels.join(', ') || 'None'}</p>
                    <p><strong>Error Models:</strong> ${summary.errorModels.join(', ') || 'None'}</p>
                </div>
                
                <h2>Detailed Results</h2>
            `;
            
            results.forEach(result => {
                const statusClass = result.status === 'working' ? 'working' : 'error';
                html += `
                <div class="model ${statusClass}">
                    <h3>${result.botId}</h3>
                    <p><strong>Model:</strong> ${result.model}</p>
                    <p><strong>Status:</strong> <span class="${result.status === 'working' ? 'success-text' : 'error-text'}">${result.status.toUpperCase()}</span></p>
                    <p><strong>Response Time:</strong> ${result.responseTime}</p>
                    ${result.error ? `<p><strong>Error:</strong> <span class="error-text">${result.error}</span></p>` : ''}
                    ${result.response ? `<div class="response"><strong>Sample Response:</strong><br>${result.response}</div>` : ''}
                </div>
                `;
            });
            
            html += `
                <script>
                    function testSingle() {
                        const botId = document.getElementById('botId').value;
                        if (botId) {
                            window.location.href = '/test?bot=' + encodeURIComponent(botId);
                        } else {
                            alert('Please enter a bot ID');
                        }
                    }
                </script>
            </body>
            </html>
            `;
            
            res.send(html);
        } else {
            // JSON response for API calls
            res.json({
                success: true,
                summary,
                results,
                timestamp: new Date().toISOString()
            });
        }

    } catch (error) {
        console.error('Test endpoint error:', error);
        res.status(500).json({
            error: 'Server error during testing',
            message: error.message
        });
    }
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

// Handle chat messages with improved prompt formatting
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

        // Build context prompt with better formatting for code models
        let contextPrompt = "";
        
        // Check if this is a coding-related bot
        const isCodeBot = botId.includes('coder') || botId.includes('code') || botId.includes('star');
        
        if (isCodeBot) {
            contextPrompt = "# Coding Assistant\n\nHelp with programming questions and provide clean, working code.\n\n";
        } else {
            contextPrompt = "You are a helpful AI assistant. Provide clear and helpful responses.\n\n";
        }
        
        // Add conversation history (last 3 exchanges)
        const recentHistory = history.slice(-6); // Last 6 messages = 3 exchanges
        for (const msg of recentHistory) {
            if (msg.role === 'user') {
                contextPrompt += `User: ${msg.content}\n`;
            } else {
                contextPrompt += `Assistant: ${msg.content}\n`;
            }
        }
        
        contextPrompt += `User: ${message}\nAssistant:`;

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
            const cleanupPatterns = [
                /^(User:|Assistant:|Human:|AI:)/gim,
                /^(You are a helpful|# Coding Assistant)/gim
            ];
            
            for (const pattern of cleanupPatterns) {
                cleanResponse = cleanResponse.replace(pattern, '').trim();
            }
            
            // If response is empty or too short, provide a fallback
            if (!cleanResponse || cleanResponse.length < 10) {
                cleanResponse = "I understand your question, but I'm having trouble generating a detailed response right now. Could you please rephrase or provide more context?";
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
            
            // Provide a more specific fallback response based on the error
            let fallbackResponse = "I'm experiencing some technical difficulties right now. ";
            
            if (aiError.message.includes('not found') || aiError.message.includes('404')) {
                fallbackResponse += "This AI model may not be available. Please try selecting a different model.";
            } else if (aiError.message.includes('loading')) {
                fallbackResponse += "The AI model is currently loading. Please try again in a few moments.";
            } else if (aiError.message.includes('rate limit')) {
                fallbackResponse += "Too many requests. Please wait a moment before trying again.";
            } else {
                fallbackResponse += "Please try again in a moment, or rephrase your question.";
            }
            
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
            
            res.json({ success: true, messages: rows });
        }
    );
});

// Get available bots
app.get('/api/bots', (req, res) => {
    const bots = Object.keys(botConfigs).map(botId => ({
        id: botId,
        name: botId.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
        model: botConfigs[botId].model
    }));
    
    res.json({ success: true, bots });
});

// Start server
async function startServer() {
    try {
        await initializeDatabase();
        
        app.listen(PORT, () => {
            console.log(`Server running on port ${PORT}`);
            console.log(`Visit http://localhost:${PORT} to use the application`);
            console.log(`Visit http://localhost:${PORT}/test to test AI models`);
            console.log(`\nMake sure to set your HF_API_KEY environment variable!`);
        });
    } catch (error) {
        console.error('Failed to start server:', error);
        process.exit(1);
    }
}

startServer();
