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

// Hugging Face API configuration - try multiple possible env var names
const HF_API_KEY = process.env.HF_API_KEY || process.env.HUGGINGFACE_API_KEY || process.env.HF_TOKEN || process.env.HUGGING_FACE_TOKEN;
const HF_API_URL = 'https://api-inference.huggingface.co/models/';

// Updated bot configurations with PROGRAMMING-FOCUSED models supporting 60+ languages
const botConfigs = {
    'starcoder2-3b': {
        model: 'bigcode/starcoder2-3b',
        maxTokens: 1024,
        temperature: 0.2,
        description: 'StarCoder2 3B - Advanced code generation model supporting 600+ programming languages'
    },
    'starcoder2-7b': {
        model: 'bigcode/starcoder2-7b',
        maxTokens: 1024,
        temperature: 0.2,
        description: 'StarCoder2 7B - Large code generation model with exceptional multilingual support'
    },
    'starcoder2-15b': {
        model: 'bigcode/starcoder2-15b',
        maxTokens: 1024,
        temperature: 0.2,
        description: 'StarCoder2 15B - Premium code generation model for complex programming tasks'
    },
    'starcoder': {
        model: 'bigcode/starcoder',
        maxTokens: 1024,
        temperature: 0.2,
        description: 'Original StarCoder - Proven code generation model supporting 80+ languages'
    },
    'starcoderbase': {
        model: 'bigcode/starcoderbase',
        maxTokens: 1024,
        temperature: 0.2,
        description: 'StarCoder Base - Foundation model for code completion and generation'
    },
    'codegen2-1b': {
        model: 'Salesforce/codegen2-1B',
        maxTokens: 768,
        temperature: 0.2,
        description: 'CodeGen2 1B - Fast code generation for multiple programming languages'
    },
    'codegen2-3b': {
        model: 'Salesforce/codegen2-3.7B',
        maxTokens: 768,
        temperature: 0.2,
        description: 'CodeGen2 3.7B - Balanced performance code generation model'
    },
    'codegen2-7b': {
        model: 'Salesforce/codegen2-7B',
        maxTokens: 768,
        temperature: 0.2,
        description: 'CodeGen2 7B - Large scale code generation with high accuracy'
    },
    'codegen2-16b': {
        model: 'Salesforce/codegen2-16B',
        maxTokens: 768,
        temperature: 0.2,
        description: 'CodeGen2 16B - Premium code generation model for complex tasks'
    },
    'codet5p-220m': {
        model: 'Salesforce/codet5p-220m',
        maxTokens: 512,
        temperature: 0.2,
        description: 'CodeT5+ 220M - Compact code understanding and generation model'
    },
    'codet5p-770m': {
        model: 'Salesforce/codet5p-770m',
        maxTokens: 512,
        temperature: 0.2,
        description: 'CodeT5+ 770M - Enhanced code generation with better language support'
    },
    'codet5p-2b': {
        model: 'Salesforce/codet5p-2b',
        maxTokens: 512,
        temperature: 0.2,
        description: 'CodeT5+ 2B - Advanced code generation supporting numerous languages'
    },
    'codet5p-6b': {
        model: 'Salesforce/codet5p-6b',
        maxTokens: 512,
        temperature: 0.2,
        description: 'CodeT5+ 6B - Large scale code model with multilingual expertise'
    },
    'santacoder': {
        model: 'bigcode/santacoder',
        maxTokens: 768,
        temperature: 0.2,
        description: 'SantaCoder - Specialized in Python, JavaScript, and Java code generation'
    },
    'incoder-1b': {
        model: 'facebook/incoder-1B',
        maxTokens: 768,
        temperature: 0.2,
        description: 'InCoder 1B - Fill-in-the-middle code generation model'
    },
    'incoder-6b': {
        model: 'facebook/incoder-6B',
        maxTokens: 768,
        temperature: 0.2,
        description: 'InCoder 6B - Advanced fill-in-the-middle programming assistant'
    },
    'wizardcoder-1b': {
        model: 'WizardLM/WizardCoder-1B-V1.0',
        maxTokens: 768,
        temperature: 0.2,
        description: 'WizardCoder 1B - Instruction-tuned code generation model'
    },
    'wizardcoder-3b': {
        model: 'WizardLM/WizardCoder-3B-V1.0',
        maxTokens: 768,
        temperature: 0.2,
        description: 'WizardCoder 3B - Enhanced instruction-following code assistant'
    },
    'wizardcoder-15b': {
        model: 'WizardLM/WizardCoder-15B-V1.0',
        maxTokens: 768,
        temperature: 0.2,
        description: 'WizardCoder 15B - Premium instruction-tuned programming model'
    },
    'replit-code-3b': {
        model: 'replit/replit-code-v1_5-3b',
        maxTokens: 768,
        temperature: 0.2,
        description: 'Replit Code 3B - Multi-language code completion and generation'
    },
    'deepseek-coder-1b': {
        model: 'deepseek-ai/deepseek-coder-1.3b-base',
        maxTokens: 768,
        temperature: 0.2,
        description: 'DeepSeek Coder 1.3B - Specialized code generation model'
    },
    'deepseek-coder-7b': {
        model: 'deepseek-ai/deepseek-coder-7b-base',
        maxTokens: 768,
        temperature: 0.2,
        description: 'DeepSeek Coder 7B - Advanced programming assistant'
    },
    'phi-1-code': {
        model: 'microsoft/phi-1',
        maxTokens: 512,
        temperature: 0.2,
        description: 'Phi-1 - Microsoft\'s compact code generation model'
    },
    'phi-1_5-code': {
        model: 'microsoft/phi-1_5',
        maxTokens: 512,
        temperature: 0.2,
        description: 'Phi-1.5 - Enhanced version with better code understanding'
    }
};

// Enhanced Hugging Face API interaction with better error handling
async function queryHuggingFace(model, prompt, config) {
    if (!HF_API_KEY) {
        console.error('API Key check failed. Checked these environment variables:');
        console.error('- HF_API_KEY:', process.env.HF_API_KEY ? 'SET' : 'NOT SET');
        console.error('- HUGGINGFACE_API_KEY:', process.env.HUGGINGFACE_API_KEY ? 'SET' : 'NOT SET');
        console.error('- HF_TOKEN:', process.env.HF_TOKEN ? 'SET' : 'NOT SET');
        console.error('- HUGGING_FACE_TOKEN:', process.env.HUGGING_FACE_TOKEN ? 'SET' : 'NOT SET');
        throw new Error('Hugging Face API key not configured. Please set HF_API_KEY, HUGGINGFACE_API_KEY, HF_TOKEN, or HUGGING_FACE_TOKEN in your environment variables.');
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
                        top_k: 50,
                        repetition_penalty: 1.1
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
                if (response.status === 401) {
                    throw new Error(`Invalid API key. Please check your Hugging Face API key.`);
                } else if (response.status === 404) {
                    throw new Error(`Model '${model}' not found on Hugging Face`);
                } else if (response.status === 503) {
                    // Model is loading, wait and retry
                    if (attempt < maxRetries) {
                        console.log(`Model ${model} is loading, waiting 15 seconds...`);
                        await new Promise(resolve => setTimeout(resolve, 15000));
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
            if (error.message.includes('not found') || error.message.includes('404') || error.message.includes('Invalid API key')) {
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
    const testPrompts = [
        "def fibonacci(n):\n    # Generate fibonacci sequence\n    if n <= 1:\n        return n\n    return",
        "function quickSort(arr) {\n    // Implement quicksort algorithm\n    if (arr.length <= 1) {\n        return arr;\n    }\n    const pivot =",
        "class Calculator {\n    // Simple calculator class\n    constructor() {\n        this.result = 0;\n    }\n    \n    add(num) {"
    ];
    
    const testPrompt = testPrompts[Math.floor(Math.random() * testPrompts.length)];
    const startTime = Date.now();
    
    try {
        const response = await queryHuggingFace(config.model, testPrompt, config);
        const responseTime = Date.now() - startTime;
        
        return {
            botId,
            model: config.model,
            description: config.description,
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
            description: config.description,
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
            message: 'Please set one of these environment variables: HF_API_KEY, HUGGINGFACE_API_KEY, HF_TOKEN, or HUGGING_FACE_TOKEN',
            checkedVars: {
                HF_API_KEY: !!process.env.HF_API_KEY,
                HUGGINGFACE_API_KEY: !!process.env.HUGGINGFACE_API_KEY,
                HF_TOKEN: !!process.env.HF_TOKEN,
                HUGGING_FACE_TOKEN: !!process.env.HUGGING_FACE_TOKEN
            }
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
                        description: config.description,
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
                <title>Programming AI Models Test Results</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                    .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
                    .summary { background: #ecf0f1; padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 4px solid #3498db; }
                    .model { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 8px; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                    .working { border-left: 5px solid #27ae60; }
                    .error { border-left: 5px solid #e74c3c; }
                    .response { background: #f8f9fa; padding: 12px; margin-top: 10px; border-radius: 4px; font-family: 'Courier New', monospace; white-space: pre-wrap; border: 1px solid #dee2e6; }
                    .error-text { color: #e74c3c; font-weight: 500; }
                    .success-text { color: #27ae60; font-weight: 500; }
                    .refresh-btn { background: #3498db; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; margin-bottom: 20px; font-size: 14px; }
                    .refresh-btn:hover { background: #2980b9; }
                    .test-single { margin-bottom: 20px; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                    .test-single input { padding: 10px; margin-right: 10px; width: 300px; border: 1px solid #ddd; border-radius: 4px; }
                    .test-single button { padding: 10px 20px; background: #f39c12; color: white; border: none; border-radius: 4px; cursor: pointer; }
                    .test-single button:hover { background: #e67e22; }
                    .model-list { background: #f8f9fa; padding: 15px; border-radius: 4px; margin: 10px 0; border: 1px solid #dee2e6; }
                    .api-key-status { background: #d4edda; padding: 15px; border-radius: 4px; margin: 10px 0; border: 1px solid #c3e6cb; color: #155724; }
                    .description { color: #666; font-style: italic; margin-top: 5px; }
                    .model-name { font-weight: bold; color: #2c3e50; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🚀 Programming AI Models Test Results</h1>
                    <p>Testing specialized code generation models supporting 60+ programming languages including Python, JavaScript, Java, C++, C#, Go, Rust, PHP, Ruby, Swift, Kotlin, and many more!</p>
                </div>
                
                <div class="api-key-status">
                    <strong>✅ API Key Status:</strong> Configured and Ready
                </div>
                
                <div class="test-single">
                    <h3>🔍 Test Individual Model:</h3>
                    <input type="text" id="botId" placeholder="Enter bot ID (e.g., starcoder2-3b)" list="botList">
                    <datalist id="botList">
                        ${Object.keys(botConfigs).map(id => `<option value="${id}">`).join('')}
                    </datalist>
                    <button onclick="testSingle()">Test Model</button>
                    
                    <div class="model-list">
                        <strong>Available Programming Models:</strong><br>
                        ${Object.entries(botConfigs).map(([id, config]) => 
                            `<span class="model-name">${id}</span> - ${config.description}`
                        ).join('<br>')}
                    </div>
                </div>
                
                <button class="refresh-btn" onclick="location.reload()">🔄 Refresh All Tests</button>
                
                <div class="summary">
                    <h2>📊 Test Summary</h2>
                    <p><strong>Total Programming Models:</strong> ${summary.totalModels}</p>
                    <p><strong class="success-text">✅ Working:</strong> ${summary.working}</p>
                    <p><strong class="error-text">❌ Errors:</strong> ${summary.errors}</p>
                    <p><strong>🟢 Working Models:</strong> ${summary.workingModels.join(', ') || 'None'}</p>
                    <p><strong>🔴 Error Models:</strong> ${summary.errorModels.join(', ') || 'None'}</p>
                </div>
                
                <h2>📋 Detailed Results</h2>
            `;
            
            results.forEach(result => {
                const statusClass = result.status === 'working' ? 'working' : 'error';
                const statusIcon = result.status === 'working' ? '✅' : '❌';
                html += `
                <div class="model ${statusClass}">
                    <h3>${statusIcon} ${result.botId}</h3>
                    <p><strong>Model:</strong> <span class="model-name">${result.model}</span></p>
                    <p class="description">${result.description}</p>
                    <p><strong>Status:</strong> <span class="${result.status === 'working' ? 'success-text' : 'error-text'}">${result.status.toUpperCase()}</span></p>
                    <p><strong>Response Time:</strong> ${result.responseTime}</p>
                    ${result.error ? `<p><strong>Error:</strong> <span class="error-text">${result.error}</span></p>` : ''}
                    ${result.response ? `<div class="response"><strong>📝 Sample Code Generation:</strong><br>${result.response}</div>` : ''}
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

// Handle chat messages with improved prompt formatting for code generation
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

        // Build enhanced context prompt for code generation
        let contextPrompt = "";
        
        // Enhanced system prompt for programming models
        contextPrompt = `# Advanced Programming Assistant

You are an expert programmer who specializes in multiple programming languages including:
- Python, JavaScript, TypeScript, Java, C++, C#, Go, Rust
- PHP, Ruby, Swift, Kotlin, Scala, R, MATLAB
- HTML, CSS, SQL, Shell/Bash scripting
- And 60+ other programming languages

Provide clean, efficient, well-commented code with explanations when helpful.

## Conversation History:
`;
        
        // Add conversation history (last 4 exchanges)
        const recentHistory = history.slice(-8); // Last 8 messages = 4 exchanges
        for (const msg of recentHistory) {
            if (msg.role === 'user') {
                contextPrompt += `User: ${msg.content}\n`;
            } else {
                contextPrompt += `Assistant: ${msg.content}\n`;
            }
        }
        
        contextPrompt += `User: ${message}\nAssistant: `;

        // Get bot configuration
        const config = botConfigs[botId];
        if (!config) {
            return res.status(400).json({ error: 'Invalid bot configuration' });
        }

        try {
            // Query the AI model
            const aiResponse = await queryHuggingFace(config.model, contextPrompt, config);
            
            // Enhanced response cleaning for code models
            let cleanResponse = aiResponse.trim();
            
            // Remove any repeated context or system prompts
            const cleanupPatterns = [
                /^(User:|Assistant:|Human:|AI:|# Advanced Programming Assistant)/gim,
                /^(You are an expert programmer|## Conversation History:)/gim,
                /^(Provide clean, efficient)/gim
            ];
            
            for (const pattern of cleanupPatterns) {
                cleanResponse = cleanResponse.replace(pattern, '').trim();
            }
            
            // Remove duplicate lines that might occur
            const lines = cleanResponse.split('\n');
            const uniqueLines = [];
            let lastLine = '';
            
            for (const line of lines) {
                if (line.trim() !== lastLine.trim() || line.trim().length === 0) {
                    uniqueLines.push(line);
                }
                lastLine = line;
            }
            
            cleanResponse = uniqueLines.join('\n').trim();
            
            // If response is empty or too short, provide a fallback
            if (!cleanResponse || cleanResponse.length < 15) {
                cleanResponse = "I understand your programming question, but I'm having trouble generating a detailed code response right now. Could you please provide more specific details about what you're trying to implement?";
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
            let fallbackResponse = "I'm experiencing some technical difficulties with the programming AI model right now. ";
            
            if (aiError.message.includes('not found') || aiError.message.includes('404')) {
                fallbackResponse += "This programming model may not be available. Please try selecting a different coding model from the list.";
            } else if (aiError.message.includes('loading')) {
                fallbackResponse += "The AI model is currently loading. Please try again in a few moments.";
            } else if (aiError.message.includes('rate limit')) {
                fallbackResponse += "Too many requests. Please wait a moment before trying again.";
            } else {
                fallbackResponse += "Please try rephrasing your programming question or try a different model.";
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

// Get available bots with enhanced information
app.get('/api/bots', (req, res) => {
    const bots = Object.keys(botConfigs).map(botId => ({
        id: botId,
        name: botId.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
        model: botConfigs[botId].model,
        description: botConfigs[botId].description,
        category: 'Programming & Code Generation',
        languages: '60+ Programming Languages',
        maxTokens: botConfigs[botId].maxTokens
    }));
    
    res.json({ success: true, bots });
});

// Get model statistics and capabilities
app.get('/api/models/info', (req, res) => {
    const modelInfo = {
        totalModels: Object.keys(botConfigs).length,
        categories: {
            'Code Generation': Object.keys(botConfigs).filter(k => k.includes('coder') || k.includes('code')).length,
            'StarCoder Family': Object.keys(botConfigs).filter(k => k.includes('star')).length,
            'CodeGen Family': Object.keys(botConfigs).filter(k => k.includes('codegen')).length,
            'Specialized Models': Object.keys(botConfigs).filter(k => k.includes('wizard') || k.includes('deepseek') || k.includes('phi')).length
        },
        supportedLanguages: [
            'Python', 'JavaScript', 'TypeScript', 'Java', 'C++', 'C#', 'C', 'Go', 'Rust',
            'PHP', 'Ruby', 'Swift', 'Kotlin', 'Scala', 'R', 'MATLAB', 'Perl', 'Lua',
            'HTML', 'CSS', 'SQL', 'Shell/Bash', 'PowerShell', 'Assembly', 'Fortran',
            'COBOL', 'Pascal', 'Delphi', 'Visual Basic', 'F#', 'Haskell', 'Clojure',
            'Erlang', 'Elixir', 'Julia', 'Dart', 'Objective-C', 'Groovy', 'CoffeeScript',
            'Elm', 'Reason', 'OCaml', 'Nim', 'Crystal', 'Zig', 'V', 'D', 'Ada',
            'Prolog', 'Lisp', 'Scheme', 'Racket', 'Smalltalk', 'Tcl', 'AWK', 'SED',
            'Makefile', 'CMake', 'Dockerfile', 'YAML', 'JSON', 'XML', 'LaTeX'
        ],
        modelDetails: Object.entries(botConfigs).map(([id, config]) => ({
            id,
            model: config.model,
            description: config.description,
            maxTokens: config.maxTokens,
            temperature: config.temperature
        }))
    };
    
    res.json({ success: true, info: modelInfo });
});

// Health check endpoint
app.get('/api/health', (req, res) => {
    res.json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        database: db ? 'connected' : 'disconnected',
        apiKey: HF_API_KEY ? 'configured' : 'missing',
        totalModels: Object.keys(botConfigs).length
    });
});

// Start server
async function startServer() {
    try {
        await initializeDatabase();
        
        // API Key check on startup
        console.log('\n=== 🔑 API Configuration Check ===');
        console.log('HF_API_KEY:', process.env.HF_API_KEY ? '✅ SET' : '❌ NOT SET');
        console.log('HUGGINGFACE_API_KEY:', process.env.HUGGINGFACE_API_KEY ? '✅ SET' : '❌ NOT SET');
        console.log('HF_TOKEN:', process.env.HF_TOKEN ? '✅ SET' : '❌ NOT SET');
        console.log('HUGGING_FACE_TOKEN:', process.env.HUGGING_FACE_TOKEN ? '✅ SET' : '❌ NOT SET');
        console.log('Using API Key:', HF_API_KEY ? '✅ YES' : '❌ NO');
        console.log('=====================================\n');
        
        // Model summary
        console.log('=== 🚀 Programming AI Models Loaded ===');
        console.log(`Total Models: ${Object.keys(botConfigs).length}`);
        console.log('Model Categories:');
        console.log(`  • StarCoder Family: ${Object.keys(botConfigs).filter(k => k.includes('star')).length} models`);
        console.log(`  • CodeGen Family: ${Object.keys(botConfigs).filter(k => k.includes('codegen')).length} models`);
        console.log(`  • CodeT5+ Family: ${Object.keys(botConfigs).filter(k => k.includes('codet5')).length} models`);
        console.log(`  • Specialized Models: ${Object.keys(botConfigs).filter(k => k.includes('wizard') || k.includes('deepseek') || k.includes('phi') || k.includes('replit')).length} models`);
        console.log('=======================================\n');
        
        app.listen(PORT, () => {
            console.log(`🌟 Programming AI Chat Server is running!`);
            console.log(`📍 Server URL: http://localhost:${PORT}`);
            console.log(`🧪 Test Models: http://localhost:${PORT}/test`);
            console.log(`📊 Model Info: http://localhost:${PORT}/api/models/info`);
            console.log(`💾 Health Check: http://localhost:${PORT}/api/health`);
            
            if (!HF_API_KEY) {
                console.log(`\n⚠️  WARNING: No Hugging Face API key found!`);
                console.log(`Please set one of these environment variables:`);
                console.log(`- HF_API_KEY`);
                console.log(`- HUGGINGFACE_API_KEY`);
                console.log(`- HF_TOKEN`);
                console.log(`- HUGGING_FACE_TOKEN`);
                console.log(`\nGet your free API key at: https://huggingface.co/settings/tokens`);
            } else {
                console.log(`\n✅ Hugging Face API key is configured`);
                console.log(`🎯 Ready to generate code in 60+ programming languages!`);
            }
            
            console.log(`\n🎨 Supported Languages Include:`);
            console.log(`Python, JavaScript, Java, C++, C#, Go, Rust, PHP, Ruby, Swift, Kotlin, and many more!`);
        });
    } catch (error) {
        console.error('❌ Failed to start server:', error);
        process.exit(1);
    }
}

startServer();
