import { GoogleGenAI } from '@google/genai';

const ai = new GoogleGenAI({ apiKey: 'mock_key' });

async function run() {
  try {
    const history = [
      { role: 'user', content: 'hello' }
    ];
    
    const formattedHistory = history ? history.filter((msg) => msg.role && msg.content).map((msg) => ({
        role: msg.role === 'assistant' ? 'model' : 'user',
        parts: [{ text: msg.content }]
    })) : [];

    const chat = ai.chats.create({
        model: 'gemini-3.1-pro-preview',
        config: {
            systemInstruction: 'You are an AI assistant',
        },
        history: formattedHistory
    });

    const response = await chat.sendMessage({
        message: 'testing 123'
    });
    console.log(response.text);
  } catch (err) {
    console.error('Chat API Error:', err);
  }
}

run();
