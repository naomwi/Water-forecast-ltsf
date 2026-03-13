import { NextResponse } from 'next/server';
import { GoogleGenAI } from '@google/genai';
import { detectIntent, buildPredictionContext } from '@/lib/predictionLoader';
import systemContextCacheData from './system_context_cache.json';

// Initialize Gemini Client
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || '' });

// Cache the system context so we don't read it from disk on every chat request
let cachedSystemContext: string | null = null;
if (systemContextCacheData && systemContextCacheData.success) {
    cachedSystemContext = systemContextCacheData.context;
    console.log("Loaded system context from cache file.");
} else {
    console.log("Cache file not found or invalid.");
    // Fallback basic system instruction if cache fails
    cachedSystemContext = 'You are HydroBot, an expert AI Water Quality Data Analyst designed for the FPT University Capstone Project.';
}

// Define message type
interface ChatMessage {
    role?: string;
    content?: string;
}

export async function POST(request: Request) {
    try {
        const { message, history } = await request.json();

        if (!process.env.GEMINI_API_KEY) {
            return NextResponse.json({ reply: 'Error: GEMINI_API_KEY is not defined in environment variables.' }, { status: 500 });
        }

        // 1. Get System Context (Cached)
        // We already loaded cachedSystemContext at the top via import.

        // 2. Get Live Context based on prompt using native TypeScript implementation
        let enhancedPrompt = message;
        try {
            const intent = detectIntent(message);

            if (intent.is_prediction && intent.horizon !== null && intent.features.length > 0) {
                const siteToUse = intent.site || 1463500;
                const predContext = await buildPredictionContext(intent.features, intent.horizon, siteToUse);
                enhancedPrompt = `${predContext}\n\n[USER QUESTION]\n${message}`;
            }
        } catch (e) {
            console.error("Failed to load live prediction context:", e);
        }

        // 3. Format history for GoogleGenAI SDK
        // The history needs to match what GenAI expects.
        // Assuming history looks like: [{role: 'user'|'assistant', content: '...'}, ...]
        // We filter out any initial "model" messages because the API requires the conversation to start with a user message.
        let validHistory: ChatMessage[] = history ? history.filter((msg: ChatMessage) => msg.role && msg.content) : [];
        if (validHistory.length > 0 && validHistory[0].role === 'assistant') {
            validHistory = validHistory.slice(1);
        }

        const formattedHistory = validHistory.map((msg: ChatMessage) => ({
            role: msg.role === 'assistant' ? 'model' : 'user',
            parts: [{ text: msg.content || '' }]
        }));

        // 4. Send to Gemini
        const chat = ai.chats.create({
            model: 'gemini-3.1-pro-preview',
            config: {
                systemInstruction: cachedSystemContext || 'You are an AI assistant',
            },
            history: formattedHistory
        });

        const response = await chat.sendMessage({
            message: enhancedPrompt
        });

        return NextResponse.json({ reply: response.text });
    } catch (error: unknown) {
        console.error('Chat API Error:', error);
        const errorMessage = error instanceof Error ? error.message : 'Internal Server Error';
        return NextResponse.json({ 
            reply: errorMessage 
        }, { status: 500 });
    }
}
