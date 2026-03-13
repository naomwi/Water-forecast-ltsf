"use client";

import { useState, useRef, useEffect } from "react";
import { Send, Bot, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";

type Message = {
  role: "user" | "assistant";
  content: string;
};

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content: "Hello! I am HydroBot, the intelligent assistant for the FPT University Water Quality Forecasting Project. How can I help you analyze the data today?",
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMsg = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: userMsg }]);
    setIsLoading(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMsg, history: messages }),
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.reply || errorData.error || `Failed to fetch response: ${res.status}`);
      }

      const data = await res.json();
      setMessages((prev) => [...prev, { role: "assistant", content: data.reply }]);
    } catch (error: unknown) {
      console.error(error);
      const errorMessage = error instanceof Error ? error.message : "I encountered an error. Please check the logs or try again.";
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: errorMessage },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-4rem)] bg-zinc-50 items-center py-8">
      <Card className="flex flex-col w-full max-w-4xl h-full shadow-lg border-zinc-200 bg-white">
        {/* Chat Header */}
        <div className="flex items-center gap-3 p-4 border-b bg-blue-50/50">
          <div className="p-2 bg-blue-100 rounded-full text-blue-600">
            <Bot size={24} />
          </div>
          <div>
            <h2 className="font-semibold text-lg text-slate-800">HydroBot Assistant</h2>
            <p className="text-sm text-slate-500">Powered by Gemini 3.1 Pro Preview</p>
          </div>
        </div>

        {/* Chat Messages */}
        <div className="flex-1 overflow-y-auto p-6 min-h-0">
          <div className="flex flex-col gap-6">
            {messages.map((msg, i) => (
              <div
                key={i}
                className={`flex gap-4 ${msg.role === "user" ? "flex-row-reverse" : "flex-row"}`}
              >
                <div
                  className={`flex-shrink-0 flex items-center justify-center w-10 h-10 rounded-full ${msg.role === "user" ? "bg-slate-800 text-white" : "bg-blue-100 text-blue-600"
                    }`}
                >
                  {msg.role === "user" ? <User size={20} /> : <Bot size={20} />}
                </div>
                <div
                  className={`px-4 py-3 rounded-2xl max-w-[80%] ${msg.role === "user"
                      ? "bg-slate-800 text-white rounded-tr-sm"
                      : "bg-zinc-100/80 text-slate-800 rounded-tl-sm border border-zinc-200"
                    }`}
                >
                  <p className="whitespace-pre-wrap leading-relaxed">{msg.content}</p>
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex gap-4 flex-row">
                <div className="flex items-center justify-center w-10 h-10 rounded-full bg-blue-100 text-blue-600">
                  <Bot size={20} />
                </div>
                <div className="px-4 py-3 rounded-2xl bg-zinc-100/80 text-slate-500 border border-zinc-200 flex items-center gap-2">
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" />
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:0.2s]" />
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:0.4s]" />
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>
        </div>

        {/* Input Area */}
        <div className="p-4 border-t bg-white">
          <form
            onSubmit={sendMessage}
            className="flex items-center bg-zinc-50 border border-zinc-300 rounded-full p-1 pl-4 focus-within:ring-2 focus-within:ring-blue-100 focus-within:border-blue-400 transition-all"
          >
            <Input
              type="text"
              placeholder="Ask about water quality models..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              className="flex-1 bg-transparent border-0 focus-visible:ring-0 shadow-none text-slate-800 placeholder:text-slate-400 px-0"
              disabled={isLoading}
            />
            <Button
              type="submit"
              size="icon"
              disabled={!input.trim() || isLoading}
              className="rounded-full bg-blue-600 hover:bg-blue-700 text-white shrink-0 ml-2"
            >
              <Send size={18} className={input.trim() ? "translate-x-0.5" : ""} />
            </Button>
          </form>
        </div>
      </Card>
    </div>
  );
}
