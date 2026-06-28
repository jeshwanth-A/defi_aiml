"use client";

import { useState, useRef, useEffect, type Dispatch, type SetStateAction } from "react";

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

type UploadedFile = {
  name: string;
  size: number;
};

const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8432").replace(/\/$/, "");
const SUGGESTIONS = [
  "predict ethereum price",
  "give last 30 days ethereum prices",
  "give ethereum price on 23rd Jan",
];

export default function Home() {
  const [ragFiles, setRagFiles] = useState<UploadedFile[]>([]);
  const [fileFiles, setFileFiles] = useState<UploadedFile[]>([]);
  const [ragUploading, setRagUploading] = useState(false);
  const [fileUploading, setFileUploading] = useState(false);
  const [ragUploadError, setRagUploadError] = useState("");
  const [fileUploadError, setFileUploadError] = useState("");

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [memoryLimit, setMemoryLimit] = useState(5);
  const [useTools, setUseTools] = useState(true);
  const [streamingText, setStreamingText] = useState("");
  const streamingRef = useRef("");
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingText]);

  async function handleUpload(
    event: React.ChangeEvent<HTMLInputElement>,
    endpoint: string,
    setFiles: Dispatch<SetStateAction<UploadedFile[]>>,
    setUploading: (v: boolean) => void,
    setError: (e: string) => void
  ) {
    const selectedFiles = event.target.files;
    if (!selectedFiles || selectedFiles.length === 0) return;

    setUploading(true);
    setError("");

    const formData = new FormData();
    for (let i = 0; i < selectedFiles.length; i++) {
      formData.append("files", selectedFiles[i]);
    }

    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const data = await response.json();
        setError(data.detail ?? "Upload failed");
        return;
      }

      const data = await response.json();
      const newFiles = (data.files ?? []) as UploadedFile[];
      setFiles((prev) => [...prev, ...newFiles]);
    } catch {
      setError("Upload failed");
    } finally {
      setUploading(false);
    }
  }

  async function sendMessage() {
    if (!chatInput.trim() || chatLoading) return;

    const userMsg: ChatMessage = { role: "user", content: chatInput.trim() };
    setMessages((prev) => [...prev, userMsg]);
    setChatInput("");
    setChatLoading(true);
    setStreamingText("");

    if (!useTools) {
      const wsUrl = API_BASE_URL.replace("http", "ws");
      const ws = new WebSocket(`${wsUrl}/ws/chat`);
      let wsCompleted = false;
      streamingRef.current = "";

      ws.onopen = () => {
        ws.send(JSON.stringify({ message: userMsg.content, memory_limit: memoryLimit }));
      };

      ws.onmessage = (e) => {
        const data = JSON.parse(e.data);
        if (data.done) {
          wsCompleted = true;
          const finalContent = streamingRef.current;
          setMessages((prev) => [...prev, { role: "assistant", content: finalContent }]);
          setStreamingText("");
          streamingRef.current = "";
          setChatLoading(false);
          ws.close();
          return;
        }
        const token = data.token ?? "";
        streamingRef.current += token;
        setStreamingText(streamingRef.current);
      };

      ws.onerror = () => {
        wsCompleted = true;
        setMessages((prev) => [...prev, { role: "assistant", content: "Error: WebSocket connection failed" }]);
        setStreamingText("");
        streamingRef.current = "";
        setChatLoading(false);
      };

      ws.onclose = () => {
        if (!wsCompleted) {
          setMessages((prev) => [...prev, { role: "assistant", content: "Error: Connection closed unexpectedly" }]);
          setStreamingText("");
          streamingRef.current = "";
          setChatLoading(false);
        }
      };
    } else {
      try {
        const response = await fetch(`${API_BASE_URL}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: userMsg.content, memory_limit: memoryLimit }),
        });

        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(data.message ?? data.detail ?? "Chat request failed");
        }

        const assistantMsg: ChatMessage = { role: "assistant", content: data.response };
        setMessages((prev) => [...prev, assistantMsg]);
      } catch (error) {
        const message = error instanceof Error ? error.message : "Failed to get response";
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: `Error: ${message}` },
        ]);
      } finally {
        setChatLoading(false);
      }
    }
  }

  return (
    <main className="main">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #1e1e1e; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #2d2d2d; }
        ::-webkit-scrollbar-thumb { background: #555; border-radius: 3px; }
        @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }
        .main { min-height: 100vh; background: #1e1e1e; color: #d4d4d4; font-family: 'JetBrains Mono', monospace; padding: 24px; box-sizing: border-box; }
        .title { text-align: center; font-size: 22px; font-weight: 600; margin-bottom: 24px; color: #e0e0e0; letter-spacing: 1px; }
        .layout { display: flex; gap: 24px; max-width: 1200px; margin: 0 auto; height: calc(100vh - 100px); }
        .leftCol { width: 300px; display: flex; flex-direction: column; gap: 20px; }
        .uploadSection { }
        .uploadSectionH2 { font-size: 14px; font-weight: 500; margin-bottom: 8px; }
        .uploadSectionH2Rag { color: #9cdcfe; }
        .uploadSectionH2File { color: #ce9178; }
        .uploadBox { border: 1px dashed #555; border-radius: 10px; padding: 16px; min-height: 140px; display: flex; flex-direction: column; align-items: center; justify-content: center; background: #252526; }
        .uploadLabel { cursor: pointer; display: flex; align-items: center; justify-content: center; width: 36px; height: 36px; border-radius: 50%; border: 1px solid #555; color: #d4d4d4; font-size: 18px; }
        .uploadMsg { margin-top: 8px; font-size: 12px; }
        .uploadMsgGray { color: #888; }
        .uploadMsgRed { color: #f48771; }
        .fileList { margin-top: 12px; list-style: none; width: 100%; max-height: 80px; overflow-y: auto; }
        .fileItem { font-size: 12px; padding: 4px 0; border-bottom: 1px solid #333; color: #aaa; }
        .chatCol { flex: 1; display: flex; flex-direction: column; background: #252526; border-radius: 10px; border: 1px solid #333; overflow: hidden; }
        .ctrlBar { padding: 10px 16px; border-bottom: 1px solid #333; display: flex; align-items: center; justify-content: space-between; }
        .ctrlGroup { display: flex; align-items: center; gap: 16px; }
        .ctrlSub { display: flex; align-items: center; gap: 8px; }
        .ctrlLabel { font-size: 12px; color: #888; }
        .numInput { width: 50px; padding: 4px 8px; font-size: 12px; border-radius: 4px; border: 1px solid #444; background: #1e1e1e; color: #d4d4d4; font-family: 'JetBrains Mono', monospace; outline: none; }
        .selInput { padding: 4px 8px; font-size: 12px; border-radius: 4px; border: 1px solid #444; background: #1e1e1e; color: #d4d4d4; font-family: 'JetBrains Mono', monospace; outline: none; cursor: pointer; }
        .clearBtn { padding: 4px 12px; font-size: 11px; cursor: pointer; border-radius: 4px; border: 1px solid #444; background: transparent; color: #f48771; font-family: 'JetBrains Mono', monospace; }
        .msgArea { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 12px; }
        .emptyMsg { color: #666; text-align: center; margin-top: 40%; font-size: 14px; }
        .bubble { max-width: 75%; padding: 10px 14px; border-radius: 10px; color: #d4d4d4; font-size: 13px; line-height: 1.5; white-space: pre-wrap; }
        .bubbleUser { align-self: flex-end; background: #007acc; }
        .bubbleAssistant { align-self: flex-start; background: #2d2d2d; }
        .bubbleStream { align-self: flex-start; background: #2d2d2d; }
        .bubbleThink { align-self: flex-start; padding: 10px 14px; border-radius: 10px; background: #2d2d2d; color: #888; font-size: 13px; }
        .suggestionBar { padding: 10px 16px 0; border-top: 1px solid #333; display: flex; gap: 8px; flex-wrap: wrap; }
        .suggestionBtn { padding: 6px 10px; font-size: 11px; border-radius: 6px; border: 1px solid #444; background: #1e1e1e; color: #9cdcfe; font-family: 'JetBrains Mono', monospace; cursor: pointer; }
        .suggestionBtn:disabled { color: #666; cursor: not-allowed; }
        .inputBar { padding: 16px; border-top: 1px solid #333; display: flex; gap: 10px; }
        .chatInput { flex: 1; padding: 10px 14px; font-size: 13px; border-radius: 8px; border: 1px solid #444; background: #1e1e1e; color: #d4d4d4; font-family: 'JetBrains Mono', monospace; outline: none; }
        .sendBtn { padding: 10px 20px; font-size: 13px; border-radius: 8px; border: none; color: #fff; font-family: 'JetBrains Mono', monospace; font-weight: 500; cursor: pointer; }
        .sendBtnActive { background: #007acc; }
        .sendBtnDisabled { background: #555; cursor: not-allowed; }
      `}</style>

      <h1 className="title">
        Defi Predictor and Analyst
      </h1>

      <div className="layout">
        {/* Left Column: Upload Boxes */}
        <div className="leftCol">
          {/* Rag Box */}
          <div className="uploadSection">
            <h2 className="uploadSectionH2 uploadSectionH2Rag">Rag Box</h2>
            <div className="uploadBox">
              <label className="uploadLabel">
                +
                <input
                  type="file"
                  accept=".md,.txt"
                  multiple
                  onChange={(e) => handleUpload(e, "/ragupload", setRagFiles, setRagUploading, setRagUploadError)}
                  style={{ display: "none" }}
                />
              </label>
              {ragUploading && <p className="uploadMsg uploadMsgGray">Uploading...</p>}
              {ragUploadError && <p className="uploadMsg uploadMsgRed">{ragUploadError}</p>}
              {ragFiles.length > 0 && (
                <ul className="fileList">
                  {ragFiles.map((f, i) => (
                    <li key={i} className="fileItem">{f.name}</li>
                  ))}
                </ul>
              )}
            </div>
          </div>

          {/* File Box */}
          <div className="uploadSection">
            <h2 className="uploadSectionH2 uploadSectionH2File">File Box</h2>
            <div className="uploadBox">
              <label className="uploadLabel">
                +
                <input
                  type="file"
                  accept=".md,.txt"
                  multiple
                  onChange={(e) => handleUpload(e, "/fileupload", setFileFiles, setFileUploading, setFileUploadError)}
                  style={{ display: "none" }}
                />
              </label>
              {fileUploading && <p className="uploadMsg uploadMsgGray">Uploading...</p>}
              {fileUploadError && <p className="uploadMsg uploadMsgRed">{fileUploadError}</p>}
              {fileFiles.length > 0 && (
                <ul className="fileList">
                  {fileFiles.map((f, i) => (
                    <li key={i} className="fileItem">{f.name}</li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        </div>

        {/* Right Column: Chat */}
        <div className="chatCol">
          {/* Control Bar */}
          <div className="ctrlBar">
            <div className="ctrlGroup">
              {/* Memory Control */}
              <div className="ctrlSub">
                <span className="ctrlLabel">Memory:</span>
                <input
                  type="number"
                  min={1}
                  max={100}
                  value={memoryLimit}
                  onChange={(e) => setMemoryLimit(Math.max(1, parseInt(e.target.value) || 1))}
                  className="numInput"
                />
                <span className="ctrlLabel">messages</span>
              </div>

              {/* Use Tools Toggle */}
              <div className="ctrlSub" style={{ gap: 6 }}>
                <span className="ctrlLabel">Use Tools:</span>
                <select
                  value={useTools ? "Yes" : "No"}
                  onChange={(e) => setUseTools(e.target.value === "Yes")}
                  className="selInput"
                >
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                </select>
              </div>
            </div>

            <button
              onClick={() => {
                setMessages([]);
                fetch(`${API_BASE_URL}/reset`, { method: "POST" }).catch(() => {});
              }}
              className="clearBtn"
            >
              Clear Chat
            </button>
          </div>

          {/* Messages */}
          <div className="msgArea">
            {messages.length === 0 && !streamingText && (
              <p className="emptyMsg">Ask a question to get started</p>
            )}
            {messages.map((msg, i) => (
              <div
                key={i}
                className={`bubble ${msg.role === "user" ? "bubbleUser" : "bubbleAssistant"}`}
              >
                {msg.content}
              </div>
            ))}
            {streamingText && (
              <div className="bubble bubbleStream">
                {streamingText}
                <span style={{ animation: "blink 1s infinite" }}>▊</span>
              </div>
            )}
            {chatLoading && !streamingText && (
              <div className="bubbleThink">Thinking...</div>
            )}
            <div ref={chatEndRef} />
          </div>

          {/* Input Bar */}
          <div className="suggestionBar">
            {SUGGESTIONS.map((suggestion) => (
              <button
                key={suggestion}
                type="button"
                disabled={chatLoading}
                onClick={() => setChatInput(suggestion)}
                className="suggestionBtn"
              >
                {suggestion}
              </button>
            ))}
          </div>
          <div className="inputBar">
            <input
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && sendMessage()}
              placeholder="Type your message..."
              className="chatInput"
            />
            <button
              onClick={sendMessage}
              disabled={chatLoading}
              className={`sendBtn ${chatLoading ? "sendBtnDisabled" : "sendBtnActive"}`}
            >
              enter
            </button>
          </div>
        </div>
      </div>
    </main>
  );
}
