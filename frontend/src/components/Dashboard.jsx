import React, { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import "./Dashboard.css";

const Dashboard = ({ user, setUser }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);
  const navigate = useNavigate();

  // Initial chat history data
  const chatHistory = [
    { id: 1, title: "Refining Project Alpha", time: "2 minutes ago", dateGroup: "TODAY" },
    { id: 2, title: "Data Analysis Trends", time: "4 hours ago", dateGroup: "TODAY" },
    { id: 3, title: "Creative Brief Brainstorm", time: "Yesterday, 10:45 PM", dateGroup: "YESTERDAY" },
    { id: 4, title: "Python Script Debugging", time: "Yesterday, 2:15 PM", dateGroup: "YESTERDAY" },
    { id: 5, title: "Project Alpha Roadmap", time: "3 days ago", dateGroup: "PREVIOUS 7 DAYS" },
  ];

  // Check if user is authenticated
  useEffect(() => {
    const currentUser = localStorage.getItem("currentUser");
    if (!currentUser && !user) {
      navigate("/");
    }
  }, [navigate, user]);

  // Initial bot animation and welcome message
  useEffect(() => {
    setIsTyping(true);
    const timer = setTimeout(() => {
      setMessages([
        {
          id: 1,
          text: "Hello! I've analyzed your project brief. Would you like to start with the UI design system or the core functionality architecture?",
          sender: "bot",
          time: "10:30 AM",
        },
        {
          id: 2,
          text: "Let's start with the UI design system. I want it to feel modern, airy, and professional with teal accents.",
          sender: "user",
          time: "10:32 AM",
        },
      ]);
      setIsTyping(false);
    }, 1000);

    return () => clearTimeout(timer);
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleSend = (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = {
      id: messages.length + 1,
      text: input,
      sender: "user",
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages([...messages, userMessage]);
    setInput("");

    setIsTyping(true);

    setTimeout(() => {
      const responses = [
        "Interesting! Let me analyze that and provide you with the best approach.",
        "Great suggestion! I'm processing your request now.",
        "I understand. Let me gather some insights on that topic.",
        "Perfect! I'll help you with that right away.",
      ];
      
      const botResponse = {
        id: messages.length + 2,
        text: responses[Math.floor(Math.random() * responses.length)],
        sender: "bot",
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };

      setMessages(prev => [...prev, botResponse]);
      setIsTyping(false);
    }, 1500);
  };

  const handleLogout = () => {
    // Clear user data
    localStorage.removeItem("currentUser");
    
    // If using a parent component to manage user state, update it
    if (setUser) {
      setUser(null);
    }
    
    // Navigate to home page
    navigate("/");
    
    // Force page reload if navigation doesn't work
    setTimeout(() => {
      if (window.location.pathname !== "/") {
        window.location.href = "/";
      }
    }, 100);
  };

  const handleNewChat = () => {
    setMessages([
      {
        id: 1,
        text: "Hello! I'm your AI assistant. What would you like to discuss today?",
        sender: "bot",
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      },
    ]);
  };

  const handleExportChat = () => {
    alert("Export functionality would save your chat history as a PDF, CSV, or JSON file.");
  };

  const handleClearHistory = () => {
    if (window.confirm("Are you sure you want to clear all chat history? This action cannot be undone.")) {
      alert("Chat history cleared.");
    }
  };

  const handleSettings = () => {
    alert("Settings panel would open with options for AI model, temperature, etc.");
  };

  return (
    <div className="dashboard-wrapper">
      {/* LEFT SIDEBAR - HISTORY ONLY */}
      <div className="sidebar-left">
        <div className="logo-section">
          <div className="app-logo">AI Workspace</div>
          <div className="user-profile-mini">
            <div className="user-avatar">
              {user?.fullName?.charAt(0) || user?.email?.charAt(0) || "U"}
            </div>
            <div className="user-details">
              <h4>{user?.fullName || user?.email || "User"}</h4>
            </div>
          </div>
        </div>

        <div className="history-section">
          <div className="history-search">
            <input type="text" placeholder="Search history..." />
          </div>
          
          <div className="history-groups">
            <div className="history-group">
              <h4 className="group-title">TODAY</h4>
              <div className="history-items">
                {chatHistory
                  .filter(item => item.dateGroup === "TODAY")
                  .map(item => (
                    <div key={item.id} className="history-item">
                      <div className="item-title">{item.title}</div>
                      <div className="item-time">{item.time}</div>
                    </div>
                  ))}
              </div>
            </div>

            <div className="history-group">
              <h4 className="group-title">YESTERDAY</h4>
              <div className="history-items">
                {chatHistory
                  .filter(item => item.dateGroup === "YESTERDAY")
                  .map(item => (
                    <div key={item.id} className="history-item">
                      <div className="item-title">{item.title}</div>
                      <div className="item-time">{item.time}</div>
                    </div>
                  ))}
              </div>
            </div>

            <div className="history-group">
              <h4 className="group-title">PREVIOUS 7 DAYS</h4>
              <div className="history-items">
                {chatHistory
                  .filter(item => item.dateGroup === "PREVIOUS 7 DAYS")
                  .map(item => (
                    <div key={item.id} className="history-item">
                      <div className="item-title">{item.title}</div>
                      <div className="item-time">{item.time}</div>
                    </div>
                  ))}
              </div>
            </div>
          </div>

          <button className="new-chat-btn" onClick={handleNewChat}>
            <span className="plus-icon">+</span> New Chat
          </button>
        </div>
      </div>

      {/* MAIN CHAT AREA - CENTER */}
      <div className="main-chat-area">
        <div className="chat-header-center">
          <div className="bot-avatar-center">
            <img 
              src="/bot.png" 
              alt="DUX Assistant" 
              className="bot-image"
              onError={(e) => {
                e.target.style.display = 'none';
                e.target.parentElement.textContent = 'AI';
              }}
            />
          </div>
          <div className="header-info-center">
            <h1>DUX</h1>
            <p className="status-text">Online and ready to help</p>
          </div>
        </div>

        <div className="messages-container-center">
          {messages.map((msg) => (
            <div 
              key={msg.id} 
              className={`message-bubble-center ${msg.sender}`}
              style={{ animationDelay: `${msg.id * 0.1}s` }}
            >
              <div className="message-header-center">
                <span className="sender-icon">
                  {msg.sender === "bot" ? "🤖" : "👤"}
                </span>
                <span className="sender-name">
                  {msg.sender === "bot" ? "DUX" : "You"} ({msg.time})
                </span>
              </div>
              <div className="message-content-center">{msg.text}</div>
            </div>
          ))}
          
          {isTyping && (
            <div className="typing-indicator-center">
              <div className="typing-bot">
                <div className="typing-dots">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
                <span className="typing-text">DUX is typing...</span>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Fixed Input Container */}
        <div className="chat-input-container">
          <form className="chat-input-center" onSubmit={handleSend}>
            <input
              type="text"
              placeholder="Type your message here..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              autoFocus
            />
            <button type="submit" className="send-btn-center">
              <span className="send-arrow">→</span>
            </button>
          </form>
        </div>
      </div>

      {/* RIGHT SIDEBAR - QUICK TOOLS */}
      <div className="sidebar-right">
        <div className="tools-section">
          <h2>QUICK TOOLS</h2>
          
          <div className="tools-grid">
            <div className="tool-card" onClick={handleExportChat}>
              <div className="tool-icon">📥</div>
              <div className="tool-info">
                <h4>Export Chat</h4>
                <p>PDF, CSV, JSON formats</p>
              </div>
            </div>

            <div className="tool-card" onClick={handleSettings}>
              <div className="tool-icon">⚙️</div>
              <div className="tool-info">
                <h4>Settings</h4>
                <p>Customize AI behavior</p>
              </div>
            </div>

            <div className="tool-card" onClick={handleClearHistory}>
              <div className="tool-icon">🗑️</div>
              <div className="tool-info">
                <h4>Clear History</h4>
                <p>Remove all conversations</p>
              </div>
            </div>
          </div>

          <div className="logout-section">
            <button 
              type="button" 
              className="logout-btn-center" 
              onClick={handleLogout}
            >
              <span className="logout-icon">↩</span>
              Logout
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;