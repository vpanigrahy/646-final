import React, { useEffect, useState } from "react";
import "./chatbox.css";

function ChatBox({ messages }) {
  const [typedMessages, setTypedMessages] = useState([]);

  useEffect(() => {
    const interval = setInterval(() => {
      if (messages.length > typedMessages.length) {
        const nextMessage = messages[typedMessages.length];
        if (nextMessage.text.trim() !== "") {
          setTypedMessages((prevMessages) => [...prevMessages, nextMessage]);
        }
      } else {
        clearInterval(interval);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [messages, typedMessages]);

  return (
    <div className="chatBox">
      <div className="chatBox__header">
        <h3>Retrieval Augmented Generation</h3>
      </div>
      <div className="chatBox__messages">
        {typedMessages.map((message, index) => (
          <div key={index} className={`chatBox__message ${message.type}`}>
            {message.text}
          </div>
        ))}
      </div>
    </div>
  );
}

export default ChatBox;