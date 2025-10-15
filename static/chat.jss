/* chat.css - Styles for the chat feature */

.chat-container {
    display: flex;
    flex-direction: column;
    height: 600px;
    border-radius: 12px;
    background-color: #fff;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
    overflow: hidden;
}

.messages-container {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
}

.message {
    max-width: 80%;
    padding: 12px 16px;
    border-radius: 12px;
    animation: fadeIn 0.3s ease-out forwards;
}

.user-message {
    align-self: flex-end;
    background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
    color: white;
    border-bottom-right-radius: 0;
}

.bot-message {
    align-self: flex-start;
    background-color: #f0f2f5;
    color: #2c3e50;
    border-bottom-left-radius: 0;
}

.system-message {
    align-self: center;
    background-color: #f8f9fa;
    color: #6c757d;
    border: 1px solid #e9ecef;
    border-radius: 8px;
    width: 90%;
}

.message-content p {
    margin-bottom: 0.5rem;
}

.message-content p:last-child {
    margin-bottom: 0;
}

.chat-input-container {
    padding: 15px;
    background-color: #f8f9fa;
    border-top: 1px solid #e9ecef;
}

.chat-input-container .form-control {
    border-radius: 20px;
    padding: 12px 20px;
}

.chat-input-container .btn {
    border-radius: 20px;
    padding: 8px 16px;
    margin-left: 8px;
}

.chat-time {
    font-size: 0.7rem;
    margin-top: 5px;
    opacity: 0.7;
    display: block;
}

.loading-dots {
    display: inline-flex;
    align-items: center;
    gap: 4px;
}

.loading-dots span {
    height: 8px;
    width: 8px;
    border-radius: 50%;
    background-color: currentColor;
    opacity: 0.5;
    animation: pulse 1.4s ease-in-out infinite;
}

.loading-dots span:nth-child(2) {
    animation-delay: 0.2s;
}

.loading-dots span:nth-child(3) {
    animation-delay: 0.4s;
}

@keyframes pulse {
    0%, 100% { opacity: 0.3; transform: scale(0.8); }
    50% { opacity: 0.8; transform: scale(1.2); }
}