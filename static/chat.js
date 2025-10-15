// chat.js - Client-side chat functionality for the emotion analysis application

// Initialize chat functionality
function initChat() {
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const messagesContainer = document.getElementById('chat-messages');
    
    if (!chatForm || !chatInput || !messagesContainer) return;
    
    chatForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const userMessage = chatInput.value.trim();
        if (!userMessage) return;
        
        // Clear input
        chatInput.value = '';
        
        // Add user message to chat
        addMessage(userMessage, 'user');
        
        // Show typing indicator
        const loadingMessage = addLoadingMessage();
        
        try {
            // Send to server
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 
                    message: userMessage,
                    analysis_id: window.analysisId // We'll set this in the template
                })
            });
            
            // Remove loading indicator
            loadingMessage.remove();
            
            if (response.ok) {
                const data = await response.json();
                // Add bot response
                addMessage(data.response, 'bot');
            } else {
                // Add error message
                addMessage('Sorry, I encountered an error while processing your message. Please try again.', 'system');
            }
        } catch (error) {
            // Remove loading indicator
            loadingMessage.remove();
            
            console.error('Chat error:', error);
            // Add error message
            addMessage('Sorry, I encountered an error while processing your message. Please try again.', 'system');
        }
        
        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    });
}

// Add a message to the chat
function addMessage(content, type) {
    const messagesContainer = document.getElementById('chat-messages');
    if (!messagesContainer) return null;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    const now = new Date();
    const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    const contentHtml = formatMessageContent(content);
    
    messageDiv.innerHTML = `
        <div class="message-content">${contentHtml}</div>
        <span class="chat-time">${timeStr}</span>
    `;
    
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    return messageDiv;
}

// Add a loading message while waiting for response
function addLoadingMessage() {
    const messagesContainer = document.getElementById('chat-messages');
    if (!messagesContainer) return null;
    
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message bot-message';
    loadingDiv.innerHTML = `
        <div class="message-content">
            <div class="loading-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    
    messagesContainer.appendChild(loadingDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    return loadingDiv;
}

// Format message content with paragraphs
function formatMessageContent(content) {
    // Split by line breaks and wrap in paragraphs
    return content.split('\n')
        .filter(line => line.trim() !== '')
        .map(line => `<p>${line}</p>`)
        .join('');
}

// Initialize chat when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initChat();
});