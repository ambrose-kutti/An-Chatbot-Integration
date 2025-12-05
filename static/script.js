document.addEventListener('DOMContentLoaded', function() {
    const form = document.createElement('form');
    const input = document.getElementById('user-input');
    const messagesDiv = document.getElementById('messages');
    const typingDiv = document.getElementById('typing');
    const sendBtn = document.getElementById('send-btn');
    
    // Add message to chat
    function addMessage(text, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
        
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        bubble.textContent = text;
        
        const time = document.createElement('div');
        time.className = 'message-time';
        time.textContent = getCurrentTime();
        
        messageDiv.appendChild(bubble);
        messageDiv.appendChild(time);
        messagesDiv.appendChild(messageDiv);
        
        // Scroll to bottom
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
        
        return messageDiv;
    }
    
    // Get current time in HH:MM format
    function getCurrentTime() {
        return new Date().toLocaleTimeString([], { 
            hour: '2-digit', 
            minute: '2-digit' 
        });
    }
    
    // Show/hide typing indicator
    function showTyping(show) {
        typingDiv.style.display = show ? 'flex' : 'none';
        if (show) {
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
    }
    
    // Send message to backend
    async function sendMessage(message) {
        showTyping(true);
        sendBtn.disabled = true;
        input.disabled = true;
        
        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ question: message })
            });
            
            const data = await response.json();
            showTyping(false);
            
            if (data.status === 'success') {
                let messageText = data.answer;
                
                addMessage(messageText, false);
            }
            
        } catch (error) {
            console.error('Error:', error);
            showTyping(false);
            addMessage('Sorry, there was an error. Please try again.', false);
        } finally {
            sendBtn.disabled = false;
            input.disabled = false;
            input.focus();
        }
    }
    
    // Handle message sending
    function handleSendMessage(e) {
        if (e) e.preventDefault();
        
        const message = input.value.trim();
        if (!message) return;
        
        // Add user message
        addMessage(message, true);
        
        // Clear input
        input.value = '';
        
        // Send to backend
        sendMessage(message);
    }
    
    // Set up event listeners
    sendBtn.addEventListener('click', handleSendMessage);
    
    input.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    });
    
    // Auto-focus input when widget opens
    document.getElementById('widget-header').addEventListener('click', function() {
        setTimeout(() => {
            input.focus();
        }, 100);
    });
});