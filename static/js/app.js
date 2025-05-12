document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chat-messages');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    
    // Generate a random session ID
    const sessionId = Math.random().toString(36).substring(2, 15);
    
    // WebSocket setup
    let socket;
    let isConnected = false;
    
    function initWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        socket = new WebSocket(`${protocol}//${window.location.host}/api/chat/ws/${sessionId}`);
        
        socket.onopen = function(e) {
            console.log('WebSocket connection established');
            isConnected = true;
        };
        
        socket.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                
                if (data.type === 'message_received') {
                    // User message already added, do nothing
                } else if (data.type === 'partial_response') {
                    // Transform agent transfer messages before displaying
                    let content = transformAgentMessages(data.message.content);
                    
                    // Only update if there's actual content to show
                    if (content) {
                        updateAssistantMessage(content);
                    }
                } else if (data.type === 'message_complete') {
                    // Transform agent transfer messages before finalizing
                    let content = transformAgentMessages(data.message.content);
                    
                    // Only update if there's actual content to show
                    if (content) {
                        // Finalize assistant's message
                        updateAssistantMessage(content, true);
                    }
                    
                    // Remove typing indicator if present
                    const typingIndicator = document.querySelector('.typing-indicator');
                    if (typingIndicator) {
                        typingIndicator.remove();
                    }
                    
                    // Also remove any tool usage messages when the response is complete
                    const toolMessages = document.querySelectorAll('.tool-usage-message');
                    toolMessages.forEach(msg => msg.remove());
                }
            } catch (error) {
                console.error('Error processing WebSocket message:', error);
            }
        };
        
        socket.onclose = function(event) {
            console.log('WebSocket connection closed');
            isConnected = false;
            
            // Try to reconnect after 2 seconds
            setTimeout(initWebSocket, 2000);
        };
        
        socket.onerror = function(error) {
            console.error('WebSocket error:', error);
            socket.close();
        };
    }
    
    // Function to display a tool usage indicator
    function displayToolUsage(toolType) {
        // Remove any existing tool usage messages
        const existingToolMessages = document.querySelectorAll('.tool-usage-message');
        existingToolMessages.forEach(msg => msg.remove());
        
        // Create new tool usage message
        const toolMessage = document.createElement('div');
        toolMessage.className = `tool-usage-message ${toolType}`;
        
        // Set message text based on tool type
        switch(toolType) {
            case 'search':
                toolMessage.textContent = 'Using Web Search Tool...';
                break;
            case 'wiki':
                toolMessage.textContent = 'Using Wikipedia Tool...';
                break;
            case 'code':
                toolMessage.textContent = 'Using Code Execution Tool...';
                break;
            case 'thinking':
                toolMessage.textContent = 'Thinking...';
                break;
            default:
                toolMessage.textContent = 'Processing...';
        }
        
        // Add to chat messages
        chatMessages.appendChild(toolMessage);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        // Return the created element so it can be removed later
        return toolMessage;
    }
    
    // Function to transform agent transfer messages into user-friendly messages
    function transformAgentMessages(content) {
        if (!content) return content;
        
        // Check for agent transfer messages and display appropriate tool usage indicators
        if (content.includes('transferred to search_agent') || content === 'search_agent' || 
            content === 'Using Web Search Tool...') {
            displayToolUsage('search');
            // Return empty string to prevent showing the technical message
            return '';
        }
        
        if (content.includes('transferred to wiki_agent') || content === 'wiki_agent' || 
            content === 'Using Wikipedia Tool...') {
            displayToolUsage('wiki');
            return '';
        }
        
        if (content.includes('transferred to code_agent') || content === 'code_agent' || 
            content === 'Using Code Execution Tool...') {
            displayToolUsage('code');
            return '';
        }
        
        if (content.includes('transferred back to supervisor') || content === 'supervisor' || 
            content === 'Thinking...') {
            displayToolUsage('thinking');
            return '';
        }
        
        // If content contains a real message (not just a transfer notification)
        if (content.length > 30 || 
            content.includes('I found') || 
            content.includes('Here is') || 
            content.includes('According to')) {
            // Remove any tool usage indicators when real content appears
            const toolMessages = document.querySelectorAll('.tool-usage-message');
            toolMessages.forEach(msg => msg.remove());
            
            // Keep the actual content
            return content;
        }
        
        // For other transfer messages or short notifications, hide them
        if (content.length < 30) {
            return '';
        }
        
        return content;
    }
    
    // Initialize WebSocket connection
    initWebSocket();
    
    // Send a message via the WebSocket
    function sendMessage() {
        const message = chatInput.value.trim();
        
        if (message === '') return;
        
        // Add user message to the chat
        addMessage('user', message);
        
        // Clear input
        chatInput.value = '';
        resetInputHeight();
        
        // Add typing indicator
        addTypingIndicator();
        
        // Send message via WebSocket if connected
        if (isConnected) {
            try {
                socket.send(JSON.stringify({
                    message: message
                }));
            } catch (error) {
                console.error('Error sending message via WebSocket:', error);
                // Fallback to REST API if WebSocket fails
                sendViaRestApi(message);
            }
        } else {
            // Fallback to REST API if WebSocket not connected
            sendViaRestApi(message);
        }
    }
    
    // Send message via REST API (fallback method)
    function sendViaRestApi(message) {
        fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: sessionId,
                message: message
            }),
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Remove typing indicator
            const typingIndicator = document.querySelector('.typing-indicator');
            if (typingIndicator) {
                typingIndicator.remove();
            }
            
            // Add assistant message
            const messages = data.messages;
            if (messages && messages.length > 0) {
                const assistantMessage = messages[messages.length - 1];
                if (assistantMessage && assistantMessage.role === 'assistant') {
                    // Transform agent messages before displaying
                    let content = transformAgentMessages(assistantMessage.content);
                    if (content) {
                        addMessage('assistant', content);
                    }
                }
            }
        })
        .catch(error => {
            console.error('Error:', error);
            // Remove typing indicator
            const typingIndicator = document.querySelector('.typing-indicator');
            if (typingIndicator) {
                typingIndicator.remove();
            }
            
            // Add error message
            addMessage('assistant', 'Sorry, I encountered an error. Please try again.');
        });
    }
    
    // Add a message to the chat
    function addMessage(role, content) {
        if (!content) return null;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        // Process content to handle markdown-like formatting
        const formattedContent = formatMessage(content);
        messageContent.innerHTML = formattedContent;
        
        messageDiv.appendChild(messageContent);
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        return messageDiv;
    }
    
    // Update an existing assistant message or create a new one
    function updateAssistantMessage(content, isFinal = false) {
        if (!content) return;
        
        // Remove typing indicator if it exists
        const typingIndicator = document.querySelector('.typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
        
        let assistantMessage = chatMessages.querySelector('.message.assistant:last-child');
        
        // Fix the conditional statement with proper parentheses for clarity
        if (!assistantMessage || 
            assistantMessage.classList.contains('typing-indicator') || 
            (assistantMessage.nextElementSibling && assistantMessage.nextElementSibling.classList.contains('user'))) {
            assistantMessage = addMessage('assistant', content);
        } else {
            // Update existing message
            const messageContent = assistantMessage.querySelector('.message-content');
            if (messageContent) {
                messageContent.innerHTML = formatMessage(content);
                
                // Scroll to bottom
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        }
        
        if (isFinal) {
            // Additional styling for final message if needed
        }
    }
    
    // Add typing indicator
    function addTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message assistant typing-indicator';
        typingDiv.innerHTML = '<span></span><span></span><span></span>';
        chatMessages.appendChild(typingDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Format message content with enhanced markdown-like processing and citation handling
    function formatMessage(content) {
        if (!content) return '';
        
        // Replace newlines with <br>
        let formatted = content.replace(/\n/g, '<br>');
        
        // Simple markdown for code blocks
        formatted = formatted.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
        
        // Inline code
        formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // Italics
        formatted = formatted.replace(/\*([^\*]+)\*/g, '<em>$1</em>');
        
        // Bold
        formatted = formatted.replace(/\*\*([^\*]+)\*\*/g, '<strong>$1</strong>');
        
        // Convert URLs to clickable links
        formatted = formatted.replace(
            /(https?:\/\/[^\s]+)/g, 
            '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>'
        );
        
        // Handle citations
        formatted = formatted.replace(/<cite index="([^"]+)">([^<]+)<\/cite>/g, 
            '<span class="citation" data-cite-index="$1">$2<sup class="citation-marker">[$1]</sup></span>');
        
        // Extract and format sources if they exist
        const sourcesMatch = formatted.match(/Sources:\s*\n?([\s\S]+)/);
        if (sourcesMatch) {
            const sourcesList = sourcesMatch[1].split('\n').filter(s => s.trim().length > 0);
            const sourcesHtml = sourcesList.map((source, index) => 
                `<li class="source-item">${source}</li>`).join('');
            
            formatted = formatted.replace(/Sources:\s*\n?([\s\S]+)/, 
                `<div class="sources-section">
                    <h4>Sources:</h4>
                    <ol class="sources-list">${sourcesHtml}</ol>
                </div>`);
        }
        
        return formatted;
    }
    
    // Reset input height
    function resetInputHeight() {
        chatInput.style.height = '';
    }
    
    // Auto-resize the textarea as the user types
    function resizeTextarea() {
        resetInputHeight();
        const maxHeight = 150; // Maximum height in pixels (increased from 100)
        if (chatInput.scrollHeight <= maxHeight) {
            chatInput.style.height = chatInput.scrollHeight + 'px';
        } else {
            chatInput.style.height = maxHeight + 'px';
        }
    }
    
    // Event listeners
    sendBtn.addEventListener('click', sendMessage);
    
    chatInput.addEventListener('input', resizeTextarea);
    
    chatInput.addEventListener('keydown', function(e) {
        // Send message on Enter (without Shift)
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Focus input on load
    chatInput.focus();
});