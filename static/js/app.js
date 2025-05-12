document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chat-messages');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    
    // Generate a random session ID
    const sessionId = Math.random().toString(36).substring(2, 15);
    
    // WebSocket setup
    let socket;
    let isConnected = false;
    let activeToolUsage = null;
    
    function initWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        socket = new WebSocket(`${protocol}//${window.location.host}/api/chat/ws/${sessionId}`);
        
        socket.onopen = function(e) {
            console.log('WebSocket connection established');
            isConnected = true;
        };
        
        socket.onmessage = function(event) {
            const data = JSON.parse(event.data);
            
            if (data.type === 'message_received') {
                // User message already added, do nothing
            } else if (data.type === 'tool_usage') {
                // Handle tool usage indicator
                const tool = data.tool;
                handleToolUsageIndicator(tool);
                activeToolUsage = tool;
            } else if (data.type === 'partial_response') {
                // If we have an active tool usage and we're getting content, we can remove the tool indicator
                if (activeToolUsage) {
                    removeToolUsageIndicator();
                    activeToolUsage = null;
                }
                
                // Update or create assistant's message
                updateAssistantMessage(data.message.content);
            } else if (data.type === 'message_complete') {
                // If we still have an active tool usage, remove it
                if (activeToolUsage) {
                    removeToolUsageIndicator();
                    activeToolUsage = null;
                }
                
                // Finalize assistant's message
                updateAssistantMessage(data.message.content, true);
                
                // Remove typing indicator if present
                const typingIndicator = document.querySelector('.typing-indicator');
                if (typingIndicator) {
                    typingIndicator.remove();
                }
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
    
    // Initialize WebSocket connection
    initWebSocket();
    
    // Handle tool usage indicator
    function handleToolUsageIndicator(tool) {
        // Remove any existing tool indicators or typing indicators
        removeToolUsageIndicator();
        
        const toolDiv = document.createElement('div');
        toolDiv.className = `message assistant tool-indicator tool-${tool}`;
        
        let toolIcon, toolText;
        switch(tool) {
            case 'search':
                toolIcon = 'fa-search';
                toolText = 'Searching the web...';
                break;
            case 'code':
                toolIcon = 'fa-code';
                toolText = 'Setting up code environment...';
                break;
            case 'wiki':
                toolIcon = 'fa-book';
                toolText = 'Checking Wikipedia...';
                break;
            default:
                toolIcon = 'fa-cog';
                toolText = 'Processing your request...';
        }
        
        toolDiv.innerHTML = `
            <div class="tool-content">
                <i class="fas ${toolIcon} tool-icon"></i>
                <span class="tool-text">${toolText}</span>
                <span class="tool-dots"><span>.</span><span>.</span><span>.</span></span>
            </div>
        `;
        
        chatMessages.appendChild(toolDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Remove tool usage indicator
    function removeToolUsageIndicator() {
        const toolIndicators = document.querySelectorAll('.tool-indicator');
        toolIndicators.forEach(indicator => indicator.remove());
        
        const typingIndicator = document.querySelector('.typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
    
    // Transform agent transfer messages
    function transformAgentMessages(content) {
        // Replace agent transfer messages with user-friendly alternatives
        const agentTransformations = {
            /I'll transfer you to the search agent|Transferring to search agent/i: '🔍 Searching the web for information...',
            /I'll transfer you to the code agent|Transferring to code agent/i: '💻 Setting up code environment...',
            /I'll transfer you to the wiki agent|Transferring to wiki agent/i: '📚 Checking Wikipedia for information...',
            /Transferring to the supervisor|Returning to supervisor/i: '🤖 Processing your request...'
        };
        
        let transformedContent = content;
        for (const [pattern, replacement] of Object.entries(agentTransformations)) {
            const regex = new RegExp(pattern);
            if (regex.test(content)) {
                transformedContent = replacement;
                break;
            }
        }
        
        return transformedContent;
    }
    
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
            socket.send(JSON.stringify({
                message: message
            }));
        } else {
            // Fallback to REST API if WebSocket not connected
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
            .then(response => response.json())
            .then(data => {
                // Remove typing indicator
                const typingIndicator = document.querySelector('.typing-indicator');
                if (typingIndicator) {
                    typingIndicator.remove();
                }
                
                // Add assistant message
                const messages = data.messages;
                const assistantMessage = messages[messages.length - 1];
                if (assistantMessage.role === 'assistant') {
                    // Transform agent messages
                    const transformedContent = transformAgentMessages(assistantMessage.content);
                    addMessage('assistant', transformedContent);
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
    }
    
    // Add a message to the chat
    function addMessage(role, content) {
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
        // Remove typing indicator if it exists
        const typingIndicator = document.querySelector('.typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
        
        // Transform agent messages to user-friendly alternatives
        content = transformAgentMessages(content);
        
        let assistantMessage = chatMessages.querySelector('.message.assistant:last-child:not(.tool-indicator)');
        
        // If the last message is a user message or there's no assistant message, create a new one
        if (!assistantMessage || assistantMessage.classList.contains('typing-indicator') || 
            (assistantMessage.nextElementSibling && assistantMessage.nextElementSibling.classList.contains('user'))) {
            assistantMessage = addMessage('assistant', content);
        } else {
            // Update existing message
            const messageContent = assistantMessage.querySelector('.message-content');
            messageContent.innerHTML = formatMessage(content);
            
            // Scroll to bottom
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        if (isFinal) {
            // Additional styling for final message if needed
            assistantMessage.classList.add('final');
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
        // Replace newlines with <br>
        let formatted = content.replace(/\n/g, '<br>');
        
        // Simple markdown for code blocks with language detection
        formatted = formatted.replace(/```(\w*)\n([\s\S]*?)```/g, function(match, language, code) {
            if (language) {
                return `<pre><code class="language-${language}">${code}</code></pre>`;
            } else {
                return `<pre><code>${code}</code></pre>`;
            }
        });
        
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
            const sourcesHtml = sourcesList.map((source, index) => {
                // Convert URL patterns to clickable links if they aren't already
                source = source.replace(
                    /\[(\d+)\]\s+(https?:\/\/[^\s]+)/, 
                    '[$1] <a href="$2" target="_blank" rel="noopener noreferrer">$2</a>'
                );
                return `<li class="source-item">${source}</li>`;
            }).join('');
            
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