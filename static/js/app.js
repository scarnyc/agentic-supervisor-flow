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
            const data = JSON.parse(event.data);

            if (data.type === 'message_received') {
                // User message already added, do nothing
            } else if (data.type === 'partial_response') {
                // Update or create assistant's message
                updateAssistantMessage(data.message.content);
            } else if (data.type === 'message_complete') {
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
                    addMessage('assistant', assistantMessage.content);
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

        let assistantMessage = chatMessages.querySelector('.message.assistant:last-child');

        // If the last message is a user message or there's no assistant message, create a new one
        if (!assistantMessage || assistantMessage.classList.contains('typing-indicator') || 
            assistantMessage.nextElementSibling && assistantMessage.nextElementSibling.classList.contains('user')) {
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

    // Format message content with enhanced markdown support and citation handling
    function formatMessage(content) {
        if (!content) {
            return '';
        }

        // Ensure content is a string
        content = String(content);

        // Create a fresh copy of the content to work with
        let processedContent = content;

        // Arrays to store elements that should be preserved
        const preservedSections = [];

        // Function to replace a section with a placeholder and store the original
        function preserveSection(regex, content) {
            return content.replace(regex, function(match) {
                const placeholder = `PRESERVED_SECTION_${preservedSections.length}`;
                preservedSections.push({ placeholder, content: match });
                return placeholder;
            });
        }

        // Save code blocks to prevent processing their contents
        processedContent = preserveSection(/```[\s\S]*?```/g, processedContent);

        // Save inline code to prevent processing their contents
        processedContent = preserveSection(/`[^`]+`/g, processedContent);

        // Save citations
        processedContent = processedContent.replace(/<cite index="([^"]+)">([^<]+)<\/cite>/g, function(match, index, text) {
            const placeholder = `CITATION_SECTION_${preservedSections.length}`;
            preservedSections.push({
                placeholder, 
                content: `<span class="citation" data-cite-index="${index}">${text}<sup class="citation-marker">[${index}]</sup></span>`
            });
            return placeholder;
        });

        // Process special sections separately

        // Check for code execution results
        let codeExecSection = null;
        const codeExecMatch = processedContent.match(/\*\*Code Execution Result:\*\*[\s\S]*?(?=\n\n|$)/);
        if (codeExecMatch) {
            codeExecSection = codeExecMatch[0];
            processedContent = processedContent.replace(codeExecSection, 'CODE_EXEC_SECTION');
        }

        // Check for sources section
        let sourcesSection = null;
        const sourcesMatch = processedContent.match(/Sources:[\s\S]*?$/);
        if (sourcesMatch) {
            sourcesSection = sourcesMatch[0];
            processedContent = processedContent.replace(sourcesSection, 'SOURCES_SECTION');
        }

        // Direct handling of Markdown headings (h1 - h6)
        // This directly converts heading notation to HTML
        for (let i = 6; i >= 1; i--) {
            const headingRegex = new RegExp(`^${'#'.repeat(i)}\\s+(.+?)$`, 'gm');
            processedContent = processedContent.replace(headingRegex, function(match, title) {
                return `<h${i}>${title}</h${i}>`;
            });
        }

        // Handle lists - Unordered lists
        processedContent = processedContent.replace(/^(\s*)-\s+(.+?)$/gm, '<li>$2</li>');
        processedContent = processedContent.replace(/(<li>.*?<\/li>\n)+/g, function(match) {
            return `<ul>${match}</ul>`;
        });

        // Handle lists - Ordered lists
        processedContent = processedContent.replace(/^(\s*)\d+\.\s+(.+?)$/gm, '<li>$2</li>');
        processedContent = processedContent.replace(/(<li>.*?<\/li>\n)+/g, function(match) {
            return `<ol>${match}</ol>`;
        });

        // Handle paragraphs - wrap text blocks in <p> tags
        processedContent = processedContent.replace(/^(?!<h|<ul|<ol|<li|<blockquote|<pre)(.+?)$/gm, '<p>$1</p>');

        // Handle line breaks
        processedContent = processedContent.replace(/\n\n+/g, '<br><br>');
        processedContent = processedContent.replace(/\n/g, '<br>');

        // Handle bold text
        processedContent = processedContent.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

        // Handle italic text
        processedContent = processedContent.replace(/\*(.+?)\*/g, '<em>$1</em>');

        // Handle links
        processedContent = processedContent.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');

        // Additional URL detection (for URLs not in Markdown link format)
        processedContent = processedContent.replace(
            /(?<![="\(])(https?:\/\/[^\s<]+)(?![^<]*>)/g, 
            '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>'
        );

        // Now restore all preserved sections
        preservedSections.forEach(section => {
            processedContent = processedContent.replace(section.placeholder, section.content);
        });

        // Handle Code Execution Result
        if (codeExecSection) {
            const formattedCodeExec = `
                <div class='code-execution-result'>
                    <strong>Code Execution Result:</strong><br><br>
                    ${codeExecSection.replace('**Code Execution Result:**', '')}
                </div>`;

            processedContent = processedContent.replace('CODE_EXEC_SECTION', formattedCodeExec);
        }

        // Handle Sources Section
        if (sourcesSection) {
            const sourcesList = sourcesSection.replace('Sources:', '').split(/\n|<br\s*\/?>/).filter(s => s.trim().length > 0);
            const sourcesHtml = sourcesList.map((source, index) => 
                `<li class="source-item">${source}</li>`).join('');

            const formattedSourcesSection = `
                <div class="sources-section">
                    <h4>Sources:</h4>
                    <ol class="sources-list">${sourcesHtml}</ol>
                </div>`;

            processedContent = processedContent.replace('SOURCES_SECTION', formattedSourcesSection);
        }

        return processedContent;
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