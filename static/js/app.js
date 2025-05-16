// Enhanced app.js with better formatting for code execution and citations

document.addEventListener("DOMContentLoaded", function () {
    const chatMessages = document.getElementById("chat-messages");
    const chatInput = document.getElementById("chat-input");
    const sendBtn = document.getElementById("send-btn");

    // Generate a random session ID
    const sessionId = Math.random().toString(36).substring(2, 15);

    // WebSocket setup
    let socket;
    let isConnected = false;

    function initWebSocket() {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        socket = new WebSocket(
            `${protocol}//${window.location.host}/api/chat/ws/${sessionId}`,
        );

        socket.onopen = function (e) {
            console.log("WebSocket connection established");
            isConnected = true;
        };

        socket.onmessage = function (event) {
            const data = JSON.parse(event.data);

            if (data.type === "message_received") {
                // User message already added, do nothing
            } else if (data.type === "partial_response") {
                // Filter out raw transition messages
                if (shouldFilterMessage(data.message.content)) {
                    // Don't display these raw transition messages
                    return;
                }

                // Update or create assistant's message
                updateAssistantMessage(data.message.content);
            } else if (data.type === "message_complete") {
                // Filter out raw transition messages on final message too
                if (shouldFilterMessage(data.message.content)) {
                    return;
                }

                // Finalize assistant's message
                updateAssistantMessage(data.message.content, true);

                // Remove typing indicator if present
                const typingIndicator = document.querySelector(".typing-indicator");
                if (typingIndicator) {
                    typingIndicator.remove();
                }
            } else if (data.type === "error") {
                // Handle error messages
                showErrorMessage(data.message.content);
            }
        };

        socket.onclose = function (event) {
            console.log("WebSocket connection closed");
            isConnected = false;

            // Try to reconnect after 2 seconds
            setTimeout(initWebSocket, 2000);
        };

        socket.onerror = function (error) {
            console.error("WebSocket error:", error);
            socket.close();
        };
    }

    // Initialize WebSocket connection
    initWebSocket();

    // Function to determine if a message should be filtered
    function shouldFilterMessage(content) {
        if (!content) return false;

        const trimmed = content.trim();

        // Agent names (exact matches)
        if (/^supervisor$/i.test(trimmed) || 
            /^wiki_agent$/i.test(trimmed) || 
            /^search_agent$/i.test(trimmed) || 
            /^code_agent$/i.test(trimmed)) {
            return true;
        }

        // Any kind of transfer message
        if (/transferred.*supervisor/i.test(trimmed) || 
            /transferred.*wiki_agent/i.test(trimmed) || 
            /transferred.*search_agent/i.test(trimmed) || 
            /transferred.*code_agent/i.test(trimmed)) {
            return true;
        }

        return false;
    }

    // Send a message via the WebSocket
    function sendMessage() {
        const message = chatInput.value.trim();

        if (message === "") return;

        // Add user message to the chat
        addMessage("user", message);

        // Clear input
        chatInput.value = "";
        resetInputHeight();

        // Add typing indicator
        addTypingIndicator();

        // Send message via WebSocket if connected
        if (isConnected) {
            socket.send(
                JSON.stringify({
                    message: message,
                }),
            );
        } else {
            // Fallback to REST API if WebSocket not connected
            fetch("/api/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    session_id: sessionId,
                    message: message,
                }),
            })
                .then((response) => response.json())
                .then((data) => {
                    // Remove typing indicator
                    const typingIndicator =
                        document.querySelector(".typing-indicator");
                    if (typingIndicator) {
                        typingIndicator.remove();
                    }

                    // Add assistant message
                    const messages = data.messages;
                    const assistantMessage = messages[messages.length - 1];
                    if (assistantMessage.role === "assistant") {
                        addMessage("assistant", assistantMessage.content);
                    }
                })
                .catch((error) => {
                    console.error("Error:", error);
                    // Remove typing indicator
                    const typingIndicator =
                        document.querySelector(".typing-indicator");
                    if (typingIndicator) {
                        typingIndicator.remove();
                    }

                    // Add error message
                    showErrorMessage(
                        "Sorry, I encountered an error. Please try again.",
                    );
                });
        }
    }

    // Show error message
    function showErrorMessage(message) {
        const errorDiv = document.createElement("div");
        errorDiv.className = "message assistant error";

        const messageContent = document.createElement("div");
        messageContent.className = "message-content";
        messageContent.innerHTML = `<p>${message}</p>`;

        errorDiv.appendChild(messageContent);
        chatMessages.appendChild(errorDiv);

        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Add a message to the chat
    function addMessage(role, content) {
        const messageDiv = document.createElement("div");
        messageDiv.className = `message ${role}`;

        const messageContent = document.createElement("div");
        messageContent.className = "message-content";

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
        const typingIndicator = document.querySelector(".typing-indicator");
        if (typingIndicator) {
            typingIndicator.remove();
        }

        let assistantMessage = chatMessages.querySelector(
            ".message.assistant:last-child",
        );

        // If the last message is a user message or there's no assistant message, create a new one
        if (
            !assistantMessage ||
            assistantMessage.classList.contains("typing-indicator") ||
            (assistantMessage.nextElementSibling &&
                assistantMessage.nextElementSibling.classList.contains("user"))
        ) {
            assistantMessage = addMessage("assistant", content);
        } else {
            // Update existing message
            const messageContent =
                assistantMessage.querySelector(".message-content");
            messageContent.innerHTML = formatMessage(content);

            // Scroll to bottom
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        if (isFinal) {
            // Additional styling for final message if needed
            assistantMessage.classList.add("final");

            // Highlight any code blocks in the final message
            highlightCodeBlocks(assistantMessage);
        }
    }

    // Add typing indicator
    function addTypingIndicator() {
        const typingDiv = document.createElement("div");
        typingDiv.className = "message assistant typing-indicator";
        typingDiv.innerHTML = "<span></span><span></span><span></span>";
        chatMessages.appendChild(typingDiv);

        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Highlight code blocks using highlight.js (if available)
    function highlightCodeBlocks(messageElement) {
        if (window.hljs) {
            const codeBlocks = messageElement.querySelectorAll("pre code");
            codeBlocks.forEach((block) => {
                hljs.highlightElement(block);
            });
        }
    }

    // Format message content with enhanced markdown-like processing and citation handling
    function formatMessage(content) {
        // Check for code execution results and add special formatting
        const isCodeExecution =
            content.includes("**Code Execution Result:**") ||
            content.includes("```python") ||
            content.includes("Code execution");

        // Replace newlines with <br>
        let formatted = content.replace(/\n/g, "<br>");

        // Handle code execution results with special formatting
        if (isCodeExecution) {
            // Look for execution results pattern
            const codeResultPattern =
                /\*\*Code Execution Result:\*\*<br><br>([\s\S]*?)(?:<br><br>|$)/;
            const match = formatted.match(codeResultPattern);

            if (match) {
                formatted = formatted.replace(
                    match[0],
                    `<div class='code-execution-result'><strong>Code Execution Result:</strong><br><br>${match[1]}</div>`,
                );
            }

            // Also look for error patterns
            const errorPattern =
                /Code execution (failed|error)([\s\S]*?)(?:<br><br>|$)/i;
            const errorMatch = formatted.match(errorPattern);

            if (errorMatch) {
                formatted = formatted.replace(
                    errorMatch[0],
                    `<div class='code-execution-error'><strong>Code Execution Error:</strong><br>${errorMatch[2]}</div>`,
                );
            }
        }

        // Simple markdown for code blocks with language detection
        formatted = formatted.replace(
            /```(\w*)([\s\S]*?)```/g,
            function (match, language, code) {
                return `<pre><code class="language-${language || "plaintext"}">${code.trim()}</code></pre>`;
            },
        );

        // Inline code
        formatted = formatted.replace(/`([^`]+)`/g, "<code>$1</code>");

        // Bold - needs to run after code blocks to avoid conflicts
        formatted = formatted.replace(
            /\*\*([^\*]+)\*\*/g,
            "<strong>$1</strong>",
        );

        // Italics - needs to run after bold
        formatted = formatted.replace(/\*([^\*]+)\*/g, "<em>$1</em>");

        // Convert URLs to clickable links
        formatted = formatted.replace(
            /(https?:\/\/[^\s<]+)/g,
            '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>',
        );

        // Enhanced citation handling
        formatted = formatted.replace(
            /<cite index="([^"]+)">([^<]+)<\/cite>/g,
            '<span class="citation" data-cite-index="$1">$2<sup class="citation-marker">[$1]</sup></span>',
        );

        // Headers (h1-h6)
        formatted = formatted.replace(
            /^(#{1,6})\s+(.+?)$/gm,
            function (match, hashes, content) {
                const level = hashes.length;
                return `<h${level}>${content}</h${level}>`;
            },
        );

        // Extract and format sources if they exist
        const sourcesMatch = formatted.match(/Sources:<br>([\s\S]+)/);
        if (sourcesMatch) {
            const sourcesList = sourcesMatch[1]
                .split("<br>")
                .filter((s) => s.trim().length > 0);
            const sourcesHtml = sourcesList
                .map((source, index) => {
                    // Convert source URLs to clickable links if not already
                    const linkedSource = source.replace(
                        /\[(\d+)\]\s*(https?:\/\/[^\s<]+)/,
                        '[$1] <a href="$2" target="_blank" rel="noopener noreferrer">$2</a>',
                    );
                    return `<li class="source-item">${linkedSource}</li>`;
                })
                .join("");

            formatted = formatted.replace(
                /Sources:<br>([\s\S]+)/,
                `<div class="sources-section">
                    <h4>Sources:</h4>
                    <ol class="sources-list">${sourcesHtml}</ol>
                </div>`,
            );
        }

        return formatted;
    }

    // Reset input height
    function resetInputHeight() {
        chatInput.style.height = "";
    }

    // Auto-resize the textarea as the user types
    function resizeTextarea() {
        resetInputHeight();
        const maxHeight = 150; // Maximum height in pixels
        if (chatInput.scrollHeight <= maxHeight) {
            chatInput.style.height = chatInput.scrollHeight + "px";
        } else {
            chatInput.style.height = maxHeight + "px";
        }
    }

    // Event listeners
    sendBtn.addEventListener("click", sendMessage);

    chatInput.addEventListener("input", resizeTextarea);

    chatInput.addEventListener("keydown", function (e) {
        // Send message on Enter (without Shift)
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Focus input on load
    chatInput.focus();

    // Add highlight.js if not already present
    if (!window.hljs) {
        const highlightCSS = document.createElement("link");
        highlightCSS.rel = "stylesheet";
        highlightCSS.href =
            "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/atom-one-dark.min.css";
        document.head.appendChild(highlightCSS);

        const highlightJS = document.createElement("script");
        highlightJS.src =
            "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js";
        document.head.appendChild(highlightJS);

        highlightJS.onload = function () {
            const assistantMessages = document.querySelectorAll(
                ".message.assistant.final",
            );
            assistantMessages.forEach((message) => {
                highlightCodeBlocks(message);
            });
        };
    }
});
