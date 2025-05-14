document.addEventListener("DOMContentLoaded", function () {
    const chatMessages = document.getElementById("chat-messages");
    const chatInput = document.getElementById("chat-input");
    const sendBtn = document.getElementById("send-btn");

    // Generate a random session ID
    const sessionId = Math.random().toString(36).substring(2, 15);

    // Configure Marked.js with custom renderer
    const renderer = new marked.Renderer();

    // Customize heading renderer to ensure proper heading output
    renderer.heading = function (text, level) {
        return `<h${level}>${text}</h${level}>`;
    };

    marked.setOptions({
        renderer: renderer,
        headerIds: false,
        gfm: true,
        breaks: true,
        pedantic: false,
        sanitize: false,
        smartLists: true,
        smartypants: false,
        xhtml: false,
    });

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
                // Update or create assistant's message
                updateAssistantMessage(data.message.content);
            } else if (data.type === "message_complete") {
                // Finalize assistant's message
                updateAssistantMessage(data.message.content, true);

                // Remove typing indicator if present
                const typingIndicator =
                    document.querySelector(".typing-indicator");
                if (typingIndicator) {
                    typingIndicator.remove();
                }
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
                    addMessage(
                        "assistant",
                        "Sorry, I encountered an error. Please try again.",
                    );
                });
        }
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

    // Format message content with enhanced markdown support and citation handling
    function formatMessage(content) {
        if (!content) {
            return "";
        }

        // Ensure content is a string
        content = String(content);

        // Check if we should use special handling for code execution results
        const isCodeExecution = content.includes("**Code Execution Result:**");

        // Extract citations before markdown parsing to ensure they aren't affected
        const citations = [];
        let citationIndex = 0;

        // Replace citation tags with placeholders for preservation
        content = content.replace(
            /<cite index="([^"]+)">([^<]+)<\/cite>/g,
            function (match, index, text) {
                const placeholder = `CITATION_PLACEHOLDER_${citationIndex}`;
                citations.push({
                    placeholder: placeholder,
                    index: index,
                    text: text,
                });
                citationIndex++;
                return placeholder;
            },
        );

        // Extract source section if it exists
        let sourcesSection = "";
        const sourcesMatch = content.match(
            /Sources:\s*(\n|<br>|<br\/>)([\s\S]+)/,
        );
        if (sourcesMatch) {
            sourcesSection = sourcesMatch[0];
            content = content.replace(sourcesSection, "SOURCES_PLACEHOLDER");
        }

        // Extract code execution result block if it exists
        let codeExecutionBlock = "";
        if (isCodeExecution) {
            const codeExecMatch = content.match(
                /\*\*Code Execution Result:\*\*\s*(\n|<br>|<br\/>)([\s\S]+?)(?=(\n\n|\n<br>|<br><br>|$))/,
            );
            if (codeExecMatch) {
                codeExecutionBlock = codeExecMatch[0];
                content = content.replace(
                    codeExecutionBlock,
                    "CODE_EXECUTION_PLACEHOLDER",
                );
            }
        }

        // Normalize line breaks
        content = content.replace(/<br\s*\/?>/gi, "\n");

        // Ensure headings have proper spaces before and after
        content = content.replace(/^(#+)([^\n#])/gm, "$1 $2");
        content = content.replace(/(\n)(#+)([^\n#])/gm, "$1$2 $3");

        // Parse the content with Marked.js
        let formatted = marked.parse(content);

        // Replace placeholders with original content
        citations.forEach((citation) => {
            formatted = formatted.replace(
                citation.placeholder,
                `<span class="citation" data-cite-index="${citation.index}">${citation.text}<sup class="citation-marker">[${citation.index}]</sup></span>`,
            );
        });

        // Handle special blocks that shouldn't be fully parsed by Markdown
        if (sourcesSection) {
            const sourcesList = sourcesSection
                .split(/\n|<br\s*\/?>/)
                .slice(1)
                .filter((s) => s.trim().length > 0);
            const sourcesHtml = sourcesList
                .map(
                    (source, index) => `<li class="source-item">${source}</li>`,
                )
                .join("");

            const formattedSourcesSection = `
                <div class="sources-section">
                    <h4>Sources:</h4>
                    <ol class="sources-list">${sourcesHtml}</ol>
                </div>`;

            formatted = formatted.replace(
                "SOURCES_PLACEHOLDER",
                formattedSourcesSection,
            );
        }

        // Handle code execution result block
        if (codeExecutionBlock && isCodeExecution) {
            const formattedCodeExec = `
                <div class='code-execution-result'>
                    <strong>Code Execution Result:</strong><br><br>
                    ${marked.parse(codeExecutionBlock.replace("**Code Execution Result:**", "").trim())}
                </div>`;

            formatted = formatted.replace(
                "CODE_EXECUTION_PLACEHOLDER",
                formattedCodeExec,
            );
        }

        // Convert URLs to clickable links (if not already done by Markdown)
        formatted = formatted.replace(
            /(?<![="])(https?:\/\/[^\s<]+)(?![^<]*>)/g,
            '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>',
        );

        return formatted;
    }

    // Reset input height
    function resetInputHeight() {
        chatInput.style.height = "";
    }

    // Auto-resize the textarea as the user types
    function resizeTextarea() {
        resetInputHeight();
        const maxHeight = 150; // Maximum height in pixels (increased from 100)
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
});
