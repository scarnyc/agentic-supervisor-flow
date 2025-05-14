document.addEventListener("DOMContentLoaded", function () {
    const chatMessages = document.getElementById("chat-messages");
    const chatInput = document.getElementById("chat-input");
    const sendBtn = document.getElementById("send-btn");

    // Generate a random session ID
    const sessionId = Math.random().toString(36).substring(2, 15);

    // WebSocket setup
    let socket;
    let isConnected = false;

    // Configure marked.js options globally
    marked.setOptions({
        gfm: true, // Enable GitHub Flavored Markdown
        breaks: true, // Convert single newlines in paragraphs to <br>
        sanitize: false, // Disable HTML sanitization by marked.js.
        // IMPORTANT: If content is untrusted, use a dedicated HTML sanitizer
        // like DOMPurify after marked.parse() and restoring custom elements.
        headerIds: false, // Don't automatically generate IDs for headers
        // langPrefix: 'language-', // Uncomment and configure if using a syntax highlighter for code blocks
    });

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
            } else if (data.type === "error") {
                // Handle error messages from WebSocket
                console.error("WebSocket error message:", data.message.content);
                updateAssistantMessage(`Error: ${data.message.content}`, true);
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
            // Display an error message to the user in the chat
            addMessage(
                "assistant",
                "Sorry, there was a connection error. Please try sending your message again.",
            );
            const typingIndicator = document.querySelector(".typing-indicator");
            if (typingIndicator) {
                typingIndicator.remove();
            }
            // socket.close(); // Ensure socket is closed if not already
        };
    }

    // Initialize WebSocket connection
    initWebSocket();

    // Send a message via the WebSocket
    function sendMessage() {
        const message = chatInput.value.trim();
        if (message === "") return;

        addMessage("user", message);
        chatInput.value = "";
        resetInputHeight();
        addTypingIndicator();

        if (isConnected && socket.readyState === WebSocket.OPEN) {
            try {
                socket.send(JSON.stringify({ message: message }));
            } catch (error) {
                console.error("Failed to send message via WebSocket:", error);
                addMessage(
                    "assistant",
                    "Sorry, I could not send your message. Please check your connection and try again.",
                );
                const typingIndicator =
                    document.querySelector(".typing-indicator");
                if (typingIndicator) {
                    typingIndicator.remove();
                }
            }
        } else {
            console.warn(
                "WebSocket not connected or not open. Attempting to send again or use fallback.",
            );
            // Optionally, you could re-attempt initWebSocket or use a fallback HTTP request here
            // For simplicity, we'll just inform the user if the message can't be sent.
            if (!isConnected) {
                addMessage(
                    "assistant",
                    "Connection issue. Trying to reconnect. Please wait a moment and try sending again.",
                );
                initWebSocket(); // Attempt to re-establish connection
            } else {
                addMessage(
                    "assistant",
                    "Could not send message. Please try again.",
                );
            }
            const typingIndicator = document.querySelector(".typing-indicator");
            if (typingIndicator) {
                typingIndicator.remove();
            }
        }
    }

    // Add a message to the chat
    function addMessage(role, content) {
        const messageDiv = document.createElement("div");
        messageDiv.className = `message ${role}`;

        const messageContent = document.createElement("div");
        messageContent.className = "message-content";
        messageContent.innerHTML = formatMessage(content); // Use the updated formatMessage

        messageDiv.appendChild(messageContent);
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return messageDiv;
    }

    // Update an existing assistant message or create a new one
    function updateAssistantMessage(content, isFinal = false) {
        const typingIndicator = document.querySelector(".typing-indicator");
        if (typingIndicator) {
            typingIndicator.remove();
        }

        let assistantMessageDiv = chatMessages.querySelector(
            ".message.assistant:last-child",
        );

        // Check if the last message is actually an assistant message and not a user message or something else.
        // Also, ensure it's not a finalized error message or a previous completed message.
        if (
            !assistantMessageDiv ||
            assistantMessageDiv.dataset.finalized === "true" ||
            !assistantMessageDiv.classList.contains("assistant")
        ) {
            assistantMessageDiv = addMessage("assistant", content);
        } else {
            const messageContent =
                assistantMessageDiv.querySelector(".message-content");
            if (messageContent) {
                messageContent.innerHTML = formatMessage(content);
            }
        }

        if (isFinal) {
            if (assistantMessageDiv)
                assistantMessageDiv.dataset.finalized = "true";
        }
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Add typing indicator
    function addTypingIndicator() {
        // Remove existing typing indicator first to prevent duplicates
        const existingIndicator = document.querySelector(".typing-indicator");
        if (existingIndicator) {
            existingIndicator.remove();
        }

        const typingDiv = document.createElement("div");
        typingDiv.className = "message assistant typing-indicator"; // Added 'assistant' class for styling consistency
        typingDiv.innerHTML = "<span></span><span></span><span></span>";
        chatMessages.appendChild(typingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    /**
     * Formats the message content using marked.js for Markdown and handles custom sections.
     * @param {string} content - The raw message content.
     * @returns {string} - The HTML formatted message.
     */
    function formatMessage(content) {
        if (typeof content !== "string" || !content.trim()) {
            return ""; // Return empty string if content is not a string or is empty
        }

        let processedContent = content;
        const customHtmlElements = [];
        let placeholderIndex = 0;

        // 1. Handle Custom <cite> tags (pre-format to HTML)
        processedContent = processedContent.replace(
            /<cite index="([^"]+)">([^<]+)<\/cite>/g,
            (match, index, text) => {
                const placeholder = `__CUSTOM_HTML_PLACEHOLDER_${placeholderIndex++}__`;
                customHtmlElements.push({
                    placeholder,
                    html: `<span class="citation" data-cite-index="${index}">${text.trim()}<sup class="citation-marker">[${index.trim()}]</sup></span>`,
                });
                return placeholder;
            },
        );

        // 2. Handle **Code Execution Result:** section
        //    Regex captures "Code Execution Result:" and the text following it,
        //    until two newlines followed by "Sources:", or just two newlines (end of section), or end of string.
        processedContent = processedContent.replace(
            /(?:^|\n\n)\*\*Code Execution Result:\*\*([\s\S]*?)(?=\n\n(?:Sources:|$)|$)/g,
            (match, codeResultText) => {
                const placeholder = `__CUSTOM_HTML_PLACEHOLDER_${placeholderIndex++}__`;
                // Parse the inner content of the code result with marked.js
                const formattedCodeResultText = marked.parse(
                    codeResultText.trim(),
                );
                customHtmlElements.push({
                    placeholder,
                    html: `<div class='code-execution-result'><strong>Code Execution Result:</strong><br><br>${formattedCodeResultText}</div>`,
                });
                return placeholder;
            },
        );

        // 3. Handle Sources: section
        //    Regex captures "Sources:" and all text following it to the end of the string,
        //    assuming it's a top-level section.
        processedContent = processedContent.replace(
            /(?:^|\n\n)Sources:([\s\S]*)/g,
            (match, sourcesText) => {
                const placeholder = `__CUSTOM_HTML_PLACEHOLDER_${placeholderIndex++}__`;
                const sourcesArray = sourcesText
                    .trim()
                    .split(/\n|<br\s*\/?>/)
                    .filter((s) => s.trim().length > 0);

                const sourcesHtmlList = sourcesArray
                    .map((sourceLine) => {
                        // Parse each source line with marked.js to handle potential Markdown links within it
                        return `<li>${marked.parseInline(sourceLine.trim())}</li>`; // Use parseInline for list items
                    })
                    .join("");

                customHtmlElements.push({
                    placeholder,
                    html: `<div class="sources-section"><h4>Sources:</h4><ol class="sources-list">${sourcesHtmlList}</ol></div>`,
                });
                return placeholder;
            },
        );

        // 4. Main Markdown Parsing for the rest of the content
        let mainHtml = marked.parse(processedContent);

        // 5. Restore Custom HTML Elements
        customHtmlElements.forEach((element) => {
            // Ensure placeholder exists before replacing to avoid errors if regexes didn't match as expected
            if (mainHtml.includes(element.placeholder)) {
                mainHtml = mainHtml.replace(element.placeholder, element.html);
            } else {
                // If a placeholder wasn't found, it might indicate an issue with the regex or content structure.
                // For robustness, append the HTML if its placeholder is missing, though this is a fallback.
                // console.warn(`Placeholder ${element.placeholder} not found in main HTML. Appending content.`);
                // mainHtml += element.html; // This might not be desired, depends on expected behavior.
                // Better to ensure regexes are robust.
            }
        });

        return mainHtml;
    }

    // Reset input height
    function resetInputHeight() {
        chatInput.style.height = ""; // Reset to auto or initial height
    }

    // Auto-resize the textarea as the user types
    function resizeTextarea() {
        resetInputHeight(); // Reset first
        const maxHeight = 150; // Max height in pixels
        // scrollHeight includes padding, border, and content.
        // clientHeight includes padding but not border or scrollbar.
        // For textarea, scrollHeight is a good measure of content height.
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
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Focus input on load
    chatInput.focus();
});
