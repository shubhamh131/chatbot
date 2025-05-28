document.addEventListener("DOMContentLoaded", function () {
  // DOM elements - cached for better performance
  const chatBox = document.getElementById("chat-box");
  const userInput = document.getElementById("user-input");
  const sendButton = document.getElementById("send-btn");
  const micButton = document.getElementById("mic-btn");
  const clearButton = document.getElementById("clear-chat");
  const themeToggle = document.getElementById("theme-toggle");
  const typingIndicator = document.getElementById("typing-indicator");

  // State management
  let isProcessing = false;

  // Typing indicator functions
  function showTypingIndicator() {
    if (typingIndicator) {
      typingIndicator.style.display = "flex";
      typingIndicator.setAttribute("aria-hidden", "false");
    }
  }

  function hideTypingIndicator() {
    if (typingIndicator) {
      typingIndicator.style.display = "none";
      typingIndicator.setAttribute("aria-hidden", "true");
    }
  }

  // Initialize chatbot
  function initChatbot() {
    try {
      // Ensure typing indicator is hidden on startup
      hideTypingIndicator();

      // Focus on input field with slight delay to ensure DOM is ready
      setTimeout(() => {
        if (userInput) {
          userInput.focus();
        }
      }, 100);

      // Load saved theme preference
      loadThemePreference();

      // Set initial button states
      updateButtonStates();

      console.log("Chatbot initialized successfully");
    } catch (error) {
      console.error("Error during chatbot initialization:", error);
    }
  }

  // Load theme preference
  function loadThemePreference() {
    try {
      const isDarkTheme = localStorage.getItem("darkTheme") === "true";
      if (isDarkTheme) {
        document.body.classList.add("dark-theme");
        const icon = themeToggle?.querySelector("i");
        if (icon) {
          icon.classList.remove("fa-moon");
          icon.classList.add("fa-sun");
        }
      }
    } catch (error) {
      console.warn("Could not load theme preference:", error);
    }
  }

  // Update button states
  function updateButtonStates() {
    if (sendButton) {
      sendButton.disabled = isProcessing;
    }
    if (userInput) {
      userInput.disabled = isProcessing;
    }
  }

  // Send message function with improved error handling
  function sendMessage() {
    const message = userInput?.value?.trim();

    // Validation
    if (!message || isProcessing) {
      return;
    }

    try {
      isProcessing = true;
      updateButtonStates();

      // Add user message to chat
      addMessage("user", message);

      // Clear input
      userInput.value = "";

      // Remove welcome message if present
      removeWelcomeMessage();

      // Show typing indicator and get bot response
      showTypingIndicator();
      fetchBotResponse(message);
    } catch (error) {
      console.error("Error sending message:", error);
      handleError("Failed to send message");
    }
  }

  // Remove welcome message
  function removeWelcomeMessage() {
    const welcomeMessage = document.querySelector(".welcome-message");
    if (welcomeMessage) {
      welcomeMessage.style.opacity = "0";
      setTimeout(() => welcomeMessage.remove(), 300);
    }
  }

  // Add message to chat with improved sanitization
  function addMessage(sender, message) {
    if (!chatBox || !message) return;

    const messageElement = document.createElement("div");
    messageElement.className = `message ${sender}-message`;

    // Sanitize message content
    const sanitizedMessage = escapeHtml(message);

    if (sender === "user") {
      messageElement.innerHTML = `
        <div class="message-content">
          <p>${sanitizedMessage}</p>
        </div>
        <div class="avatar">
          <i class="fas fa-user" aria-label="User"></i>
        </div>
      `;
    } else {
      messageElement.innerHTML = `
        <div class="avatar">
          <i class="fas fa-robot" aria-label="Bot"></i>
        </div>
        <div class="message-content">
          <p>${sanitizedMessage}</p>
        </div>
      `;
    }

    // Add with animation
    messageElement.style.opacity = "0";
    messageElement.style.transform = "translateY(20px)";
    chatBox.appendChild(messageElement);

    // Trigger animation
    requestAnimationFrame(() => {
      messageElement.style.transition =
        "opacity 0.3s ease, transform 0.3s ease";
      messageElement.style.opacity = "1";
      messageElement.style.transform = "translateY(0)";
    });

    scrollToBottom();
  }

  // HTML escape function for security
  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  // Fetch response from server with improved error handling
  async function fetchBotResponse(message) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout

      const response = await fetch("/get_response", {
        method: "POST",
        body: JSON.stringify({ message: message }),
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (!data || !data.response) {
        throw new Error("Invalid response format");
      }

      // Add realistic delay for better UX
      const delay = Math.min(1500, Math.max(800, message.length * 20));

      setTimeout(() => {
        hideTypingIndicator();
        addMessage("bot", data.response);
        isProcessing = false;
        updateButtonStates();

        // Re-focus input for better UX
        if (userInput) {
          userInput.focus();
        }
      }, delay);
    } catch (error) {
      console.error("Fetch error:", error);
      handleError(getErrorMessage(error));
    }
  }

  // Get user-friendly error message
  function getErrorMessage(error) {
    if (error.name === "AbortError") {
      return "Request timed out. Please try again.";
    } else if (error.message.includes("Failed to fetch")) {
      return "Network error. Please check your connection.";
    } else {
      return "Sorry, I encountered an error processing your request.";
    }
  }

  // Handle errors consistently
  function handleError(message) {
    hideTypingIndicator();
    addMessage("bot", message);
    isProcessing = false;
    updateButtonStates();

    if (userInput) {
      userInput.focus();
    }
  }

  // Smooth scroll to bottom
  function scrollToBottom() {
    if (chatBox) {
      chatBox.scrollTo({
        top: chatBox.scrollHeight,
        behavior: "smooth",
      });
    }
  }

  // Clear chat history with improved UX
  function clearChat() {
    if (isProcessing) return;

    const hasMessages = chatBox && chatBox.children.length > 1;

    if (hasMessages && !confirm("Clear the entire conversation?")) {
      return;
    }

    if (chatBox) {
      // Fade out current content
      chatBox.style.opacity = "0.5";

      setTimeout(() => {
        chatBox.innerHTML = `
          <div class="welcome-message">
            <div class="welcome-icon">
              <i class="fas fa-robot"></i>
            </div>
            <h2>Hello, I'm GLITCH</h2>
            <p>Your AI assistant. How can I help you today?</p>
          </div>
        `;
        chatBox.style.opacity = "1";

        if (userInput) {
          userInput.focus();
        }
      }, 200);
    }
  }

  // Toggle dark/light theme with improved performance
  function toggleTheme() {
    try {
      const body = document.body;
      const icon = themeToggle?.querySelector("i");

      if (!icon) return;

      const isDarkMode = body.classList.toggle("dark-theme");

      if (isDarkMode) {
        icon.classList.remove("fa-moon");
        icon.classList.add("fa-sun");
        localStorage.setItem("darkTheme", "true");
      } else {
        icon.classList.remove("fa-sun");
        icon.classList.add("fa-moon");
        localStorage.setItem("darkTheme", "false");
      }
    } catch (error) {
      console.warn("Theme toggle error:", error);
    }
  }

  // Debounced input validation
  function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  }

  // Handle input changes for better UX
  const handleInputChange = debounce(() => {
    if (sendButton && userInput) {
      const hasText = userInput.value.trim().length > 0;
      sendButton.style.opacity = hasText ? "1" : "0.6";
    }
  }, 100);

  // Event listeners with improved error handling
  function setupEventListeners() {
    try {
      // Send button
      sendButton?.addEventListener("click", sendMessage);

      // Enter key in input
      userInput?.addEventListener("keypress", function (event) {
        if (event.key === "Enter" && !event.shiftKey) {
          event.preventDefault();
          sendMessage();
        }
      });

      // Input change for visual feedback
      userInput?.addEventListener("input", handleInputChange);

      // Clear button
      clearButton?.addEventListener("click", clearChat);

      // Theme toggle
      themeToggle?.addEventListener("click", toggleTheme);

      // Mic button
      micButton?.addEventListener("click", function () {
        // Show a more user-friendly message
        const message =
          "🎤 Voice input feature is coming soon! Stay tuned for updates.";
        addMessage("bot", message);
      });

      // Prevent form submission if input is in a form
      userInput?.closest("form")?.addEventListener("submit", function (e) {
        e.preventDefault();
        sendMessage();
      });
    } catch (error) {
      console.error("Error setting up event listeners:", error);
    }
  }

  // Initialize everything
  function initialize() {
    try {
      initChatbot();
      setupEventListeners();

      // Add keyboard shortcuts
      document.addEventListener("keydown", function (e) {
        // Ctrl/Cmd + K to clear chat
        if ((e.ctrlKey || e.metaKey) && e.key === "k") {
          e.preventDefault();
          clearChat();
        }
        // Escape to focus input
        if (e.key === "Escape" && userInput) {
          userInput.focus();
        }
      });
    } catch (error) {
      console.error("Initialization error:", error);
    }
  }

  // Start the application
  initialize();
});
