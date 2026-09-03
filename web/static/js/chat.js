(() => {
    const chatContainer = document.getElementById("chatContainer");
    const chatForm = document.getElementById("chatForm");
    const messageInput = document.getElementById("messageInput");
    const sendBtn = document.getElementById("sendBtn");
    const clearBtn = document.getElementById("clearBtn");
    const ingestBtn = document.getElementById("ingestBtn");
    const statsBtn = document.getElementById("statsBtn");
    const testBtn = document.getElementById("testBtn");
    const statusBar = document.getElementById("statusBar");
    const modal = document.getElementById("modal");
    const modalBody = document.getElementById("modalBody");
    const modalClose = document.getElementById("modalClose");

    let sessionId = localStorage.getItem("rag_session_id") || null;
    let isLoading = false;

    messageInput.addEventListener("input", () => {
        messageInput.style.height = "auto";
        messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + "px";
    });

    messageInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event("submit"));
        }
    });

    function appendMessage(text, role, sources = []) {
        const div = document.createElement("div");
        div.className = `message ${role}-message`;

        let html = `<div class="message-content">${escapeHtml(text)}</div>`;
        if (sources.length > 0) {
            html += `<div class="message-sources">📚 Источники: ${sources.join(", ")}</div>`;
        }
        div.innerHTML = html;
        chatContainer.appendChild(div);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function showTyping() {
        const div = document.createElement("div");
        div.className = "message bot-message";
        div.id = "typingIndicator";
        div.innerHTML = `
            <div class="message-content">
                <div class="typing-indicator">
                    <span></span><span></span><span></span>
                </div>
            </div>`;
        chatContainer.appendChild(div);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function hideTyping() {
        const el = document.getElementById("typingIndicator");
        if (el) el.remove();
    }

    function escapeHtml(text) {
        const div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }

    function setLoading(loading) {
        isLoading = loading;
        sendBtn.disabled = loading;
        messageInput.disabled = loading;
    }

    function showModal(text) {
        modalBody.textContent = text;
        modal.classList.remove("hidden");
    }

    modalClose.addEventListener("click", () => modal.classList.add("hidden"));
    modal.addEventListener("click", (e) => {
        if (e.target === modal) modal.classList.add("hidden");
    });

    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const message = messageInput.value.trim();
        if (!message || isLoading) return;

        appendMessage(message, "user");
        messageInput.value = "";
        messageInput.style.height = "auto";
        setLoading(true);
        showTyping();

        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message, session_id: sessionId }),
            });

            const data = await res.json();
            hideTyping();

            if (!res.ok) {
                appendMessage(data.error || "Ошибка сервера", "bot");
                return;
            }

            sessionId = data.session_id;
            localStorage.setItem("rag_session_id", sessionId);
            appendMessage(data.answer, "bot", data.sources || []);
        } catch (err) {
            hideTyping();
            appendMessage("Не удалось связаться с сервером.", "bot");
        } finally {
            setLoading(false);
            messageInput.focus();
        }
    });

    clearBtn.addEventListener("click", async () => {
        if (!sessionId) {
            showModal("История пуста.");
            return;
        }

        const res = await fetch("/api/clear", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: sessionId }),
        });
        const data = await res.json();
        chatContainer.innerHTML = "";
        showModal(`История очищена. Удалено сообщений: ${data.messages_removed}`);
    });

    ingestBtn.addEventListener("click", async () => {
        ingestBtn.disabled = true;
        showModal("Индексация документов...\nЭто может занять несколько минут.");

        try {
            const res = await fetch("/api/ingest", { method: "POST" });
            const data = await res.json();

            if (!res.ok) {
                showModal(`Ошибка: ${data.error}`);
                return;
            }

            statusBar.innerHTML = `<span class="status-ok">✅ База знаний загружена (${data.documents} док.)</span>`;
            showModal(
                `Индексация завершена!\n\n` +
                `Документов: ${data.documents}\n` +
                `Векторов: ${data.vectors}\n` +
                `Размерность: ${data.dimension}`
            );
        } catch (err) {
            showModal("Ошибка при индексации.");
        } finally {
            ingestBtn.disabled = false;
        }
    });

    statsBtn.addEventListener("click", async () => {
        const res = await fetch("/api/stats");
        const data = await res.json();

        let text = `📊 Статистика RAG-системы\n\n`;
        text += `Провайдер: ${data.provider}\n`;
        text += `База знаний: ${data.is_loaded ? "загружена" : "не загружена"}\n`;
        text += `Документов: ${data.total_documents}\n`;
        text += `Векторов: ${data.total_vectors}\n`;
        text += `Размерность: ${data.dimension}\n`;
        if (data.cached_queries !== undefined) {
            text += `Кэш SQLite: ${data.cached_queries} запросов\n`;
        }
        text += `\nМодели:\n`;
        text += `  Чат: ${data.chat_model}\n`;
        text += `  Эмбеддинги: ${data.embed_model}\n`;
        if (data.api_url) text += `\nAPI URL: ${data.api_url}`;

        showModal(text);
    });

    testBtn.addEventListener("click", async () => {
        testBtn.disabled = true;
        showModal("Проверка подключения к API...");

        try {
            const res = await fetch("/api/test");
            const data = await res.json();
            showModal(
                data.ok
                    ? `✅ Подключение к ${data.provider} работает!`
                    : `❌ Ошибка подключения к ${data.provider}. Проверьте .env`
            );
        } catch (err) {
            showModal("Не удалось выполнить проверку.");
        } finally {
            testBtn.disabled = false;
        }
    });
})();
