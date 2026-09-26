/* ==========================================================================
   QUARK AI COMPANION — PURE MODERN JAVASCRIPT
   - Interactive 3D Neural Sphere Canvas Animation
   - Real-time RAG Chat Stream Integration
   - Settings & Theme Controls
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const heroSection = document.getElementById("heroSection");
  const chatStream = document.getElementById("chatStream");
  const messagesList = document.getElementById("messagesList");
  const chatForm = document.getElementById("chatForm");
  const messageInput = document.getElementById("messageInput");
  const themeToggleBtn = document.getElementById("themeToggleBtn");
  const themeIcon = document.getElementById("themeIcon");
  const settingsToggleBtn = document.getElementById("settingsToggleBtn");
  const settingsDrawer = document.getElementById("settingsDrawer");
  const closeDrawerBtn = document.getElementById("closeDrawerBtn");
  const tavilyKeyInput = document.getElementById("tavilyKeyInput");
  const tempSlider = document.getElementById("tempSlider");
  const tempValue = document.getElementById("tempValue");
  const deviceBadge = document.getElementById("deviceBadge");
  const clearHistoryBtn = document.getElementById("clearHistoryBtn");

  // State
  let chatStarted = false;
  let isThinking = false;

  // Load Saved Preferences
  const savedKey = localStorage.getItem("quark_tavily_key") || "";
  tavilyKeyInput.value = savedKey;

  const savedTheme = localStorage.getItem("quark_theme") || "light";
  applyTheme(savedTheme);

  localStorage.removeItem("quark_visual");

  // Fetch Server Hardware Status
  fetch("/api/status")
    .then((r) => r.json())
    .then((data) => {
      deviceBadge.textContent = `Hardware: ${data.device} (${data.precision}) • ${data.model}`;
    })
    .catch(() => {
      deviceBadge.textContent = "Backend offline or connecting...";
    });

  // Theme Toggle
  themeToggleBtn.addEventListener("click", () => {
    const isDark = document.body.classList.contains("theme-dark");
    applyTheme(isDark ? "light" : "dark");
  });

  function applyTheme(theme) {
    if (theme === "dark") {
      document.body.classList.remove("theme-light");
      document.body.classList.add("theme-dark");
      themeIcon.textContent = "☀️";
      localStorage.setItem("quark_theme", "dark");
    } else {
      document.body.classList.remove("theme-dark");
      document.body.classList.add("theme-light");
      themeIcon.textContent = "🌙";
      localStorage.setItem("quark_theme", "light");
    }
  }

  // Settings Drawer Toggle
  settingsToggleBtn.addEventListener("click", () => {
    settingsDrawer.classList.toggle("hidden");
  });
  closeDrawerBtn.addEventListener("click", () => {
    settingsDrawer.classList.add("hidden");
  });

  tavilyKeyInput.addEventListener("input", (e) => {
    localStorage.setItem("quark_tavily_key", e.target.value.trim());
  });

  tempSlider.addEventListener("input", (e) => {
    tempValue.textContent = e.target.value;
  });

  clearHistoryBtn.addEventListener("click", () => {
    messagesList.innerHTML = "";
    chatStream.classList.add("hidden");
    heroSection.classList.remove("hidden");
    chatStarted = false;
    settingsDrawer.classList.add("hidden");
  });

  // ==========================================================================
  // 3D NEURAL SPHERE PARTICLE CANVAS (Matching Image 3)
  // ==========================================================================
  const canvas = document.getElementById("neuralCanvas");
  const ctx = canvas.getContext("2d");
  const miniCanvas = document.getElementById("miniNeuralCanvas");
  const miniCtx = miniCanvas ? miniCanvas.getContext("2d") : null;

  const NUM_NODES = 140;
  const SPHERE_RADIUS = 85;
  const CONNECT_DIST = 38;
  const nodes = [];

  // Generate nodes using Fibonacci Sphere algorithm
  const phi = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < NUM_NODES; i++) {
    const y = 1 - (i / (NUM_NODES - 1)) * 2;
    const radiusAtY = Math.sqrt(1 - y * y);
    const theta = phi * i;
    const x = Math.cos(theta) * radiusAtY;
    const z = Math.sin(theta) * radiusAtY;

    nodes.push({
      x: x * SPHERE_RADIUS,
      y: y * SPHERE_RADIUS,
      z: z * SPHERE_RADIUS,
      baseRadius: Math.random() * 1.6 + 1.2,
      pulsePhase: Math.random() * Math.PI * 2,
    });
  }

  let angleX = 0.003;
  let angleY = 0.005;
  let rotX = 0;
  let rotY = 0;
  let isDragging = false;
  let lastMouseX = 0;
  let lastMouseY = 0;

  canvas.addEventListener("mousedown", (e) => {
    isDragging = true;
    lastMouseX = e.clientX;
    lastMouseY = e.clientY;
  });
  window.addEventListener("mouseup", () => {
    isDragging = false;
  });
  window.addEventListener("mousemove", (e) => {
    if (isDragging) {
      const dx = e.clientX - lastMouseX;
      const dy = e.clientY - lastMouseY;
      rotY += dx * 0.008;
      rotX += dy * 0.008;
      lastMouseX = e.clientX;
      lastMouseY = e.clientY;
    }
  });

  function renderSphere(targetCanvas, targetCtx, scaleFactor = 1.0) {
    const width = targetCanvas.width;
    const height = targetCanvas.height;
    const centerX = width / 2;
    const centerY = height / 2;

    targetCtx.clearRect(0, 0, width, height);

    const isDark = document.body.classList.contains("theme-dark");
    const nodeColor = isDark ? "rgba(255, 255, 255," : "rgba(20, 20, 20,";
    const lineColor = isDark ? "rgba(56, 189, 248," : "rgba(20, 20, 20,";

    // Speed multiplier if Quark is thinking
    const speedMult = isThinking ? 3.5 : 1.0;
    rotX += angleX * speedMult;
    rotY += angleY * speedMult;

    const cosX = Math.cos(rotX);
    const sinX = Math.sin(rotX);
    const cosY = Math.cos(rotY);
    const sinY = Math.sin(rotY);

    const projected = [];

    for (let i = 0; i < nodes.length; i++) {
      const node = nodes[i];

      // Rotate Y
      const x1 = node.x * cosY - node.z * sinY;
      const z1 = node.z * cosY + node.x * sinY;

      // Rotate X
      const y2 = node.y * cosX - z1 * sinX;
      const z2 = z1 * cosX + node.y * sinX;

      // Perspective projection
      const fov = 240 * scaleFactor;
      const distance = 260;
      const scale = fov / (distance + z2);

      const px = x1 * scale + centerX;
      const py = y2 * scale + centerY;
      const alpha = Math.max(0.12, Math.min(1.0, (z2 + SPHERE_RADIUS) / (SPHERE_RADIUS * 2)));

      projected.push({
        x: px,
        y: py,
        z: z2,
        alpha: alpha,
        radius: node.baseRadius * scale,
        node: node,
      });
    }

    // Sort by depth for correct occlusion
    projected.sort((a, b) => a.z - b.z);

    // Draw connecting neural filaments
    targetCtx.lineWidth = 0.85 * scaleFactor;
    for (let i = 0; i < projected.length; i++) {
      const p1 = projected[i];
      for (let j = i + 1; j < projected.length; j++) {
        const p2 = projected[j];
        const dx = p1.x - p2.x;
        const dy = p1.y - p2.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < CONNECT_DIST * scaleFactor) {
          const lineAlpha = (1 - dist / (CONNECT_DIST * scaleFactor)) * Math.min(p1.alpha, p2.alpha) * 0.45;
          targetCtx.strokeStyle = `${lineColor} ${lineAlpha})`;
          targetCtx.beginPath();
          targetCtx.moveTo(p1.x, p1.y);
          targetCtx.lineTo(p2.x, p2.y);
          targetCtx.stroke();
        }
      }
    }

    // Draw nodes / glowing neural vertices
    for (let i = 0; i < projected.length; i++) {
      const p = projected[i];
      targetCtx.fillStyle = `${nodeColor} ${p.alpha})`;
      targetCtx.beginPath();
      targetCtx.arc(p.x, p.y, Math.max(0.6, p.radius), 0, Math.PI * 2);
      targetCtx.fill();

      // Outer glow on foreground nodes
      if (p.z > 25 && isDark) {
        targetCtx.fillStyle = `rgba(56, 189, 248, ${p.alpha * 0.3})`;
        targetCtx.beginPath();
        targetCtx.arc(p.x, p.y, p.radius * 2.2, 0, Math.PI * 2);
        targetCtx.fill();
      }
    }
  }

  function animationLoop() {
    renderSphere(canvas, ctx, 1.0);
    if (miniCanvas && miniCtx && chatStarted) {
      renderSphere(miniCanvas, miniCtx, 0.28);
    }
    requestAnimationFrame(animationLoop);
  }
  animationLoop();

  // ==========================================================================
  // CHAT INTERACTION & RAG STREAMING
  // ==========================================================================
  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = messageInput.value.trim();
    if (!query || isThinking) return;

    messageInput.value = "";

    // Switch from Hero screen to Conversation view
    if (!chatStarted) {
      heroSection.classList.add("hidden");
      chatStream.classList.remove("hidden");
      chatStarted = true;
    }

    // Append User Message
    appendUserMessage(query);

    // Append Thinking Status Card
    const thinkingCard = appendThinkingCard();
    isThinking = true;

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: query,
          tavily_key: tavilyKeyInput.value.trim(),
          temperature: parseFloat(tempSlider.value),
        }),
      });

      if (!res.ok) throw new Error("Server responded with error");
      const data = await res.json();

      // Replace thinking card with real response
      thinkingCard.remove();
      appendAssistantMessage(data);
    } catch (err) {
      thinkingCard.remove();
      appendErrorMessage("Could not connect to Quark server. Please make sure server is running.");
    } finally {
      isThinking = false;
      messageInput.focus();
    }
  });

  function appendUserMessage(text) {
    const bubble = document.createElement("div");
    bubble.className = "user-bubble";
    bubble.textContent = `🧑 ${text}`;
    messagesList.appendChild(bubble);
    scrollToBottom();
  }

  function appendThinkingCard() {
    const card = document.createElement("div");
    card.className = "assistant-card thinking-card";
    card.innerHTML = `
      <div class="spinner"></div>
      <span>Quark is retrieving knowledge & generating answer...</span>
    `;
    messagesList.appendChild(card);
    scrollToBottom();
    return card;
  }

  function appendAssistantMessage(data) {
    const card = document.createElement("div");
    card.className = "assistant-card";

    const webInfo = data.web_info || {};
    const engine = webInfo.engine || "Verified Knowledge";
    const latency = webInfo.latency_ms || 0;
    const answer = webInfo.answer || "";
    const sources = webInfo.sources || [];

    let sourcesSection = "";
    if (sources.length > 0) {
      const chips = sources
        .filter((s) => s.url)
        .map(
          (s) =>
            `<a href="${s.url}" target="_blank" class="source-chip" rel="noopener">🔗 ${escapeHtml(
              s.title.slice(0, 32)
            )}</a>`
        )
        .join("");
      if (chips) {
        sourcesSection = `
          <div style="margin-top: 10px; padding-top: 8px; border-top: 1px dashed rgba(128,128,128,0.2);">
            <div style="font-size: 11px; font-weight: 700; color: var(--text-muted); margin-bottom: 4px;">WEB SOURCES FOUND:</div>
            <div class="source-chips">${chips}</div>
          </div>`;
      }
    }

    card.innerHTML = `
      <div class="quark-response-meta">🦙 QUARKLLAMA (0.5B):</div>
      <div class="quark-response-text" style="white-space: pre-line;">${escapeHtml(data.reply || "No reply generated.")}</div>
      ${sourcesSection}
    `;

    messagesList.appendChild(card);
    scrollToBottom();
  }

  function appendErrorMessage(msg) {
    const card = document.createElement("div");
    card.className = "assistant-card";
    card.style.borderColor = "#ef4444";
    card.innerHTML = `<div style="color: #ef4444; font-weight: 600;">⚠️ ${escapeHtml(msg)}</div>`;
    messagesList.appendChild(card);
    scrollToBottom();
  }

  function scrollToBottom() {
    messagesList.scrollTop = messagesList.scrollHeight;
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});
