// DeepGrove chat — streaming chat UI with a live-updating reasoning trace.

const $ = (sel) => document.querySelector(sel);

const mainEl       = $("#main");
const topbarEl     = $("#topbar");
const composerEl   = $("#composer");
const chatArea     = $("#chatArea");
const messagesEl   = $("#messages");
const composerForm = $("#composerForm");
const composerInput= $("#composerInput");
const sendBtn      = $("#sendBtn");
const newChatBtn   = $("#newChatBtn");

let messages = [];
let pending = false;

// ─── settings ─────────────────────────────────────────────────────────────
// Temperature, search and python. All three shape the request rather than the
// display, so they are fixed once a conversation has started — see
// setSettingsLocked. sessionStorage, matching the access code: a reload keeps
// the choice, a new tab starts from the defaults.
const settingsBtn    = $("#settingsBtn");
const settingsPanel  = $("#settingsPanel");
const settingsLocked = $("#settingsLocked");
const tempRange      = $("#tempRange");
const tempVal        = $("#tempVal");
const searchOpt      = $("#searchOpt");
const pythonOpt      = $("#pythonOpt");

const SETTINGS_KEY = "dg.settings";
const DEFAULTS = { temperature: 1.0, search: true, python: true };

let settings = (() => {
  try {
    const raw = JSON.parse(sessionStorage.getItem(SETTINGS_KEY) || "{}");
    return {
      // Clamped on the way in as well as on the server — a hand-edited value in
      // storage should not be able to ask for something the slider cannot.
      temperature: Math.min(2, Math.max(0, Number(raw.temperature ?? DEFAULTS.temperature))) || 0,
      search: raw.search !== false,
      python: raw.python !== false,
    };
  } catch (_) { return { ...DEFAULTS }; }
})();
let settingsFixed = false;

function saveSettings() {
  try { sessionStorage.setItem(SETTINGS_KEY, JSON.stringify(settings)); } catch (_) {}
}

function renderSettings() {
  tempRange.value = String(settings.temperature);
  tempVal.textContent = settings.temperature.toFixed(1);
  searchOpt.checked = settings.search;
  pythonOpt.checked = settings.python;
  for (const el of [tempRange, searchOpt, pythonOpt]) el.disabled = settingsFixed;
  settingsPanel.classList.toggle("fixed", settingsFixed);
  settingsLocked.classList.toggle("hidden", !settingsFixed);
  // A dot on the button when anything differs from the defaults, so the state
  // is visible without opening the panel.
  const changed = settings.search !== DEFAULTS.search
    || settings.python !== DEFAULTS.python
    || Math.abs(settings.temperature - DEFAULTS.temperature) > 0.001;
  settingsBtn.classList.toggle("changed", changed);
}

// Locked once the conversation has one. Temperature could safely vary per turn,
// but the tool settings cannot: turning search off after the model has searched
// leaves the calls it made in the transcript while the prompt now says it has no
// tools, and sanitize_messages drops exactly those calls — so the conversation
// the model reads stops matching the one on screen. They lock together because
// they are one decision about how this chat behaves.
function setSettingsLocked(locked) {
  settingsFixed = locked;
  renderSettings();
}

function openSettings(on) {
  settingsPanel.classList.toggle("hidden", !on);
  settingsBtn.setAttribute("aria-expanded", on ? "true" : "false");
}

settingsBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  openSettings(settingsPanel.classList.contains("hidden"));
});
// Click-away and escape both close it; a panel you can only dismiss by finding
// the button again is a panel that feels stuck.
document.addEventListener("click", (e) => {
  if (!settingsPanel.contains(e.target) && e.target !== settingsBtn) openSettings(false);
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !settingsPanel.classList.contains("hidden")) openSettings(false);
});

tempRange.addEventListener("input", () => {
  if (settingsFixed) return;
  settings.temperature = Number(tempRange.value);
  saveSettings(); renderSettings();
});
searchOpt.addEventListener("change", () => {
  if (settingsFixed) { renderSettings(); return; }
  settings.search = searchOpt.checked;
  saveSettings(); renderSettings();
});
pythonOpt.addEventListener("change", () => {
  if (settingsFixed) { renderSettings(); return; }
  settings.python = pythonOpt.checked;
  saveSettings(); renderSettings();
});

// The names the server expects. It intersects this with GATEWAY_TOOLS, so
// asking for something the operator has not enabled simply yields nothing.
function activeTools() {
  const t = [];
  if (settings.search) t.push("search");
  if (settings.python) t.push("python");
  return t;
}

renderSettings();

// Handle on the in-flight request so the stop button, the esc key, and a
// page unload can all cancel the same stream.
let currentAborter = null;
let stopRequested = false;
// Bumped on reset. A cancelled turn finishes unwinding after the chat has
// already been cleared, and without this its partial answer would be pushed
// into the fresh conversation.
let chatEpoch = 0;

// ─── helpers ──────────────────────────────────────────────────────────────
function autosize(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 180) + "px";
}
composerInput.addEventListener("input", () => autosize(composerInput));
composerInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    composerForm.requestSubmit();
  }
});

// ─── cancelling ───────────────────────────────────────────────────────────
function setPending(on) {
  pending = on;
  // While streaming the send button becomes a stop button rather than going
  // disabled — cancelling has to stay reachable with the mouse.
  sendBtn.classList.toggle("stop", on);
  sendBtn.setAttribute("aria-label", on ? "stop generating" : "send");
  sendBtn.title = on ? "stop generating (esc)" : "send";
}

function abortCurrent() {
  if (!currentAborter) return false;
  stopRequested = true;
  try { currentAborter.abort(); } catch (_) {}
  return true;
}
// ─── scrolling ────────────────────────────────────────────────────────────
// Stick to the bottom while streaming, but only if the user hasn't scrolled
// up to read earlier messages — and never fight a smooth-scroll animation
// with itself on every single token.
function isNearBottom() {
  return chatArea.scrollHeight - chatArea.scrollTop - chatArea.clientHeight < 120;
}

// Cached instead of measured. Reading scrollHeight forces a synchronous
// layout, and doing that once per frame against a document that keeps growing
// was its own source of slowdown. The scroll event tells us when it changes.
let stickToBottom = true;
chatArea.addEventListener("scroll", () => { stickToBottom = isNearBottom(); }, { passive: true });
function scrollToBottom(smooth) {
  chatArea.scrollTo({ top: chatArea.scrollHeight, behavior: smooth ? "smooth" : "auto" });
}

// Lift the question just asked to the top of the viewport and let the answer
// print beneath it. The reply is empty at this point, so there's nothing to
// scroll into — a min-height on the assistant row reserves the space that
// makes the pin hold. It's released on the next turn so old exchanges
// collapse back to their natural height.
function pinTurnToTop(userEl, assistantEl) {
  messagesEl.querySelectorAll(".msg.spacer").forEach((el) => {
    el.classList.remove("spacer");
    el.style.minHeight = "";
  });

  const gap = parseFloat(getComputedStyle(messagesEl).rowGap) || 26;
  // Derived from the rows that aren't moving rather than read off chatArea.
  // On the first message the composer is still dropping from centre, so
  // chatArea is mid-transition and measuring it directly yields roughly half
  // the final height — and a spacer half as tall as the pin needs.
  const viewH = mainEl.clientHeight - topbarEl.offsetHeight - composerEl.offsetHeight;
  const room = viewH - userEl.offsetHeight - gap * 2;
  assistantEl.classList.add("spacer");
  assistantEl.style.minHeight = Math.max(room, 0) + "px";

  // offsetTop is relative to the nearest positioned ancestor, which isn't
  // necessarily the scroller — measure against chatArea directly.
  const top = userEl.getBoundingClientRect().top
    - chatArea.getBoundingClientRect().top
    + chatArea.scrollTop;
  chatArea.scrollTo({ top: Math.max(top - gap / 2, 0), behavior: "smooth" });
  // pinning deliberately parks the view away from the bottom, so streaming
  // must not immediately yank it back down
  stickToBottom = false;
}

// ─── rendering ────────────────────────────────────────────────────────────
function escapeHTML(s) {
  return s.replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}
function renderContent(text) {
  // marked + DOMPurify are both required before trusting the rich path —
  // marked alone passes raw HTML straight through (by design), so without
  // DOMPurify a model response containing literal HTML would render as-is.
  if (window.marked && window.DOMPurify) {
    // Protect math spans from markdown's inline-formatting pass (e.g. a `_`
    // inside a LaTeX subscript like `a_1` getting read as an italic marker)
    // — swap them for placeholders, run markdown, then restore afterward.
    const mathSpans = [];
    const guarded = text.replace(
      /\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\$[^\n$]+?\$|\\\([\s\S]+?\\\)/g,
      (m) => { mathSpans.push(m); return `${mathSpans.length - 1}`; }
    );
    let html = marked.parse(guarded, { breaks: true, gfm: true });
    // re-escape on the way back in: this is plain text landing inside HTML,
    // and the browser will decode the entities right back to literal
    // characters once parsed, which is exactly what KaTeX needs to see.
    html = html.replace(/(\d+)/g, (_, i) => escapeHTML(mathSpans[Number(i)]));
    return DOMPurify.sanitize(html);
  }
  // fallback if the CDN libs didn't load: escape everything, then carve out
  // code blocks — never set unescaped model output as innerHTML.
  text = escapeHTML(text);
  text = text.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, body) =>
    `<pre><code class="language-${lang || "plaintext"}">${body}</code></pre>`);
  text = text.replace(/`([^`]+)`/g, (_, code) => `<code>${code}</code>`);
  return text;
}

// ─── copying ──────────────────────────────────────────────────────────────
async function copyText(text) {
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch (_) { /* refused, or not a secure context — fall through */ }
  // The clipboard API needs a secure context and can be denied outright, so a
  // hidden textarea remains the only path that works everywhere. Deprecated,
  // not gone, and the alternative is a button that silently does nothing.
  try {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.cssText = "position:fixed;top:-1000px;opacity:0";
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand("copy");
    ta.remove();
    return ok;
  } catch (_) {
    return false;
  }
}

// `getText` is a function rather than a string because the text is read at click
// time — a block copied mid-stream should yield what is on screen now, not what
// existed when the button was built.
function makeCopyButton(getText, label = "copy") {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "copy-btn";
  btn.setAttribute("aria-label", "copy to clipboard");
  btn.innerHTML = `
    <svg viewBox="0 0 24 24" width="13" height="13" aria-hidden="true">
      <rect x="9" y="9" width="11" height="11" rx="2" fill="none"
            stroke="currentColor" stroke-width="2" />
      <path d="M5 15V5a2 2 0 0 1 2-2h8" fill="none" stroke="currentColor"
            stroke-width="2" stroke-linecap="round" />
    </svg><span class="copy-label"></span>`;
  const labelEl = btn.querySelector(".copy-label");
  labelEl.textContent = label;
  btn.addEventListener("click", async () => {
    const ok = await copyText(getText());
    labelEl.textContent = ok ? "copied" : "failed";
    btn.classList.toggle("copied", ok);
    btn.classList.toggle("failed", !ok);
    clearTimeout(btn._resetTimer);
    btn._resetTimer = setTimeout(() => {
      labelEl.textContent = label;
      btn.classList.remove("copied", "failed");
    }, 1400);
  });
  return btn;
}

// Wrap each code block in a header carrying its language and a copy button.
//
// A persistent bar rather than a button revealed on hover: there is no hover on
// a phone, and a control that only exists for mouse users is not a feature the
// whole audience has.
function enhanceCodeBlocks(root) {
  root.querySelectorAll("pre").forEach((pre) => {
    // finalize() re-renders the message once at the end, but guard anyway so
    // this can be called more than once without stacking headers.
    if (pre.parentElement && pre.parentElement.classList.contains("code-block")) return;

    const code = pre.querySelector("code");
    const lang = ((code && code.className.match(/language-([\w+-]+)/)) || [])[1] || "";

    const wrap = document.createElement("div");
    wrap.className = "code-block";
    const head = document.createElement("div");
    head.className = "code-head";
    const langEl = document.createElement("span");
    langEl.className = "code-lang";
    // Not innerHTML: the language comes out of the model's own fence.
    langEl.textContent = lang === "plaintext" ? "" : lang;
    head.appendChild(langEl);
    head.appendChild(makeCopyButton(() => (code || pre).textContent));

    pre.replaceWith(wrap);
    wrap.append(head, pre);
  });
}

// Syntax-highlight code blocks + typeset math. Deferred to finalize() rather
// than run on every streamed delta — re-highlighting/re-parsing the whole
// message on every token would be wasted work and visibly laggy on long
// responses, and partial code/math mid-stream would render broken anyway.
function polishContent(el) {
  el.querySelectorAll("pre code").forEach((block) => {
    if (window.hljs) {
      try { hljs.highlightElement(block); } catch (_) { /* leave as plain text */ }
    }
  });
  // After highlighting, so the copy button reads the finished node — though
  // textContent would be the same either way, since highlighting only wraps.
  enhanceCodeBlocks(el);
  if (window.renderMathInElement) {
    try {
      renderMathInElement(el, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "\\[", right: "\\]", display: true },
          { left: "$", right: "$", display: false },
          { left: "\\(", right: "\\)", display: false },
        ],
        throwOnError: false,
        // html-only output: KaTeX's default also emits an invisible MathML
        // copy for screen readers, which browsers still let you select/copy
        // — that's what produces doubled, garbled text when copying a
        // rendered formula off the page.
        output: "html",
      });
    } catch (_) { /* leave as raw text */ }
  }
}

function buildUserMessage(text) {
  const el = document.createElement("div");
  el.className = "msg user";
  el.innerHTML = `
    <div class="msg-body">
      <div class="msg-role">user</div>
      <div class="msg-content"></div>
    </div>
  `;
  el.querySelector(".msg-content").textContent = text;
  messagesEl.appendChild(el);
  return el;
}

// A turn is a sequence of things that happened: the model thinks, calls a tool,
// reads what came back, thinks again, then answers. The shell used to have one
// reasoning panel, one tool host and one content area, so all of that collapsed
// into three blobs whatever the real order was — a second round of thinking
// appended to the first, and every tool row clustered together, away from the
// reasoning that produced it.
//
// Blocks are now appended to one flow in arrival order, so the transcript reads
// as the turn actually ran.

// A contiguous burst of thinking. Its own block, so a second round of reasoning
// after a tool result is visibly a second round.
function makeReasoningBlock() {
  const wrap = document.createElement("div");
  wrap.className = "reasoning open thinking";
  wrap.innerHTML = `
    <button class="reasoning-toggle" type="button">
      <span class="chev">›</span>
      <span class="label">thinking</span>
      <span class="tok"></span>
    </button>
    <div class="reasoning-body"></div>`;
  const body = wrap.querySelector(".reasoning-body");
  const labelEl = wrap.querySelector(".label");
  const tokEl = wrap.querySelector(".tok");
  wrap.querySelector(".reasoning-toggle")
      .addEventListener("click", () => wrap.classList.toggle("open"));

  // Streamed text lands in a text node we append to, rather than reassigning
  // textContent — that was itself O(n) per flush against a large budget.
  const node = document.createTextNode("");
  body.appendChild(node);
  let text = "", flushed = 0, dirty = false;

  return {
    el: wrap,
    kind: "reasoning",
    get text() { return text; },
    add(delta) { text += delta; dirty = true; },
    flush() {
      if (!dirty) return;
      dirty = false;
      if (text.length > flushed) {
        node.appendData(text.slice(flushed));
        flushed = text.length;
        // A big number rather than scrollHeight: the browser clamps it, and we
        // avoid the layout flush that reading scrollHeight would force.
        body.scrollTop = 1e9;
      }
    },
    done() {
      this.flush();
      wrap.classList.remove("open", "thinking");
      labelEl.textContent = "reasoning";
      tokEl.textContent = `~${Math.round(text.length / 4)} tok`;
    },
  };
}

// A contiguous burst of answer text. Carries the incremental markdown renderer,
// which is per-block now that a turn can contain several.
function makeContentBlock() {
  const wrap = document.createElement("div");
  wrap.className = "md-out";
  let liveHost = document.createElement("div");
  liveHost.className = "md-live";
  wrap.appendChild(liveHost);

  let text = "", committedLen = 0, dirty = false;

  // Structure is scanned once as text arrives and never rescanned. Previously
  // every frame ran lastIndexOf + a fence count over the whole message, which
  // stayed quadratic even after parsing was made incremental.
  const fenceAt = [];
  let fenceScanFrom = 0, lastBlank = -1, blankScanFrom = 0;
  // A long code block is one DOM node the browser must re-lay-out on every
  // append. Splitting it lets finished segments drop out of layout entirely.
  const CODE_SEG_CHARS = 16000;
  let codePre = null, codeSegEl = null, codeSegNode = null;
  let codeSegLen = 0, codeAppendedTo = -1, openFenceAt = -1;

  function scanDelta() {
    let i = fenceScanFrom, idx;
    while ((idx = text.indexOf("```", i)) !== -1) { fenceAt.push(idx); i = idx + 3; }
    fenceScanFrom = Math.max(i, text.length - 2);
    let j = blankScanFrom, b;
    while ((b = text.indexOf("\n\n", j)) !== -1) { lastBlank = b; j = b + 2; }
    blankScanFrom = Math.max(j, text.length - 1);
  }
  function fencesBefore(pos) {
    let n = fenceAt.length;
    while (n > 0 && fenceAt[n - 1] >= pos) n--;
    return n;
  }
  function resetCode() {
    codePre = codeSegEl = codeSegNode = null;
    codeSegLen = 0; codeAppendedTo = -1; openFenceAt = -1;
  }
  function commitCompleted() {
    if (lastBlank < committedLen) return;
    const end = lastBlank + 2;
    if (fencesBefore(end) % 2 !== 0) return;
    const block = document.createElement("div");
    block.className = "md-block";
    block.innerHTML = renderContent(text.slice(committedLen, end));
    wrap.insertBefore(block, liveHost);
    committedLen = end;
    resetCode();
  }
  function appendCode(chunk) {
    let i = 0;
    while (i < chunk.length) {
      if (!codeSegNode || codeSegLen >= CODE_SEG_CHARS) {
        if (codeSegEl) codeSegEl.classList.add("seg-done");
        codeSegEl = document.createElement("code");
        codeSegEl.className = "code-seg";
        codeSegNode = document.createTextNode("");
        codeSegEl.appendChild(codeSegNode);
        codePre.appendChild(codeSegEl);
        codeSegLen = 0;
      }
      const part = chunk.slice(i, i + (CODE_SEG_CHARS - codeSegLen));
      codeSegNode.appendData(part);
      codeSegLen += part.length;
      i += part.length;
    }
  }
  // Inside an open fence we skip marked and DOMPurify entirely and append raw
  // text — that's the "model is emitting a whole website" case, where parsing
  // the tail every frame is what hurts.
  function renderLive() {
    const open = fenceAt.length % 2 === 1;
    if (!open) {
      if (codePre) resetCode();
      liveHost.innerHTML = renderContent(text.slice(committedLen));
      return;
    }
    const fenceIdx = fenceAt[fenceAt.length - 1];
    if (openFenceAt !== fenceIdx) {
      const nl = text.indexOf("\n", fenceIdx);
      const lang = text.slice(fenceIdx + 3, nl === -1 ? text.length : nl).trim();
      liveHost.innerHTML = "";
      const before = text.slice(committedLen, fenceIdx);
      if (before.trim()) {
        const b = document.createElement("div");
        b.innerHTML = renderContent(before);
        liveHost.appendChild(b);
      }
      codePre = document.createElement("pre");
      if (lang) codePre.dataset.lang = lang;
      liveHost.appendChild(codePre);
      codeSegEl = codeSegNode = null;
      codeSegLen = 0;
      openFenceAt = fenceIdx;
      codeAppendedTo = nl === -1 ? text.length : nl + 1;
    }
    if (text.length > codeAppendedTo) {
      appendCode(text.slice(codeAppendedTo));
      codeAppendedTo = text.length;
    }
  }

  return {
    el: wrap,
    kind: "content",
    get text() { return text; },
    add(delta) { text += delta; dirty = true; },
    flush() {
      if (!dirty) return;
      dirty = false;
      scanDelta();
      commitCompleted();
      renderLive();
    },
    done() { this.flush(); },
    // Incremental rendering is an approximation — it splits on blank lines and
    // renders open fences as raw text. Rebuild properly once at the end, where
    // a single parse costs nothing.
    rerender() {
      this.flush();
      wrap.innerHTML = renderContent(text);
      liveHost = null;
      resetCode();
      polishContent(wrap);
    },
  };
}

function buildAssistantShell() {
  const el = document.createElement("div");
  el.className = "msg assistant";
  el.innerHTML = `
    <div class="msg-body">
      <div class="msg-role">assistant</div>
      <div class="turn-flow">
        <span class="typing-dots"><span></span><span></span><span></span></span>
      </div>
      <div class="msg-actions hidden"></div>
      <div class="msg-stats hidden"></div>
    </div>
  `;
  const flow      = el.querySelector(".turn-flow");
  const actionsEl = el.querySelector(".msg-actions");
  const statsEl   = el.querySelector(".msg-stats");

  const RENDER_MS = 66;
  let renderQueued = false, lastRenderAt = 0;
  let current = null;        // the block currently receiving deltas
  let lastContent = null;    // most recent content block — the answer
  let reasoningChars = 0;

  function dropDots() { flow.querySelector(".typing-dots")?.remove(); }

  function scheduleFlush() {
    if (renderQueued) return;
    renderQueued = true;
    requestAnimationFrame(function tick(now) {
      if (now - lastRenderAt < RENDER_MS) { requestAnimationFrame(tick); return; }
      renderQueued = false;
      lastRenderAt = now;
      current?.flush();
      if (stickToBottom) scrollToBottom(false);
    });
  }

  // Anything that isn't more of the same kind closes the open block, which is
  // what makes the sequence legible rather than one running panel.
  function openBlock(kind) {
    if (current?.kind === kind) return current;
    if (current) { current.done(); if (current.kind === "reasoning") reasoningChars += current.text.length; }
    dropDots();
    current = kind === "reasoning" ? makeReasoningBlock() : makeContentBlock();
    flow.appendChild(current.el);
    if (kind === "content") lastContent = current;
    if (stickToBottom) scrollToBottom(false);
    return current;
  }

  // Into the document. Dropping this line during the rewrite is what made the
  // whole turn vanish: the shell was built, streamed into and finalised
  // correctly, all inside an element that was never in the page — which a
  // test calling buildAssistantShell() and inspecting shell.el cannot see.
  messagesEl.appendChild(el);

  return {
    el,
    addTool(name, code) {
      if (current) { current.done(); if (current.kind === "reasoning") reasoningChars += current.text.length; current = null; }
      dropDots();
      const row = document.createElement("div");
      row.className = "tool running";
      row.innerHTML = `
        <button class="tool-toggle" type="button">
          <span class="chev">›</span>
          <span class="label"></span>
          <span class="tok"></span>
        </button>
        <div class="tool-body"><pre class="tool-code"></pre><pre class="tool-out"></pre></div>
      `;
      const label = row.querySelector(".label");
      const tok = row.querySelector(".tok");
      const out = row.querySelector(".tool-out");
      row.querySelector(".tool-code").textContent = code;
      label.textContent = `running ${name}`;
      row.querySelector(".tool-toggle").addEventListener("click", () =>
        row.classList.toggle("open"));
      flow.appendChild(row);
      if (stickToBottom) scrollToBottom(false);

      return {
        // `output` is what the model was given, verbatim. `summary` is the
        // one-line label for the collapsed row.
        settle(ok, output, summary) {
          row.classList.remove("running");
          row.classList.toggle("failed", !ok);
          label.textContent = ok ? name : `${name} failed`;
          const line = summary || (output || "").split("\n").find((l) => l.trim()) || "";
          tok.textContent = line.length > 48 ? line.slice(0, 48) + "…" : line;
          out.textContent = output || "";
        },
      };
    },
    addReasoning(delta) { openBlock("reasoning").add(delta); scheduleFlush(); },
    addContent(delta)   { openBlock("content").add(delta);   scheduleFlush(); },
    finalize({ tokens, seconds, errored, stopped }) {
      if (current) { current.done(); if (current.kind === "reasoning") reasoningChars += current.text.length; }
      current = null;
      dropDots();

      const answer = lastContent ? lastContent.text : "";
      if (lastContent) lastContent.rerender();

      // If nothing was ever answered, show the thinking rather than an empty
      // bubble — it is the only thing the user has.
      if (!answer.trim()) {
        const note = document.createElement("div");
        note.className = "md-out";
        flow.querySelectorAll(".reasoning").forEach((r) => r.classList.add("open"));
        note.textContent = stopped ? "(stopped)"
          : errored ? "(no response — see error above)"
          : reasoningChars ? "" : "(empty response)";
        if (note.textContent) flow.appendChild(note);
      }

      if (answer.trim()) {
        actionsEl.innerHTML = "";
        actionsEl.appendChild(makeCopyButton(() => answer));
        actionsEl.classList.remove("hidden");
      }

      const parts = [];
      if (tokens > 0 && seconds > 0) {
        parts.push(`${tokens} tok · ${seconds.toFixed(2)}s · ${(tokens / seconds).toFixed(1)} tok/s`);
      }
      if (stopped) parts.push("stopped");
      if (parts.length) {
        statsEl.textContent = parts.join(" · ");
        statsEl.classList.remove("hidden");
      }
      return answer;
    },
  };
}

// Centred while the conversation is empty, bottom-docked once it isn't. The
// class drives grid-template-rows; see #main in style.css.
function setEmpty(on) { mainEl.classList.toggle("empty", on); }

function resetChat() {
  chatEpoch++;
  setSettingsLocked(false);
  messages = [];
  messagesEl.innerHTML = "";
  composerInput.value = "";
  autosize(composerInput);
  setEmpty(true);
}

// ─── api + session ────────────────────────────────────────────────────────
// Empty string keeps the same-origin behaviour used by local dev (app.py
// serves both the page and /api/chat). In production the page is static and
// the gateway lives on its own host, so index.html sets this.
const API_BASE = window.DEEPGROVE_API_BASE || "";

// Held in memory, deliberately not localStorage: a token that survives in
// storage is a token an XSS can lift and replay for its full hour. Losing it
// on reload costs one extra round trip and nothing else.
let sessionToken = null;
let sessionPromise = null;

// ─── turnstile ────────────────────────────────────────────────────────────
// Proves a human is present before the server will mint a session. Without
// it the mint endpoint is open, and per-session quota means nothing when
// sessions are free to create in a loop.
//
// Empty sitekey disables the whole path, which is what local dev against
// app.py wants — it has no verification and expects none.
const TURNSTILE_SITEKEY = window.DEEPGROVE_TURNSTILE_SITEKEY || "";

const turnstileHost = $("#turnstileHost");
let turnstileWidget = null;
let turnstileScript = null;
let turnstileToken = null;
let turnstilePending = null;

function loadTurnstile() {
  if (turnstileScript) return turnstileScript;
  turnstileScript = new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = "https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit";
    s.async = true;
    s.defer = true;
    s.onerror = () => reject(new Error("verification unavailable"));
    s.onload = () => {
      turnstileWidget = window.turnstile.render("#turnstileHost", {
        sitekey: TURNSTILE_SITEKEY,
        // Stays invisible unless Cloudflare actually wants an interaction, so
        // the overwhelming majority of visitors never see a challenge.
        appearance: "interaction-only",
        callback: (t) => {
          turnstileToken = t;
          turnstilePending?.resolve(t);
          turnstilePending = null;
        },
        // Turnstile hands the reason in as a code and it is the only thing
        // that distinguishes causes that look identical from the outside:
        // 110200 is a hostname missing from the widget's allow-list, 110100
        // is a wrong sitekey, the 300xxx/600xxx families are transient. A
        // bare "verification failed" makes all of those the same bug report.
        "error-callback": (code) => {
          console.error("turnstile error", code, "on", location.hostname);
          const hint = String(code || "").startsWith("110200")
            ? `verification rejected this domain (${location.hostname}) — add it to the Turnstile widget`
            : `verification failed (${code || "no code"})`;
          turnstilePending?.reject(new Error(hint));
          turnstilePending = null;
        },
        "expired-callback": () => { turnstileToken = null; },
      });
      resolve();
    };
    document.head.appendChild(s);
  });
  return turnstileScript;
}

// Nothing below is allowed to wait forever. Turnstile has no timeout of its
// own, and when it can't run it simply never calls back — which stalls the
// session mint before the request is even sent, with no error anywhere.
function withTimeout(promise, ms, message) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(message)), ms);
    promise.then(
      (v) => { clearTimeout(timer); resolve(v); },
      (e) => { clearTimeout(timer); reject(e); },
    );
  });
}

// The widget is on screen only while a token is genuinely outstanding. It stays
// in the DOM either way — Turnstile owns that iframe and re-rendering it per
// mint would be slower and more fragile — so visibility is a class we control
// rather than a consequence of the iframe existing.
function showTurnstile(on) {
  turnstileHost?.classList.toggle("awaiting", on);
}

async function getTurnstileToken() {
  if (!TURNSTILE_SITEKEY) return null;
  showTurnstile(true);
  try {
    await withTimeout(loadTurnstile(), 15000, "verification didn't load");
    // Tokens are single-use — hand back the one the initial render produced,
    // then reset for any subsequent mint rather than replaying a spent token.
    if (turnstileToken) {
      const t = turnstileToken;
      turnstileToken = null;
      return t;
    }
    return await withTimeout(
      new Promise((resolve, reject) => {
        turnstilePending = { resolve, reject };
        window.turnstile.reset(turnstileWidget);
      }),
      30000,
      "verification timed out",
    );
  } finally {
    // Hidden on every exit, including failure — a widget left up after a
    // timeout is just clutter the user cannot act on.
    showTurnstile(false);
  }
}

// ─── access gate ──────────────────────────────────────────────────────────
// A shared code in front of the whole app. This overlay only decides what the
// page shows — the code itself is checked by the server when it mints a
// session, so hiding the lock screen from devtools buys nothing.
//
// sessionStorage, not localStorage: a reload inside the tab shouldn't
// re-prompt, but the code shouldn't outlive the tab either.
const gateForm  = $("#gateForm");
const gateInput = $("#gateInput");
const gateBtn   = $("#gateBtn");
const gateError = $("#gateError");

const GATE_KEY = "dg.access";
const readStored = (k) => { try { return sessionStorage.getItem(k) || ""; } catch (_) { return ""; } };
const clearStored = (k) => { try { sessionStorage.removeItem(k); } catch (_) {} };

let accessPassword = readStored(GATE_KEY);

// Distinguishes "wrong code" from a transport failure so only the former
// re-locks the page.
class AccessDenied extends Error {
  constructor(message) { super(message); this.name = "AccessDenied"; }
}

function showGate(message) {
  document.documentElement.setAttribute("data-gate", "");
  gateError.textContent = message || "";
  gateInput.focus();
}

function hideGate() {
  document.documentElement.removeAttribute("data-gate");
  gateError.textContent = "";
  gateInput.value = "";
  composerInput.focus();
}

gateForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const code = gateInput.value.trim();
  if (!code) return;
  gateBtn.disabled = true;
  gateError.textContent = "checking…";
  accessPassword = code;
  try {
    await getSession(true);
    try { sessionStorage.setItem(GATE_KEY, code); } catch (_) {}
    hideGate();
  } catch (err) {
    // getSession already re-locked and cleared on AccessDenied. Anything else
    // is verification or the network — surface the real reason rather than a
    // generic one, because "wrong code" and "Turnstile never answered" call
    // for completely different reactions from the user.
    gateError.textContent = err.message || "couldn't reach the server — try again";
  } finally {
    gateBtn.disabled = false;
  }
});

async function getSession(force = false) {
  if (sessionToken && !force) return sessionToken;
  if (force) { sessionToken = null; sessionPromise = null; }
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const turnstile_token = await getTurnstileToken();
      const res = await fetch(API_BASE + "/api/session", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...(turnstile_token ? { turnstile_token } : {}),
          ...(accessPassword ? { password: accessPassword } : {}),
        }),
      });
      // 401 is a bad/absent code, 429 is the brute-force lockout. Both mean
      // the code we hold won't work, so both send the user back to the gate.
      if (res.status === 401 || res.status === 429) {
        const info = await res.json().catch(() => ({}));
        throw new AccessDenied(info.error || "incorrect access code");
      }
      if (!res.ok) throw new Error(`could not start a session (${res.status})`);
      sessionToken = (await res.json()).token;
      return sessionToken;
    })().catch((e) => {
      sessionPromise = null;
      if (e instanceof AccessDenied) {
        accessPassword = "";
        clearStored(GATE_KEY);
        showGate(e.message);
      }
      throw e;
    });
  }
  return sessionPromise;
}

// A code carried over from an earlier page view redeems itself silently, so a
// reload mid-conversation doesn't stop to ask again.
(async function initGate() {
  if (!document.documentElement.hasAttribute("data-gate")) return;
  // Warm Turnstile while the user is still typing, so submitting the code
  // isn't also the first time the challenge script is fetched.
  if (TURNSTILE_SITEKEY) loadTurnstile().catch(() => {});
  if (!accessPassword) { gateInput.focus(); return; }
  gateError.textContent = "checking…";
  try {
    await getSession();
    hideGate();
  } catch (_) {
    // AccessDenied already re-rendered the gate; anything else leaves it up
    // with a prompt to retry, which is the right state either way.
    if (gateError.textContent === "checking…") gateError.textContent = "";
  }
})();

// A 429 is a normal operating state at capacity, not a fault. Marked so the
// renderer can present it as a wait rather than stamping a ⚠️ on it.
class Backpressure extends Error {
  constructor(message, retryAfter) {
    super(message);
    this.name = "Backpressure";
    this.retryAfter = retryAfter;
  }
}

async function postChat(body, signal) {
  const send = async (token) =>
    fetch(API_BASE + "/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Authorization": `Bearer ${token}`,
      },
      body: JSON.stringify(body),
      signal,
    });

  let res = await send(await getSession());
  // Sessions expire on their own schedule; one silent re-mint keeps a long
  // idle tab working instead of surfacing an auth error the user can't act on.
  if (res.status === 401) res = await send(await getSession(true));

  if (res.status === 429) {
    const info = await res.json().catch(() => ({}));
    const wait = parseInt(res.headers.get("Retry-After") || "0", 10);
    throw new Backpressure(info.error || "we're at capacity — try again shortly", wait);
  }
  if (!res.ok) {
    const errText = await res.text().catch(() => "");
    throw new Error(`server error ${res.status}: ${errText.slice(0, 200)}`);
  }
  return res;
}

// ─── python tool ──────────────────────────────────────────────────────────
// Executed here, not on the server. A container per call would put untrusted
// code on the box holding the weights and the sglang key, and would spend the
// same cores that run sglang's tokenizer — so one visitor's loop would slow
// token delivery for everyone. Here, a thousand users are a thousand browsers
// doing their own work and the worst case is someone freezing their own tab.
const PY_TIMEOUT_MS = 20000;
const TOOL_OUTPUT_CHARS = 6000;   // mirrors GATEWAY_TOOL_OUTPUT_CHARS
// Browser round trips per turn, which is a different quantity from the server's
// GATEWAY_MAX_TOOL_ITERATIONS — that one bounds the loop inside a single
// stream. This bounds how many times the stream may hand back to us. The real
// ceiling on tool work is the server's per-turn call budget, which now carries
// across these legs rather than resetting on each one.
const MAX_TOOL_ROUNDS = 5;

let pyWorker = null;
let pySeq = 0;

function pyWorkerGet() {
  if (!pyWorker) pyWorker = new Worker("/static/pyworker.js?v=stream40");
  return pyWorker;
}

function runPython(code) {
  return new Promise((resolve) => {
    const id = ++pySeq;
    const w = pyWorkerGet();
    let settled = false;

    const finish = (payload) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      w.removeEventListener("message", onmsg);
      w.removeEventListener("error", onerr);
      resolve(payload);
    };

    // WASM runs synchronously and ignores signals, so an infinite loop can only
    // be stopped by killing the worker. The next call boots a fresh one, losing
    // the warm interpreter — the right trade against a wedged tab.
    const timer = setTimeout(() => {
      try { w.terminate(); } catch (_) {}
      pyWorker = null;
      finish({ ok: false, output: `timed out after ${PY_TIMEOUT_MS / 1000}s` });
    }, PY_TIMEOUT_MS);

    const onmsg = (e) => { if (e.data && e.data.id === id) finish(e.data); };
    const onerr = (e) => finish({ ok: false, output: `worker error: ${e.message || "failed to load"}` });

    w.addEventListener("message", onmsg);
    w.addEventListener("error", onerr);
    w.postMessage({ id, code, maxChars: TOOL_OUTPUT_CHARS });
  });
}

// Run everything the model asked for and return the `tool` turns to send back.
// Concurrent on purpose, matching the server side: the model emits parallel
// calls, and serialising them here would waste that.
async function executeToolCalls(req, shell) {
  return Promise.all(req.calls.map(async (call) => {
    let args = {};
    try { args = JSON.parse(call.arguments || "{}"); } catch (_) {}
    // Same wire format the server-side rows use, so a turn that mixes a search
    // and a calculation reads as one transcript rather than two conventions.
    const row = shell.addTool(
      call.name,
      "<tool_call>\n"
        + JSON.stringify({ name: call.name, arguments: args }, null, 2)
        + "\n</tool_call>",
    );

    let res;
    if (call.name === "python") {
      res = await runPython(args.code || "");
    } else {
      // The server only forwards enabled tools, so this means the two sides
      // disagree about what exists — report it rather than hanging the turn.
      res = { ok: false, output: `unsupported tool: ${call.name}` };
    }

    // The turn sent back below is res.output verbatim; the wrapper is what the
    // template puts around it, so the row shows what the model will read.
    row.settle(res.ok, "<tool_response>\n" + (res.output || "") + "\n</tool_response>");
    return { role: "tool", tool_call_id: call.id, content: res.output };
  }));
}

// ─── streaming networking ─────────────────────────────────────────────────
async function sendMessage(text) {
  if (!text.trim() || pending) return;
  stopRequested = false;
  setPending(true);
  try {
    await _doSend(text);
  } finally {
    // outer guarantee: no matter what blows up inside, we unlock the UI
    setPending(false);
    currentAborter = null;
    composerInput.focus();
  }
}

async function _doSend(text) {
  // Before the turn is built, so the composer drops as the first message
  // appears rather than after the answer starts arriving.
  setEmpty(false);
  // From here the conversation has its settings and keeps them.
  setSettingsLocked(true);
  const epoch = chatEpoch;

  messages.push({ role: "user", content: text });
  const userEl = buildUserMessage(text);

  composerInput.value = "";
  autosize(composerInput);

  const shell = buildAssistantShell();
  pinTurnToTop(userEl, shell.el);

  const t0 = performance.now();
  let serverSeconds = null;
  let serverTokens = 0;
  let errored = false;
  let stopped = false;

  // hard wall in case the stream hangs. Vercel's own function still kills at
  // 60s regardless of this value; local app.py now allows much longer
  // generations (bumped for the 81920-token budget), so give it real room.
  const aborter = new AbortController();
  currentAborter = aborter;
  const abortTimer = setTimeout(() => aborter.abort(new Error("stream timeout (920s)")), 920_000);

  try {
    // A turn can take several legs when tools are involved: the model asks for
    // a call, the stream ends so the server can release its slot, the call runs
    // here, and we re-POST with the result appended. Everything renders into
    // the same assistant shell, so it reads as one answer.
    //
    // `sendMessages` is the conversation as the *server* sees it and includes
    // tool turns. `messages` deliberately does not keep them — see below.
    let sendMessages = messages;
    // Server-side tool rows, keyed by call id: opened by a tool_call frame and
    // settled by the matching tool_result later in the same stream.
    const serverToolRows = new Map();

    for (let round = 0; round <= MAX_TOOL_ROUNDS; round++) {
      let toolRequest = null;

      // No temperature in the body — the server's configured sampling defaults
      // apply, which is the only setting the model is actually tuned for.
      // Read per leg rather than captured once: a turn that hands off to the
      // browser and comes back should still reflect the toggle as it stands.
      const res = await postChat(
        { messages: sendMessages, tools: activeTools(),
          temperature: settings.temperature },
        aborter.signal);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      let streamDone = false;

      try {
        while (!streamDone) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });

          let idx;
          while ((idx = buf.indexOf("\n\n")) !== -1) {
            const frame = buf.slice(0, idx).trim();
            buf = buf.slice(idx + 2);
            if (!frame.startsWith("data:")) continue;
            const payload = frame.slice(5).trim();
            if (payload === "[DONE]") { streamDone = true; break; }

            let evt;
            try { evt = JSON.parse(payload); } catch { continue; }

            if (evt.type === "reasoning") {
              shell.addReasoning(evt.text || "");
            } else if (evt.type === "content") {
              shell.addContent(evt.text || "");
            } else if (evt.type === "tool_request") {
              toolRequest = evt;
            } else if (evt.type === "tool_call") {
              // A server-side tool. It runs and returns inside this same
              // stream, so both frames arrive here and the row is opened now
              // and settled a moment later by its tool_result.
              //
              // Shown in the wire format the model actually emitted rather than
              // as a tidied-up query string. `<tool_call>` is what the chat
              // template wraps this in, so the row reads as the transcript does.
              let args;
              try { args = JSON.parse(evt.arguments || "{}"); }
              catch (_) { args = evt.arguments || {}; }
              const call = "<tool_call>\n"
                + JSON.stringify({ name: evt.name, arguments: args }, null, 2)
                + "\n</tool_call>";
              serverToolRows.set(evt.id, shell.addTool(evt.name, call));
            } else if (evt.type === "tool_result") {
              const row = serverToolRows.get(evt.id);
              if (row) {
                // The full tool turn, exactly as the model received it —
                // expanding a search row shows the ten results it actually read,
                // not a count of them. `truncated` is marked because the model
                // saw the cut too, and an answer that trails off is usually
                // explained by it.
                const raw = evt.output
                  + (evt.truncated ? "\n… [truncated — the model saw this much]" : "");
                const body = evt.output
                  ? "<tool_response>\n" + raw + "\n</tool_response>"
                  : (evt.ok ? "done" : "failed");
                row.settle(evt.ok, body, evt.title);
                serverToolRows.delete(evt.id);
              }
            } else if (evt.type === "done") {
              serverSeconds = evt.seconds;
              serverTokens += evt.usage?.completion_tokens || 0;
              streamDone = true;
              break;
            } else if (evt.type === "error") {
              errored = true;
              shell.addContent(`\n\n⚠️ ${evt.message}`);
            }
          }
        }
      } finally {
        // release the connection immediately, don't wait for server EOF
        try { await reader.cancel(); } catch (_) {}
      }

      if (!toolRequest || errored || stopRequested) break;

      const toolTurns = await executeToolCalls(toolRequest, shell);
      if (stopRequested) break;
      // `turns` is everything the server appended on this leg — every assistant
      // turn it produced and every server-side tool result, not just the last
      // message. The browser is the only thing that survives the handover, so
      // dropping any of it means the model resumes without the search results
      // it asked the question about. (`assistant` is the older, narrower field;
      // kept as a fallback so a mid-flight turn against an older gateway still
      // completes.)
      sendMessages = sendMessages.concat(
        toolRequest.turns || [toolRequest.assistant],
        toolTurns,
      );
    }
  } catch (e) {
    // A user-initiated stop is not a failure — keep whatever streamed in and
    // mark it, rather than stamping a scary ⚠️ on a deliberate cancel.
    if (stopRequested || e?.name === "AbortError") {
      stopped = true;
    } else if (e instanceof Backpressure) {
      // Being told to wait is not an error state — drop the turn from history
      // so a retry re-sends it cleanly rather than duplicating the question.
      errored = true;
      const when = e.retryAfter ? ` try again in ${e.retryAfter}s.` : "";
      try { shell.addContent(`\n\n⏳ ${e.message}.${when}`); } catch (_) {}
      if (messages[messages.length - 1]?.role === "user") messages.pop();
    } else {
      errored = true;
      try { shell.addContent(`\n\n⚠️ ${e.message}`); } catch (_) {}
    }
  } finally {
    try {
      const seconds = serverSeconds ?? (performance.now() - t0) / 1000;
      const finalContent = shell.finalize({ tokens: serverTokens, seconds, errored, stopped });
      // partial answers still count as turns — the model should see what it
      // had already said if the user follows up
      //
      // Only the answer is kept, never the tool call and result turns. Those
      // would be re-sent and re-prefilled on every later turn, and tool output
      // is the largest thing in a conversation — the answer already states
      // whatever mattered from it. Dropping them is a deliberate context
      // decision, not an oversight: it trades the model's ability to re-read
      // raw tool output for a conversation that stays affordable.
      if (finalContent && !errored && epoch === chatEpoch) {
        messages.push({ role: "assistant", content: finalContent });
      }
    } catch (_) { /* never let render errors lock the UI */ }
    clearTimeout(abortTimer);
    currentAborter = null;
  }
}

// ─── event wiring ─────────────────────────────────────────────────────────
composerForm.addEventListener("submit", (e) => {
  e.preventDefault();
  sendMessage(composerInput.value);
});

// The send button doubles as stop while a response is streaming. Cancelling
// on click has to happen here rather than in the submit handler: preventing
// the default on the click is what stops the form submitting at all.
sendBtn.addEventListener("click", (e) => {
  if (!pending) return;
  e.preventDefault();
  abortCurrent();
});

// esc cancels from anywhere on the page, including from inside the composer
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && pending) {
    e.preventDefault();
    abortCurrent();
  }
});

// Refresh, close, or navigate away: tear the request down on the way out so
// the backend isn't left generating into a socket nobody is reading.
// pagehide covers reload/close/bfcache in every browser; beforeunload is a
// belt-and-braces fallback for older Safari.
window.addEventListener("pagehide", abortCurrent);
window.addEventListener("beforeunload", abortCurrent);

newChatBtn.addEventListener("click", () => {
  abortCurrent();
  resetChat();
  composerInput.focus();
});

composerInput.focus();
autosize(composerInput);
