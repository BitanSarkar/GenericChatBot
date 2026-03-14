/**
 * GenericChatBot — frontend
 *
 * SSE event protocol (POST /chat/stream):
 *   event: token    data: {"text": "..."}
 *   event: sources  data: {"sources": [{source, chunk_index, reranker_score, rrf_score}]}
 *   event: error    data: {"message": "..."}
 *   event: done     data: {}
 */

// ── DOM ───────────────────────────────────────────────────────────────────────
const messagesEl    = document.getElementById('messages');
const inputEl       = document.getElementById('questionInput');
const sendBtn       = document.getElementById('sendBtn');
const clearBtn      = document.getElementById('clearBtn');
const modeSelect    = document.getElementById('modeSelect');
const fileInput     = document.getElementById('fileInput');
const folderInput   = document.getElementById('folderInput');
const sidebar       = document.getElementById('sidebar');
const sidebarToggle = document.getElementById('sidebarToggle');
const statusDot     = document.getElementById('statusDot');
const statusText    = document.getElementById('statusMessage');
const fileListEl    = document.getElementById('fileList');
const chunkBadge    = document.getElementById('chunkCount');

// ── State ─────────────────────────────────────────────────────────────────────
let isChatting = false;

// ── Markdown + code + LaTeX renderer ─────────────────────────────────────────

// Configure marked once
marked.setOptions({
  breaks:   true,   // single newline → <br>
  gfm:      true,   // GitHub-flavoured markdown
  pedantic: false,
});

/**
 * Render markdown + code highlighting + LaTeX into a DOM element.
 * Call this AFTER streaming is complete (not during — too expensive per-token).
 */
function renderFull(el, text) {
  // 1. Remove the streaming class (restores normal whitespace handling)
  el.classList.remove('streaming');

  // 2. Markdown → HTML
  el.innerHTML = marked.parse(text);

  // 3. Syntax-highlight all code blocks
  el.querySelectorAll('pre code').forEach(block => {
    hljs.highlightElement(block);
  });

  // 4. KaTeX — render inline $...$ and display $$...$$
  if (typeof renderMathInElement !== 'undefined') {
    renderMathInElement(el, {
      delimiters: [
        { left: '$$',   right: '$$',   display: true  },
        { left: '$',    right: '$',    display: false },
        { left: '\\[', right: '\\]',  display: true  },
        { left: '\\(', right: '\\)',  display: false },
      ],
      throwOnError: false,
    });
  }
}

// ── Boot ──────────────────────────────────────────────────────────────────────
(async function init() {
  setupInput();
  setupSidebar();
  await loadHistory();
  pollStatus();
  setInterval(pollStatus, 2500);
})();

// ── Sidebar ───────────────────────────────────────────────────────────────────
function setupSidebar() {
  // Toggle
  sidebarToggle.addEventListener('click', () => sidebar.classList.toggle('hidden'));

  // File input
  fileInput.addEventListener('change',   () => uploadFiles(fileInput.files));
  folderInput.addEventListener('change', () => uploadFiles(folderInput.files));

  // Whole sidebar is the drag target
  sidebar.addEventListener('dragover',  e => { e.preventDefault(); sidebar.classList.add('drag-over'); });
  sidebar.addEventListener('dragleave', e => { if (!sidebar.contains(e.relatedTarget)) sidebar.classList.remove('drag-over'); });
  sidebar.addEventListener('drop', e => {
    e.preventDefault();
    sidebar.classList.remove('drag-over');
    uploadFiles(e.dataTransfer.files);
  });
}

// ── Input & send ──────────────────────────────────────────────────────────────
function setupInput() {
  // Auto-resize textarea
  inputEl.addEventListener('input', () => {
    inputEl.style.height = 'auto';
    inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + 'px';
  });

  inputEl.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });

  sendBtn.addEventListener('click', sendMessage);

  clearBtn.addEventListener('click', async () => {
    if (!confirm('Clear conversation history?')) return;
    await fetch('/history', { method: 'DELETE' });
    [...messagesEl.querySelectorAll('.msg:not(#welcomeMsg)')].forEach(el => el.remove());
  });
}

// ── Send ──────────────────────────────────────────────────────────────────────
async function sendMessage() {
  const q = inputEl.value.trim();
  if (!q || isChatting) return;

  const mode = modeSelect.value;

  // Clear input
  inputEl.value = '';
  inputEl.style.height = 'auto';

  // Remove welcome on first real message
  document.getElementById('welcomeMsg')?.remove();

  addUserMsg(q);
  const { rowEl, bodyEl } = addBotPlaceholder();

  isChatting       = true;
  sendBtn.disabled = true;

  let fullText     = '';
  let typingGone   = false;
  let currentEvent = '';

  try {
    const res = await fetch('/chat/stream', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ question: q, mode }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let   buf     = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();

      for (const line of lines) {
        if (line.startsWith('event: ')) {
          currentEvent = line.slice(7).trim();
        } else if (line.startsWith('data: ')) {
          const payload = JSON.parse(line.slice(6));

          if (currentEvent === 'token') {
            if (!typingGone) {
              bodyEl.innerHTML = '';
              bodyEl.classList.add('streaming'); // plain text while streaming
              typingGone = true;
            }
            fullText += payload.text;
            bodyEl.textContent = fullText;         // fast raw text during stream
            scrollBottom();

          } else if (currentEvent === 'sources') {
            renderSources(rowEl, payload.sources);

          } else if (currentEvent === 'error') {
            rowEl.remove();
            addErrorMsg(payload.message);

          } else if (currentEvent === 'done') {
            renderFull(bodyEl, fullText);           // full render once complete
            scrollBottom();
          }
          currentEvent = '';
        }
      }
    }

    if (!typingGone) bodyEl.innerHTML = '<span class="muted">No response.</span>';

  } catch (e) {
    rowEl.remove();
    addErrorMsg(e.message || 'Request failed');
  } finally {
    isChatting       = false;
    sendBtn.disabled = false;
    inputEl.focus();
    scrollBottom();
  }
}

// ── Message builders ──────────────────────────────────────────────────────────
function addUserMsg(text) {
  const row = document.createElement('div');
  row.className = 'msg user';
  row.innerHTML = `<div class="msg-body">${esc(text)}</div>`;
  messagesEl.appendChild(row);
  scrollBottom();
}

function addBotPlaceholder() {
  const rowEl  = document.createElement('div');
  rowEl.className = 'msg bot';
  const bodyEl = document.createElement('div');
  bodyEl.className = 'msg-body';
  bodyEl.innerHTML = '<div class="typing"><span></span><span></span><span></span></div>';
  rowEl.appendChild(bodyEl);
  messagesEl.appendChild(rowEl);
  scrollBottom();
  return { rowEl, bodyEl };
}

function addBotMsg(text) {
  const row  = document.createElement('div');
  row.className = 'msg bot';
  const body = document.createElement('div');
  body.className = 'msg-body';
  renderFull(body, text);     // full markdown + code + LaTeX
  row.appendChild(body);
  messagesEl.appendChild(row);
  scrollBottom();
  return row;
}

function addErrorMsg(text) {
  const row = document.createElement('div');
  row.className = 'msg error';
  row.innerHTML = `<div class="msg-body">${esc(text)}</div>`;
  messagesEl.appendChild(row);
  scrollBottom();
}

function renderSources(rowEl, sources) {
  if (!sources || sources.length === 0) return;

  const btn  = document.createElement('button');
  btn.className = 'sources-btn';
  btn.innerHTML = `<span class="arrow">▶</span> ${sources.length} source${sources.length !== 1 ? 's' : ''}`;

  const list = document.createElement('div');
  list.className = 'sources-list';
  list.innerHTML = sources.map(s => `
    <div class="src-row">
      <span class="src-name">${esc(s.source)}</span>
      <span class="src-tag">chunk ${s.chunk_index}</span>
      <span class="src-tag">${s.reranker_score.toFixed(2)}</span>
    </div>`).join('');

  btn.addEventListener('click', () => {
    const open = list.classList.toggle('open');
    btn.classList.toggle('open', open);
  });

  const body = rowEl.querySelector('.msg-body');
  body.appendChild(btn);
  body.appendChild(list);
}

// ── History ───────────────────────────────────────────────────────────────────
async function loadHistory() {
  try {
    const res  = await fetch('/history');
    const data = await res.json();
    if (!data.history?.length) return;

    document.getElementById('welcomeMsg')?.remove();
    for (const t of data.history) {
      if (t.role === 'user') addUserMsg(t.content);
      else                   addBotMsg(t.content);
    }
    scrollBottom();
  } catch { /* server not ready yet */ }
}

// ── Upload ────────────────────────────────────────────────────────────────────
async function uploadFiles(files) {
  if (!files?.length) return;

  setStatus('running', `Uploading ${files.length} file(s)…`);

  const fd = new FormData();
  for (const f of files) {
    fd.append('files', f, f.webkitRelativePath || f.name);
  }

  try {
    const res  = await fetch('/upload', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    setStatus('running', 'Indexing…');
  } catch (e) {
    setStatus('error', `Upload failed: ${e.message}`);
  } finally {
    fileInput.value   = '';
    folderInput.value = '';
  }
}

// ── Status ────────────────────────────────────────────────────────────────────
async function pollStatus() {
  try {
    const res  = await fetch('/index/status');
    const data = await res.json();
    setStatus(data.status, data.message);
    renderFiles(data.files);
    chunkBadge.textContent = data.chunk_count > 0 ? `${data.chunk_count.toLocaleString()} chunks` : '';
  } catch { /* ignore */ }
}

function setStatus(status, message) {
  statusDot.className  = `status-dot ${status}`;
  statusText.textContent = message || status;
}

function renderFiles(files) {
  fileListEl.innerHTML = '';
  if (!files?.length) {
    fileListEl.innerHTML = '<li class="file-empty">No files yet</li>';
    return;
  }
  for (const f of files) {
    const li = document.createElement('li');
    li.textContent = f;
    li.title = f;
    fileListEl.appendChild(li);
  }
}

// ── Util ──────────────────────────────────────────────────────────────────────
function esc(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function scrollBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}
