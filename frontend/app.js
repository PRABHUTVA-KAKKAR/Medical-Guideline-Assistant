const queryEl = document.getElementById("query");
const askBtn = document.getElementById("ask");
const clearBtn = document.getElementById("clear");
const retryBtn = document.getElementById("retry");
const emptyEl = document.getElementById("empty");
const skelEl = document.getElementById("skeleton");
const errEl = document.getElementById("error");
const errText = document.getElementById("errorText");
const resultEl = document.getElementById("result");
const verdictEl = document.getElementById("verdict");
const confEl = document.getElementById("confidence");
const meterFill = document.getElementById("meterFill");
const answerEl = document.getElementById("answerText");
const citesEl = document.getElementById("cites");
const discEl = document.getElementById("disclaimer");
const pillEl = document.getElementById("indexPill");
const docList = document.getElementById("docList");
const docFilter = document.getElementById("docFilter");
const docCount = document.getElementById("docCount");

let docs = [];
let lastQuery = "";

function setState(name) {
  emptyEl.hidden = name !== "empty";
  skelEl.hidden = name !== "loading";
  errEl.hidden = name !== "error";
  resultEl.hidden = name !== "result";
  askBtn.disabled = name === "loading";
}

function verdictClass(v) {
  if (v === "ANSWER") return "verdict verdict-ok";
  if (v === "Idle") return "verdict verdict-idle";
  return "verdict verdict-warn";
}

function renderAnswer(body) {
  verdictEl.textContent = body.verdict;
  verdictEl.className = verdictClass(body.verdict);
  const pct = Math.round((body.confidence || 0) * 100);
  confEl.textContent = "Confidence " + pct + " percent. Verdict " + body.verdict + ".";
  meterFill.style.width = pct + "%";
  answerEl.innerHTML = "";
  for (const para of String(body.answer || "").split("\n").filter(Boolean)) {
    const p = document.createElement("p");
    const parts = para.split(/(\[[^\]]+\])/g);
    for (const part of parts) {
      if (/^\[[^\]]+\]$/.test(part)) {
        const s = document.createElement("span");
        s.className = "cite";
        s.textContent = part;
        p.appendChild(s);
      } else {
        p.appendChild(document.createTextNode(part));
      }
    }
    answerEl.appendChild(p);
  }
  citesEl.innerHTML = "";
  for (const c of body.citations || []) {
    const li = document.createElement("li");
    const head = document.createElement("div");
    const id = document.createElement("span");
    id.className = "cite-id";
    id.textContent = c.chunk_id;
    head.appendChild(id);
    head.appendChild(document.createTextNode("  " + (c.title || c.doc_id || "")));
    const meta = document.createElement("div");
    meta.className = "cite-meta";
    meta.textContent = (c.doc_id || "") + (c.section ? "  " + c.section : "") + (c.source ? "  " + c.source : "");
    li.appendChild(head);
    li.appendChild(meta);
    if (c.text) {
      const t = document.createElement("div");
      t.className = "cite-text";
      t.textContent = String(c.text).slice(0, 280);
      li.appendChild(t);
    }
    citesEl.appendChild(li);
  }
  discEl.textContent = body.disclaimer || "";
}

async function ask(q) {
  const query = (q !== undefined ? q : queryEl.value).trim();
  if (!query) {
    queryEl.focus();
    return;
  }
  lastQuery = query;
  setState("loading");
  try {
    const r = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, k: 8 }),
    });
    if (!r.ok) throw new Error("Server returned " + r.status);
    renderAnswer(await r.json());
    setState("result");
  } catch (e) {
    errText.textContent = "Could not reach the guideline index. Check that the API is running, then retry. Detail: " + e.message;
    setState("error");
  }
}

function renderDocs(filter) {
  const f = (filter || "").toLowerCase();
  const rows = docs.filter((d) => (d.title || "").toLowerCase().includes(f));
  docCount.textContent = rows.length + " of " + docs.length + " MOHFW documents indexed.";
  docList.innerHTML = "";
  for (const d of rows.slice(0, 200)) {
    const li = document.createElement("li");
    const t = document.createElement("div");
    t.className = "doc-title";
    t.textContent = d.title || d.doc_id;
    const m = document.createElement("div");
    m.className = "doc-meta";
    m.textContent = (d.doc_id || "") + "  " + (d.num_chunks || 0) + " chunks";
    li.appendChild(t);
    li.appendChild(m);
    docList.appendChild(li);
  }
}

async function boot() {
  setState("empty");
  document.getElementById("examples").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-q]");
    if (b) {
      queryEl.value = b.dataset.q;
      ask(b.dataset.q);
    }
  });
  askBtn.addEventListener("click", () => ask());
  clearBtn.addEventListener("click", () => {
    queryEl.value = "";
    setState("empty");
    queryEl.focus();
  });
  retryBtn.addEventListener("click", () => ask(lastQuery));
  queryEl.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") ask();
  });
  docFilter.addEventListener("input", () => renderDocs(docFilter.value));
  try {
    const h = await (await fetch("/api/health")).json();
    pillEl.textContent = h.doc_count + " docs. " + h.chunk_count + " chunks. Index " + h.index_version + ".";
    const d = await (await fetch("/api/docs-list")).json();
    docs = Array.isArray(d) ? d : [];
    renderDocs("");
  } catch (e) {
    pillEl.textContent = "Index unavailable. Start the API.";
  }
}

boot();
