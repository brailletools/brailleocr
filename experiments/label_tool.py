#!/usr/bin/env python3
"""
Correction-based labeling tool for Braille pages.

Labeling every cell on a page by hand is hours of work; correcting the handful
the model gets wrong is minutes. This runs the detector + classifier, then emits
a self-contained HTML page where anything you DON'T touch counts as confirmed
correct. Ground truth is reconstructed later as predictions + your corrections.

That inverts the usual cost: effort scales with the model's error rate, not with
page length. It also means the labeled set improves fastest exactly where the
model is worst, which is where we need data (phone photos, natural scenes).

Bias warning, which matters when this data is used for evaluation: confirming
predictions by silence inherits the model's blind spots. A cell the model never
proposed AND the labeler never notices stays missing from ground truth, so
recall measured against this data is optimistic. Mitigation: the tool draws
expected-but-empty grid positions in a distinct colour so gaps are visible, and
adding a missed cell is one click. Spot-check a few pages against a full manual
label before trusting recall numbers from this set.

The output HTML has no network dependencies (image is embedded) and works
offline in any browser; corrections download as JSON.

Usage:
  python experiments/label_tool.py path/to/page.jpg
  python experiments/label_tool.py page.jpg --out /tmp/label --open
"""

import argparse
import base64
import datetime
import io
import json
import statistics
import sys
from pathlib import Path

import PIL.Image
import PIL.ImageOps
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipeline                                                 # noqa: E402
from dot_pattern_utils import TILE_SIZE                         # noqa: E402
from model_fetch import resolve_model, resolve_classifier_path  # noqa: E402

MAX_EMBED_PX = 2400   # downscale for display only; coordinates stay in original px


def predict(img, model, max_det, use_classifier=True):
    """Core path only: scale-normalized tiled detection + classifier. No recovery
    stages — experiment 5 showed they contribute no real cells, and a labeler
    should not be asked to delete 50 invented cells per page."""
    cells = pipeline.run_detection_tiled(img, model, max_det, TILE_SIZE)
    if len(cells) >= pipeline.CONTAINER_MIN_CANDIDATES:
        hi = [c for c in cells if c['conf'] >= pipeline.HIGH_CONF]
        sample = hi if len(hi) >= pipeline.CONTAINER_MIN_CANDIDATES else cells
        median_w = statistics.median(c['w'] for c in sample)
        native = max(pipeline.MIN_NATIVE_TILE,
                     round(median_w * TILE_SIZE / pipeline.TARGET_CELL_PX))
        if not 0.67 <= native / TILE_SIZE <= 1.5:
            cells = pipeline.run_detection_tiled(img, model, max_det, native)

    cells = [c for c in cells if c['conf'] >= pipeline.HIGH_CONF]
    if use_classifier and cells:
        clf, dev, tf = pipeline.load_cell_classifier(resolve_classifier_path())
        cells = pipeline.reclassify_cells(img, cells, clf, dev, tf)
    return cells


HTML = r'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Braille correction — __NAME__</title>
<style>
  :root { --bg:#faf9f7; --fg:#1a1a1a; --panel:#fff; --line:#d8d4cd;
          --ok:#2d6a4f; --todo:#7c3aed; --edit:#b45309; --del:#b91c1c; --add:#1d4ed8; }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
    --bg:#16151a; --fg:#ececec; --panel:#22212a; --line:#3a3845; } }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif; }
  header { position:sticky; top:0; z-index:30; background:var(--panel);
           border-bottom:1px solid var(--line); padding:10px 16px;
           display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
  h1 { font-size:15px; margin:0; font-weight:600; }
  .count { font-variant-numeric:tabular-nums; }
  .nav a, .nav span.disabled { padding:5px 9px; border:1px solid var(--line);
      border-radius:6px; text-decoration:none; color:inherit; font-size:13px; }
  .nav span.disabled { opacity:.4; }
  .todo-chip { color:var(--todo); font-weight:600; }
  button { font:inherit; padding:6px 12px; border:1px solid var(--line);
           border-radius:6px; background:var(--panel); color:var(--fg); cursor:pointer; }
  button.primary { background:var(--ok); color:#fff; border-color:var(--ok); }
  .legend { display:flex; gap:14px; font-size:12px; opacity:.85; margin:8px 16px 0; }
  .legend i { display:inline-block; width:11px; height:11px; border:2px solid;
              border-radius:2px; margin-right:4px; vertical-align:-1px; }
  .hint { font-size:12.5px; opacity:.8; margin:6px 16px 10px; }
  kbd { border:1px solid var(--line); border-radius:4px; padding:0 4px; font-size:11px; }
  #stage { position:relative; margin:0 16px 40px; width:max-content;
           cursor:crosshair; user-select:none; }
  #stage img { display:block; pointer-events:none; }
  .cell { position:absolute; border:2px solid var(--ok); border-radius:2px;
          background:transparent; padding:0; cursor:move; }
  .cell:focus-visible { outline:3px solid #2563eb; outline-offset:2px; }
  .cell.todo { border-color:var(--todo); background:rgba(124,58,237,.14);
               border-style:dashed; }
  .cell.edited { border-color:var(--edit); background:rgba(180,83,9,.16); }
  .cell.deleted { border-color:var(--del); background:rgba(185,28,28,.26);
                  border-style:dashed; }
  .cell.added { border-color:var(--add); background:rgba(29,78,216,.16); }
  .cell .grip { position:absolute; right:-5px; bottom:-5px; width:11px; height:11px;
                background:var(--panel); border:2px solid currentColor;
                border-radius:2px; cursor:nwse-resize; }
  .cell .tag { position:absolute; left:0; top:-16px; font-size:11px;
               line-height:1; background:var(--panel); padding:1px 3px;
               border-radius:3px; pointer-events:none; }
  /* The model's reading, drawn ON the cell at the six canonical dot positions,
     so a wrong dot is visible against the photo underneath instead of having to
     be decoded from a glyph. Filled = model says raised; faint ring = not. */
  .cell .d { position:absolute; width:26%; height:16%; border-radius:50%;
             transform:translate(-50%,-50%); pointer-events:none;
             box-sizing:border-box; }
  .cell .d.on  { background:#e11d48; box-shadow:0 0 0 1px #fff; }
  .cell .d.off { border:1px solid rgba(225,29,72,.45); }
  /* The classifier's own uncertainty: a dot whose probability sits in the
     ambiguous band gets a ring, so review attention goes where the model is
     least sure rather than being spread evenly over confident cells. */
  .cell .d.unsure { box-shadow:0 0 0 2px #f59e0b, 0 0 0 3px rgba(0,0,0,.35); }
  .cell.has-unsure { border-color:#f59e0b; }
  body.hide-dots .cell .d { display:none; }
  #rubber { position:absolute; border:2px dashed var(--add);
            background:rgba(29,78,216,.12); display:none; pointer-events:none; }
  #editor { position:absolute; z-index:40; background:var(--panel);
            border:1px solid var(--line); border-radius:10px; padding:12px;
            box-shadow:0 8px 28px rgba(0,0,0,.28); display:none; }
  .dots { display:grid; grid-template-columns:repeat(2,38px); gap:6px; }
  .dot { width:38px; height:38px; border-radius:50%; border:2px solid var(--line);
         background:transparent; font-size:11px; color:var(--fg); cursor:pointer; }
  .dot[aria-pressed="true"] { background:var(--fg); color:var(--bg); }
  .preview { font-size:32px; text-align:center; margin:6px 0; min-height:38px; }
  .row { display:flex; gap:8px; margin-top:10px; }
</style></head><body>
<header>
  <h1>__NAME__</h1>
  <span class="nav">__NAV__</span>
  <span class="count" id="tally"></span>
  <button id="nextTodo">Next uncertain →</button>
  <button id="toggleDots" aria-pressed="true">Hide dot overlay</button>
  <button id="undo">Undo</button>
  <button id="zoomOut">−</button><button id="zoomIn">+</button>
  <button id="scope" aria-pressed="false">Not embossed Braille</button>
  <button id="save">Download corrections</button>
  <button id="copy">Copy JSON</button>
  <button id="saveNext" class="primary">Save &amp; next →</button>
  <span id="status" class="count"></span>
</header>
<div class="legend">
  <span><i style="border-color:var(--todo)"></i>needs a dot pattern</span>
  <span><i style="border-color:var(--ok)"></i>confirmed correct</span>
  <span><i style="border-color:var(--edit)"></i>you edited</span>
  <span><i style="border-color:var(--add)"></i>you added</span>
  <span><i style="border-color:var(--del)"></i>marked not-a-cell</span>
</div>
<p class="hint">
  Red markers show <strong>which dots the model read</strong> — filled = raised.
  An <span style="color:#b45309;font-weight:600">amber ring</span> means the model
  was unsure about that dot; those are worth checking first (<kbd>n</kbd>).
  Compare the markers against the photo underneath.
  <strong>Click a box</strong> to set its dots · <strong>drag a box</strong> to move it ·
  <strong>drag its corner</strong> to resize · <strong>drag on blank paper</strong> to draw a
  new cell. In the editor: <kbd>1</kbd>–<kbd>6</kbd> toggle dots, <kbd>Enter</kbd> save,
  <kbd>Esc</kbd> cancel, <kbd>x</kbd> not-a-cell. Anything you leave green is recorded as correct.
</p>
<div id="stage">
  <img id="page" src="data:image/jpeg;base64,__IMG__" alt="Braille page being corrected">
  <div id="rubber"></div>
  <div id="editor" role="dialog" aria-label="Edit cell dots">
    <div class="preview" id="preview">⠿</div>
    <div class="dots" id="dots"></div>
    <div class="row">
      <button id="ok" class="primary">Save</button>
      <button id="notcell">Not a cell</button>
      <button id="cancel">Cancel</button>
    </div>
  </div>
</div>
<script>
const DATA = __DATA__;
const S = DATA.display_scale;
let zoom = 1, cur = null, bits = "000000";
const edits = new Map(), deleted = new Set(), added = [], moved = new Map();
const undoStack = [];
const stage = document.getElementById('stage');
const editor = document.getElementById('editor');
const rubber = document.getElementById('rubber');
const dotsEl = document.getElementById('dots');
const BLANK = '000000';
const braille = b => String.fromCharCode(0x2800 + [...b].reduce(
  (a, ch, i) => a + (ch === '1' ? [1,2,4,8,16,32][i] : 0), 0));
const toImg = (clientX, clientY) => {
  const r = document.getElementById('page').getBoundingClientRect();
  return { x: (clientX - r.left) / (S * zoom), y: (clientY - r.top) / (S * zoom) };
};

[[0,'1'],[3,'4'],[1,'2'],[4,'5'],[2,'3'],[5,'6']].forEach(([idx,label]) => {
  const b = document.createElement('button');
  b.className = 'dot'; b.textContent = label; b.dataset.idx = idx;
  b.setAttribute('aria-pressed','false'); b.setAttribute('aria-label','dot ' + label);
  b.onclick = e => { e.stopPropagation(); toggle(idx); };
  dotsEl.appendChild(b);
});
function toggle(i) {
  bits = bits.substring(0,i) + (bits[i]==='1'?'0':'1') + bits.substring(i+1);
  syncEditor();
}
function syncEditor() {
  document.getElementById('preview').textContent = braille(bits);
  if (cur) {           // show the pending pattern on the cell itself
    cur.querySelectorAll('.d').forEach(d => {
      const on = bits[+d.dataset.d] === '1';
      d.classList.toggle('on', on);
      d.classList.toggle('off', !on);
    });
  }
  dotsEl.querySelectorAll('.dot').forEach(d =>
    d.setAttribute('aria-pressed', bits[+d.dataset.idx]==='1' ? 'true':'false'));
}

function classFor(c, el) {
  el.classList.toggle('todo', !el.classList.contains('deleted') &&
                              !el.classList.contains('added') &&
                              !edits.has(c.id) && c.bits === BLANK);
}
function styleCell(el) {
  const c = el._cell;
  el.style.left   = ((c.cx - c.w/2) * S * zoom) + 'px';
  el.style.top    = ((c.cy - c.h/2) * S * zoom) + 'px';
  el.style.width  = (c.w * S * zoom) + 'px';
  el.style.height = (c.h * S * zoom) + 'px';
  const tag = el.querySelector('.tag');
  tag.textContent = c.bits === BLANK ? '?' : braille(c.bits);
  let anyUnsure = false;
  el.querySelectorAll('.d').forEach(d => {
    const i = +d.dataset.d, on = c.bits[i] === '1';
    d.classList.toggle('on', on);
    d.classList.toggle('off', !on);
    const p = c.probs ? c.probs[i] : null;
    const unsure = p != null && p > 0.2 && p < 0.8 && !edits.has(c.id);
    d.classList.toggle('unsure', !!unsure);
    if (unsure) anyUnsure = true;
    if (p != null) d.title = 'dot ' + (i+1) + ': p=' + p.toFixed(2);
  });
  el.classList.toggle('has-unsure', anyUnsure);
  classFor(c, el);
}
function draw() {
  document.getElementById('page').style.width = (DATA.display_w * zoom) + 'px';
  document.querySelectorAll('.cell').forEach(styleCell);
}
function tally() {
  const todo = [...document.querySelectorAll('.cell')].filter(
    el => el._cell.bits === BLANK && !el.classList.contains('deleted')).length;
  document.getElementById('tally').innerHTML =
    `${edits.size} edited · ${added.length} added · ${deleted.size} removed · ` +
    `${moved.size} moved` + (todo ? ` · <span class="todo-chip">${todo} unlabeled</span>` : '') +
    (() => { const u = document.querySelectorAll('.cell.has-unsure').length;
             return u ? ` · <span style="color:#b45309;font-weight:600">${u} uncertain</span>` : ''; })();
}
function makeCell(c, cls) {
  const el = document.createElement('button');
  el.className = 'cell' + (cls ? ' ' + cls : '');
  el._cell = c;
  // dot1..dot6 at (x%, y%): left column 30%, right 70%; rows 20/50/80%
  const POS = [[30,20],[30,50],[30,80],[70,20],[70,50],[70,80]];
  el.innerHTML = '<span class="tag"></span><span class="grip"></span>' +
    POS.map(([x,y],i) => `<span class="d" data-d="${i}" ` +
                         `style="left:${x}%;top:${y}%"></span>`).join('');
  el.setAttribute('aria-label', 'braille cell, click to set dots');
  el.addEventListener('mousedown', e => startCellDrag(e, el));
  stage.appendChild(el);
  styleCell(el);
  return el;
}

// ── dragging: move a cell, resize via its grip, or rubber-band a new cell ──
let drag = null;
function startCellDrag(e, el) {
  e.stopPropagation(); e.preventDefault();
  const p = toImg(e.clientX, e.clientY);
  drag = { el, mode: e.target.classList.contains('grip') ? 'resize' : 'move',
           x0: p.x, y0: p.y, orig: { ...el._cell }, movedFar: false };
}
stage.addEventListener('mousedown', e => {
  if (e.target.closest('#editor') || e.target.closest('.cell')) return;
  close();
  const p = toImg(e.clientX, e.clientY);
  drag = { mode: 'new', x0: p.x, y0: p.y, movedFar: false };
});
window.addEventListener('mousemove', e => {
  if (!drag) return;
  const p = toImg(e.clientX, e.clientY);
  const dx = p.x - drag.x0, dy = p.y - drag.y0;
  if (Math.abs(dx) > 2 || Math.abs(dy) > 2) drag.movedFar = true;
  if (drag.mode === 'new') {
    rubber.style.display = 'block';
    rubber.style.left   = (Math.min(drag.x0, p.x) * S * zoom) + 'px';
    rubber.style.top    = (Math.min(drag.y0, p.y) * S * zoom) + 'px';
    rubber.style.width  = (Math.abs(dx) * S * zoom) + 'px';
    rubber.style.height = (Math.abs(dy) * S * zoom) + 'px';
  } else if (drag.mode === 'move') {
    drag.el._cell.cx = drag.orig.cx + dx;
    drag.el._cell.cy = drag.orig.cy + dy;
    styleCell(drag.el);
  } else {
    drag.el._cell.w = Math.max(6, drag.orig.w + dx * 2);
    drag.el._cell.h = Math.max(6, drag.orig.h + dy * 2);
    styleCell(drag.el);
  }
});
window.addEventListener('mouseup', e => {
  if (!drag) return;
  const d = drag; drag = null;
  rubber.style.display = 'none';
  if (d.mode === 'new') {
    const p = toImg(e.clientX, e.clientY);
    // A click (no drag) drops a median-sized cell; a drag uses the drawn box.
    const w = d.movedFar ? Math.abs(p.x - d.x0) : DATA.median_w;
    const h = d.movedFar ? Math.abs(p.y - d.y0) : DATA.median_h;
    const c = { cx: d.movedFar ? (d.x0 + p.x)/2 : d.x0,
                cy: d.movedFar ? (d.y0 + p.y)/2 : d.y0,
                w, h, bits: BLANK, id: null };
    added.push(c);
    const el = makeCell(c, 'added');
    undoStack.push(() => { added.splice(added.indexOf(c),1); el.remove(); });
    openEditor(el); tally();
  } else if (d.movedFar) {
    const c = d.el._cell;
    if (c.id != null) { moved.set(c.id, { cx:c.cx, cy:c.cy, w:c.w, h:c.h }); }
    const before = d.orig, el = d.el;
    undoStack.push(() => { Object.assign(el._cell, before);
                           if (before.id != null) moved.delete(before.id);
                           styleCell(el); tally(); });
    tally();
  } else {
    openEditor(d.el);   // a plain click opens the dot editor
  }
});

function openEditor(el) {
  cur = el; bits = el._cell.bits; syncEditor();
  editor.style.display = 'block';
  const c = el._cell;
  const maxLeft = DATA.display_w * zoom - editor.offsetWidth - 8;
  editor.style.left = Math.max(0, Math.min((c.cx + c.w) * S * zoom, maxLeft)) + 'px';
  editor.style.top  = Math.max(0, (c.cy - c.h) * S * zoom) + 'px';
  document.getElementById('ok').focus();
}
function close() {
  if (cur) styleCell(cur);   // discard any live preview not saved
  editor.style.display = 'none'; cur = null;
}

document.getElementById('ok').onclick = () => {
  if (!cur) return;
  const c = cur._cell, before = c.bits, el = cur;
  c.bits = bits;
  if (c.id != null) { edits.set(c.id, bits); el.classList.add('edited'); }
  undoStack.push(() => { c.bits = before;
                         if (c.id != null && before === BLANK) edits.delete(c.id);
                         el.classList.toggle('edited', edits.has(c.id));
                         styleCell(el); tally(); });
  styleCell(el); close(); tally();
};
document.getElementById('notcell').onclick = () => {
  if (!cur) return;
  const c = cur._cell, el = cur;
  if (c.id != null) { deleted.add(c.id); el.classList.add('deleted');
                      undoStack.push(() => { deleted.delete(c.id);
                                             el.classList.remove('deleted'); tally(); }); }
  else { const i = added.indexOf(c); if (i>=0) added.splice(i,1); el.remove(); }
  close(); tally();
};
document.getElementById('cancel').onclick = close;
document.getElementById('undo').onclick = () => {
  const fn = undoStack.pop(); if (fn) { fn(); tally(); }
};
document.getElementById('nextTodo').onclick = () => {
  // Blank cells first (nothing predicted), then cells the classifier was
  // unsure about — the two places a labeler's time is worth most.
  const cells = [...document.querySelectorAll('.cell')].filter(
    el => !el.classList.contains('deleted'));
  const el = cells.find(el => el._cell.bits === BLANK)
          || cells.find(el => el.classList.contains('has-unsure'));
  if (!el) { alert('Nothing blank or uncertain left on this page.'); return; }
  el.scrollIntoView({ block:'center', behavior:'smooth' });
  openEditor(el);
};
document.addEventListener('keydown', e => {
  if (e.key === 'z' && (e.metaKey || e.ctrlKey)) {
    e.preventDefault(); document.getElementById('undo').click(); return; }
  if (editor.style.display !== 'block') {
    if (e.key === 'n') document.getElementById('nextTodo').click();
    return;
  }
  if (e.key >= '1' && e.key <= '6') { toggle(+e.key - 1); e.preventDefault(); }
  else if (e.key === 'Enter') document.getElementById('ok').click();
  else if (e.key === 'Escape') close();
  else if (e.key === 'x') document.getElementById('notcell').click();
});
document.getElementById('toggleDots').onclick = e => {
  const hidden = document.body.classList.toggle('hide-dots');
  e.target.textContent = hidden ? 'Show dot overlay' : 'Hide dot overlay';
  e.target.setAttribute('aria-pressed', hidden ? 'false' : 'true');
};
document.getElementById('zoomIn').onclick  = () => { zoom = Math.min(6, zoom*1.25); draw(); };
document.getElementById('zoomOut').onclick = () => { zoom = Math.max(.25, zoom/1.25); draw(); };
function payload() {
  return {
    image: DATA.image, image_size: [DATA.width, DATA.height],
    generated: DATA.generated, tool: 'label_tool.py',
    note: 'Ground truth = predictions, with edits and moves applied, deletions removed, additions included. Untouched entries are confirmed correct by the labeler. Cells still blank (000000) are NOT labeled.',
    out_of_scope: outOfScope,
    predictions: DATA.cells,
    corrections: {
      edited: [...edits].map(([id, b]) => ({ id, bits: b })),
      deleted: [...deleted],
      moved: [...moved].map(([id, box]) => ({ id, ...box })),
      added: added.map(c => ({ cx:c.cx, cy:c.cy, w:c.w, h:c.h, bits:c.bits })),
    },
  };
}
function flash(msg, bad) {
  const el = document.getElementById('status');
  el.textContent = msg;
  el.style.color = bad ? '#b91c1c' : 'var(--ok)';
}
// One download path, called from exactly one place. An earlier version fired
// twice per click (a direct call plus a synthetic click event), which trips
// Chrome's repeated-download block and then silently kills every later save.
// The anchor is attached to the document before clicking: a detached anchor is
// ignored outright by Safari.
let saved = false;
function download() {
  const json = JSON.stringify(payload(), null, 2);
  try {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([json], { type:'application/json' }));
    a.download = DATA.image.replace(/\.[^.]+$/, '') + '.corrections.json';
    a.style.display = 'none';
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 2000);
    saved = true;
    flash('Saved ' + a.download + ' — move it into labels/corrections/');
    try {
      const KEY = 'brailleLabelDone';
      const done = JSON.parse(localStorage.getItem(KEY) || '[]');
      if (!done.includes(DATA.image)) done.push(DATA.image);
      localStorage.setItem(KEY, JSON.stringify(done));
    } catch (e) { /* file:// storage may be blocked; progress just won't persist */ }
  } catch (e) {
    flash('Download blocked — use “Copy JSON” instead', true);
    showJson(json);
  }
  return saved;
}
function showJson(json) {
  let box = document.getElementById('jsonbox');
  if (!box) {
    box = document.createElement('textarea');
    box.id = 'jsonbox';
    box.style.cssText = 'position:fixed;right:12px;bottom:12px;width:380px;' +
      'height:220px;z-index:60;font:11px ui-monospace,monospace;';
    document.body.appendChild(box);
  }
  box.value = json; box.select();
}
document.getElementById('save').onclick = () => download();
document.getElementById('copy').onclick = () => {
  const json = JSON.stringify(payload(), null, 2);
  navigator.clipboard?.writeText(json)
    .then(() => flash('JSON copied — paste into labels/corrections/' +
                      DATA.image.replace(/\.[^.]+$/, '') + '.corrections.json'))
    .catch(() => showJson(json));
};
document.getElementById('saveNext').onclick = () => {
  // Never navigate away from unsaved work: if the download did not fire, stay
  // on the page rather than silently losing the corrections.
  if (!download()) return;
  if (DATA.next) setTimeout(() => { location.href = DATA.next; }, 600);
  else setTimeout(() => alert('That was the last page in this batch.'), 600);
};
let outOfScope = false;
document.getElementById('scope').onclick = e => {
  outOfScope = !outOfScope;
  e.target.setAttribute('aria-pressed', outOfScope ? 'true' : 'false');
  e.target.textContent = outOfScope ? '⚠ marked out of scope' : 'Not embossed Braille';
  document.body.style.filter = outOfScope ? 'grayscale(.6)' : '';
  flash(outOfScope
    ? 'Marked as not embossed Braille — save to record it, then move on'
    : 'Scope flag cleared');
};
window.addEventListener('beforeunload', e => {
  const touched = edits.size + added.length + deleted.size + moved.size;
  if (touched && !saved) { e.preventDefault(); e.returnValue = ''; }
});
DATA.cells.forEach(c => makeCell(c, null));
draw(); tally();
</script></body></html>
'''


def build_page(img, cells, src_name, out_path, prev_href=None, next_href=None,
               index_href=None, position=None):
    """Render one self-contained correction page. Returns the written path.

    prev/next/index hrefs wire a batch together so a labeler never returns to a
    directory listing between pages; position is a (i, n) pair shown in the nav.
    """
    disp = img.copy()
    disp.thumbnail((MAX_EMBED_PX, MAX_EMBED_PX), PIL.Image.LANCZOS)
    buf = io.BytesIO()
    disp.save(buf, 'JPEG', quality=88)

    data = {
        'image': src_name, 'width': img.width, 'height': img.height,
        'display_w': disp.width, 'display_scale': disp.width / img.width,
        'median_w': statistics.median([c['w'] for c in cells]) if cells else 40,
        'median_h': statistics.median([c['h'] for c in cells]) if cells else 60,
        'generated': datetime.date.today().isoformat(),
        'next': next_href, 'prev': prev_href,
        'cells': [{'id': i, 'cx': c['cx'], 'cy': c['cy'], 'w': c['w'], 'h': c['h'],
                   'bits': c.get('bits', '000000'),
                   'conf': (round(float(c['conf']), 4)
                            if c.get('conf') is not None else None),
                   'probs': c.get('probs')}
                  for i, c in enumerate(cells)],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    def link(href, label):
        return (f'<a href="{href}">{label}</a>' if href
                else f'<span class="disabled">{label}</span>')
    nav = (link(prev_href, '← prev') + link(index_href, 'index') +
           link(next_href, 'next →'))
    if position:
        nav += f'<span class="disabled">{position[0]} of {position[1]}</span>'

    out_path.write_text(HTML
                        .replace('__NAV__', nav)
                        .replace('__NAME__', src_name)
                        .replace('__IMG__', base64.b64encode(buf.getvalue()).decode())
                        .replace('__DATA__', json.dumps(data)))
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('image')
    ap.add_argument('--out', default='/tmp/braille-label')
    ap.add_argument('--max-det', type=int, default=2000)
    ap.add_argument('--no-classifier', action='store_true')
    ap.add_argument('--open', action='store_true', help='open in the default browser')
    args = ap.parse_args()

    src = Path(args.image)
    img = PIL.ImageOps.exif_transpose(PIL.Image.open(src)).convert('RGB')
    model = YOLO(str(resolve_model('cell_detector.pt')))
    cells = predict(img, model, args.max_det, not args.no_classifier)
    print(f'  {len(cells)} cells predicted at conf >= {pipeline.HIGH_CONF}')

    out = build_page(img, cells, src.name,
                     Path(args.out) / f'{src.stem}.label.html')
    print(f'  Wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)')
    if args.open:
        import webbrowser
        webbrowser.open(out.as_uri())


if __name__ == '__main__':
    main()
