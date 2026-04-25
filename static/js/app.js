document.getElementById('yr').textContent = new Date().getFullYear();

// ============ Live UTC clock + dateline ============
function pad(n) { return String(n).padStart(2, '0'); }
function tickClock() {
  const d = new Date();
  const utc = `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}`;
  const c1 = document.getElementById('utcClock');
  const c2 = document.getElementById('utcClock2');
  if (c1) c1.textContent = `UTC ${utc}`;
  if (c2) c2.textContent = utc;
}
tickClock();
setInterval(tickClock, 1000);

const dateEl = document.getElementById('todayDate');
if (dateEl) {
  const d = new Date();
  const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  dateEl.textContent = `${pad(d.getDate())} ${months[d.getMonth()]} ${d.getFullYear()}`;
}

// ============ REC indicator state ============
function setRec(state, label) {
  const ind = document.getElementById('recIndicator');
  if (!ind) return;
  ind.classList.toggle('live', state === 'live');
  ind.querySelector('.rec-label').textContent = label || (state === 'live' ? 'Analysing' : 'Idle');
}

// ============ Number tick animation ============
function tickNumber(el, to, suffix = '%', duration = 900) {
  if (!el) return;
  const from = 0;
  const start = performance.now();
  function step(now) {
    const t = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = Math.round(from + (to - from) * eased) + suffix;
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// ============================================================
// State
// ============================================================
const state = {
  current: null,           // {file, srcToken, lastResult}
  recBlob: null,
  cmp: { a: null, b: null },
};

// ============================================================
// Helpers
// ============================================================
function $(id) { return document.getElementById(id); }
function fmtSize(b) {
  if (b < 1024) return b + ' B';
  if (b < 1024 * 1024) return (b / 1024).toFixed(1) + ' KB';
  return (b / 1024 / 1024).toFixed(1) + ' MB';
}
function fmtTime(s) {
  const m = Math.floor(s / 60), sec = Math.floor(s % 60);
  return `${String(m).padStart(2,'0')}:${String(sec).padStart(2,'0')}`;
}
function showStatus(text) {
  $('statusText').textContent = text;
  $('status').hidden = false;
}
function hideStatus() { $('status').hidden = true; }

function showError(msg, container = 'tabs') {
  clearError();
  const banner = document.createElement('div');
  banner.className = 'error-banner';
  banner.id = 'errBanner';
  banner.textContent = msg;
  document.querySelector('.tabs').appendChild(banner);
}
function clearError() {
  const e = $('errBanner'); if (e) e.remove();
}

// ============================================================
// Tabs
// ============================================================
document.querySelectorAll('.tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.querySelector(`[data-panel="${btn.dataset.tab}"]`).classList.add('active');
  });
});

// ============================================================
// Upload tab
// ============================================================
const dz = $('dropzone');
const fileInput = $('fileInput');
const player = $('player');
const playBtn = $('playBtn');
const playIcon = playBtn.querySelector('.play-icon');
const pauseIcon = playBtn.querySelector('.pause-icon');
const resetBtn = $('resetBtn');
const analyzeBtn = $('analyzeBtn');
let wavesurfer = null;

function attachUploadFile(file) {
  clearError();
  $('result').hidden = true;
  if (!file) return;
  if (!/\.(wav|mp3|flac|m4a|ogg|webm)$/i.test(file.name)) {
    showError('Unsupported format. Use .wav .mp3 .flac .m4a .ogg or .webm'); return;
  }
  if (file.size > 50 * 1024 * 1024) { showError('File too large (50 MB max).'); return; }

  state.current = { file, srcToken: null };
  $('fileName').textContent = file.name;
  $('fileSize').textContent = fmtSize(file.size);

  dz.hidden = true;
  player.hidden = false;

  if (wavesurfer) wavesurfer.destroy();
  wavesurfer = WaveSurfer.create({
    container: '#waveform',
    waveColor: 'rgba(255,255,255,0.25)',
    progressColor: '#7c5cff',
    cursorColor: 'rgba(255,255,255,0.5)',
    barWidth: 2, barGap: 2, barRadius: 2, height: 56, normalize: true,
  });
  wavesurfer.loadBlob(file);
  wavesurfer.on('finish', () => setPlaying(playIcon, pauseIcon, false));
  wavesurfer.on('play',   () => setPlaying(playIcon, pauseIcon, true));
  wavesurfer.on('pause',  () => setPlaying(playIcon, pauseIcon, false));
}
function setPlaying(pi, pa, on) { pi.hidden = on; pa.hidden = !on; }

function resetUpload() {
  state.current = null;
  if (wavesurfer) { wavesurfer.destroy(); wavesurfer = null; }
  player.hidden = true; dz.hidden = false;
  $('result').hidden = true;
  hideStatus();
  fileInput.value = '';
  clearError();
}

dz.addEventListener('click', () => fileInput.click());
dz.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }});
fileInput.addEventListener('change', (e) => attachUploadFile(e.target.files[0]));
['dragenter', 'dragover'].forEach(ev =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); e.stopPropagation(); dz.classList.add('drag'); }));
['dragleave', 'drop'].forEach(ev =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); e.stopPropagation(); dz.classList.remove('drag'); }));
dz.addEventListener('drop', (e) => attachUploadFile(e.dataTransfer.files[0]));
document.addEventListener('paste', (e) => {
  const f = [...(e.clipboardData?.files || [])][0];
  if (f) attachUploadFile(f);
});
playBtn.addEventListener('click', () => wavesurfer && wavesurfer.playPause());
resetBtn.addEventListener('click', resetUpload);
$('againBtn').addEventListener('click', resetUpload);

analyzeBtn.addEventListener('click', () => analyseUploadedFile(state.current.file));

async function analyseUploadedFile(file) {
  clearError();
  $('result').hidden = true;
  analyzeBtn.disabled = true;
  setRec('live', 'Analysing');
  showStatus('Engaging the bureau’s instruments… first run downloads the model.');
  try {
    const fd = new FormData();
    fd.append('file', file);
    const res = await fetch('/api/predict', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Request failed');
    state.current.srcToken = data.src_token;
    state.current.lastResult = data;
    renderResult(data);
  } catch (err) { showError(err.message); }
  finally { analyzeBtn.disabled = false; hideStatus(); setRec('idle', 'Idle'); }
}

// ============================================================
// Record tab
// ============================================================
const recBtn = $('recBtn');
const recStatus = $('recStatus');
const recTime = $('recTime');
const recPlayerWrap = $('recPlayerWrap');
const recPlayBtn = $('recPlayBtn');
const recPlayIcon = recPlayBtn.querySelector('.play-icon');
const recPauseIcon = recPlayBtn.querySelector('.pause-icon');
let mediaRecorder = null, recChunks = [], recTimer = null, recStart = 0, recWavesurfer = null;

recBtn.addEventListener('click', async () => {
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.stop();
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mime = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : '';
    mediaRecorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
    recChunks = [];
    mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) recChunks.push(e.data); };
    mediaRecorder.onstop = () => {
      stream.getTracks().forEach(t => t.stop());
      clearInterval(recTimer);
      recBtn.classList.remove('recording');
      setRec('idle', 'Idle');
      recStatus.textContent = '— Capture complete —';
      const blob = new Blob(recChunks, { type: mime || 'audio/webm' });
      state.recBlob = blob;
      recPlayerWrap.hidden = false;
      if (recWavesurfer) recWavesurfer.destroy();
      recWavesurfer = WaveSurfer.create({
        container: '#recWaveform',
        waveColor: 'rgba(255,255,255,0.25)',
        progressColor: '#7c5cff',
        cursorColor: 'rgba(255,255,255,0.5)',
        barWidth: 2, barGap: 2, barRadius: 2, height: 56, normalize: true,
      });
      recWavesurfer.loadBlob(blob);
      recWavesurfer.on('play',  () => setPlaying(recPlayIcon, recPauseIcon, true));
      recWavesurfer.on('pause', () => setPlaying(recPlayIcon, recPauseIcon, false));
      recWavesurfer.on('finish',() => setPlaying(recPlayIcon, recPauseIcon, false));
    };
    mediaRecorder.start();
    recStart = Date.now();
    recBtn.classList.add('recording');
    recStatus.textContent = '— Live · click to halt —';
    setRec('live', 'On record');
    recTimer = setInterval(() => {
      const ms = Date.now() - recStart;
      const m = Math.floor(ms / 60000);
      const s = Math.floor((ms % 60000) / 1000);
      const cs = Math.floor((ms % 1000)).toString().padStart(3, '0');
      recTime.textContent = `${pad(m)}:${pad(s)}.${cs}`;
    }, 50);
  } catch (err) {
    showError('Microphone permission denied or unavailable: ' + err.message);
  }
});

recPlayBtn.addEventListener('click', () => recWavesurfer && recWavesurfer.playPause());
$('recResetBtn').addEventListener('click', () => {
  state.recBlob = null;
  recPlayerWrap.hidden = true;
  recStatus.textContent = '— Press to begin a live session —';
  recTime.textContent = '00:00.000';
  if (recWavesurfer) { recWavesurfer.destroy(); recWavesurfer = null; }
});
$('recAnalyzeBtn').addEventListener('click', () => {
  if (!state.recBlob) return;
  const file = new File([state.recBlob], `recording_${Date.now()}.webm`, { type: state.recBlob.type });
  state.current = { file, srcToken: null };
  analyseUploadedFile(file);
});

// ============================================================
// URL tab
// ============================================================
$('urlBtn').addEventListener('click', async () => {
  const raw = $('urlInput').value.trim();
  const url = raw.startsWith('http') ? raw : 'https://' + raw;
  if (!raw) return;
  clearError();
  $('result').hidden = true;
  $('urlBtn').disabled = true;
  setRec('live', 'Fetching');
  showStatus('Pulling the recording over the wire — 10–60s.');
  try {
    const res = await fetch('/api/predict-url', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Request failed');
    state.current = { file: null, srcToken: data.src_token, lastResult: data };
    renderResult(data);
  } catch (err) { showError(err.message); }
  finally { $('urlBtn').disabled = false; hideStatus(); setRec('idle', 'Idle'); }
});

// ============================================================
// Compare tab
// ============================================================
document.querySelectorAll('.cmp-input').forEach(inp => {
  inp.addEventListener('change', (e) => {
    const side = inp.dataset.side;
    state.cmp[side] = e.target.files[0];
    document.querySelector(`[data-side-name="${side}"]`).textContent = e.target.files[0]?.name || 'Click or drop a file';
    $('compareBtn').disabled = !(state.cmp.a && state.cmp.b);
  });
});
document.querySelectorAll('.dropzone.small').forEach(dzs => {
  const side = dzs.dataset.side;
  dzs.addEventListener('click', () => document.querySelector(`.cmp-input[data-side="${side}"]`).click());
  ['dragenter','dragover'].forEach(ev => dzs.addEventListener(ev, (e) => { e.preventDefault(); dzs.classList.add('drag'); }));
  ['dragleave','drop'].forEach(ev => dzs.addEventListener(ev, (e) => { e.preventDefault(); dzs.classList.remove('drag'); }));
  dzs.addEventListener('drop', (e) => {
    state.cmp[side] = e.dataTransfer.files[0];
    document.querySelector(`[data-side-name="${side}"]`).textContent = e.dataTransfer.files[0]?.name || '—';
    $('compareBtn').disabled = !(state.cmp.a && state.cmp.b);
  });
});
$('compareBtn').addEventListener('click', async () => {
  $('compareBtn').disabled = true;
  setRec('live', 'Side-by-side');
  showStatus('Running both specimens through the bureau…');
  $('compareResult').hidden = true;
  try {
    const [resA, resB] = await Promise.all(
      ['a', 'b'].map(s => {
        const fd = new FormData(); fd.append('file', state.cmp[s]);
        return fetch('/api/predict', { method: 'POST', body: fd }).then(r => r.json());
      })
    );
    if (resA.error) throw new Error('A: ' + resA.error);
    if (resB.error) throw new Error('B: ' + resB.error);
    renderCompare(resA, resB);
  } catch (err) { showError(err.message); }
  finally { hideStatus(); setRec('idle', 'Idle'); $('compareBtn').disabled = false; }
});
function renderCompare(a, b) {
  const c = $('compareResult');
  const card = (lbl, r) => {
    const v = r.label === 'FAKE' ? 'Fake' : 'Real';
    return `
    <div class="cmp-card ${r.label.toLowerCase()}">
      <div class="cmp-side">— Specimen ${lbl} —</div>
      <div class="cmp-verdict">${v}.</div>
      <div class="cmp-conf">${r.confidence.toFixed(1)}% confidence</div>
      <div class="cmp-meta">${r.chunks.length} chunks · ${r.duration.toFixed(1)} s · p(fake) ${r.fake_prob.toFixed(3)}</div>
    </div>`;
  };
  c.innerHTML = card('A', a) + card('B', b);
  c.hidden = false;
  c.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ============================================================
// Result rendering (timeline, spectrogram, augment, export)
// ============================================================
function renderResult(data) {
  const isFake = data.label === 'FAKE';
  $('verdict').textContent = isFake ? 'Fake' : 'Real';
  $('verdict').classList.toggle('real', !isFake);
  $('verdict').classList.toggle('fake', isFake);
  $('verdictSub').textContent = isFake
    ? 'Likely AI-generated or synthetic speech.'
    : 'Likely a genuine human voice.';
  // Model badge shows resolved label mapping so inversion is spottable
  const lm = data.label_map?.[0];
  const fakeLbl = lm ? lm.labels[lm.fake_idx] : '?';
  const modelTxt = data.models.length === 1
    ? `Model · ${data.models[0]} · fake=${fakeLbl} · agg=${data.aggregation || 'mean'}`
    : `Ensemble · ${data.models.length} models · agg=${data.aggregation || 'mean'}`;
  $('modelBadge').textContent = modelTxt;

  const realPct = Math.round((1 - data.fake_prob) * 100);
  const fakePct = Math.round(data.fake_prob * 100);
  const conf = Math.round(data.confidence);
  const C = 2 * Math.PI * 50;
  const gauge = document.querySelector('.gauge');
  gauge.classList.toggle('real', !isFake);
  gauge.classList.toggle('fake', isFake);
  $('gaugeBar').style.strokeDashoffset = C - (C * conf) / 100;
  tickNumber($('gaugeNum'), conf, '%', 1100);
  requestAnimationFrame(() => {
    $('barReal').style.width = realPct + '%';
    $('barFake').style.width = fakePct + '%';
  });
  tickNumber($('barRealVal'), realPct, '%', 1100);
  tickNumber($('barFakeVal'), fakePct, '%', 1100);
  $('chunksMeta').textContent = `${data.chunks.length} chunks · ${data.duration.toFixed(2)} s · ${data.models.length} model${data.models.length>1?'s':''}`;

  $('augResults').innerHTML = '';
  $('specImg').src = data.spectrogram_url;
  // Show the panel BEFORE drawing canvas so clientWidth is non-zero
  $('result').hidden = false;
  requestAnimationFrame(() => {
    drawTimeline(data.chunks, data.saliency);
    drawSpecOverlay(data.chunks, data.duration);
  });
  $('result').scrollIntoView({ behavior: 'smooth', block: 'start' });
  loadHistory();
}

// Redraw on resize so the timeline scales correctly
let resizeTimer = null;
window.addEventListener('resize', () => {
  if (!state.current?.lastResult) return;
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    const d = state.current.lastResult;
    drawTimeline(d.chunks, d.saliency);
    drawSpecOverlay(d.chunks, d.duration);
  }, 120);
});

function drawTimeline(chunks, saliency) {
  const cv = $('timelineCanvas');
  const dpr = window.devicePixelRatio || 1;
  let w = cv.clientWidth;
  if (!w) {
    requestAnimationFrame(() => drawTimeline(chunks, saliency));
    return;
  }
  const h = 180;
  cv.width = w * dpr; cv.height = h * dpr;
  cv.style.height = h + 'px';
  const ctx = cv.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  // Palette
  const PAL = {
    bg: '#0c0b09',
    rule: 'rgba(244,185,66,0.10)',
    ruleStrong: 'rgba(244,185,66,0.22)',
    threshold: 'rgba(239,233,217,0.30)',
    line: '#e63946',
    fillTop: 'rgba(230,57,70,0.45)',
    fillBot: 'rgba(230,57,70,0.0)',
    saliency: 'rgba(244,185,66,',
    text: 'rgba(150,141,120,0.85)',
    textStrong: 'rgba(239,233,217,0.85)',
  };
  const monoFont = '10.5px "JetBrains Mono", monospace';
  const padL = 36, padR = 12, padT = 12, padB = 28;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  // Frame
  ctx.fillStyle = PAL.bg;
  ctx.fillRect(0, 0, w, h);

  // Y grid (graticule) every 0.25
  ctx.strokeStyle = PAL.rule;
  ctx.lineWidth = 1;
  for (let p = 0; p <= 1.001; p += 0.25) {
    const y = padT + plotH * (1 - p);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
  }
  // Y labels
  ctx.fillStyle = PAL.text;
  ctx.font = monoFont;
  ctx.textAlign = 'right';
  for (const p of [0, 0.25, 0.5, 0.75, 1]) {
    ctx.fillText(p.toFixed(2), padL - 8, padT + plotH * (1 - p) + 4);
  }
  // 0.5 threshold (dashed)
  ctx.strokeStyle = PAL.threshold;
  ctx.setLineDash([3, 4]);
  ctx.beginPath();
  ctx.moveTo(padL, padT + plotH * 0.5);
  ctx.lineTo(w - padR, padT + plotH * 0.5);
  ctx.stroke();
  ctx.setLineDash([]);

  if (!chunks.length) return;
  const tMax = chunks[chunks.length - 1].end || 1;

  // Saliency strip
  if (saliency && saliency.length === chunks.length) {
    for (let i = 0; i < chunks.length; i++) {
      const c = chunks[i], v = saliency[i];
      const x1 = padL + (c.start / tMax) * plotW;
      const x2 = padL + (c.end / tMax) * plotW;
      ctx.fillStyle = PAL.saliency + (0.10 + 0.55 * v) + ')';
      ctx.fillRect(x1, h - padB + 4, x2 - x1, 14);
    }
    ctx.fillStyle = PAL.text;
    ctx.font = monoFont;
    ctx.textAlign = 'left';
    ctx.fillText('saliency →', padL, h - padB - 4);
  }

  // Area fill
  ctx.beginPath();
  ctx.moveTo(padL, padT + plotH);
  for (const c of chunks) {
    const x = padL + ((c.start + c.end) / 2 / tMax) * plotW;
    const y = padT + plotH * (1 - c.fake_prob);
    ctx.lineTo(x, y);
  }
  ctx.lineTo(padL + plotW, padT + plotH);
  ctx.closePath();
  const grad = ctx.createLinearGradient(0, padT, 0, padT + plotH);
  grad.addColorStop(0, PAL.fillTop);
  grad.addColorStop(1, PAL.fillBot);
  ctx.fillStyle = grad;
  ctx.fill();

  // Line + dots
  ctx.beginPath();
  for (let i = 0; i < chunks.length; i++) {
    const c = chunks[i];
    const x = padL + ((c.start + c.end) / 2 / tMax) * plotW;
    const y = padT + plotH * (1 - c.fake_prob);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.strokeStyle = PAL.line;
  ctx.lineWidth = 1.5;
  ctx.stroke();
  for (const c of chunks) {
    const x = padL + ((c.start + c.end) / 2 / tMax) * plotW;
    const y = padT + plotH * (1 - c.fake_prob);
    ctx.beginPath();
    ctx.arc(x, y, 2.5, 0, Math.PI * 2);
    ctx.fillStyle = PAL.line;
    ctx.fill();
  }

  // X axis ticks (time markers)
  ctx.strokeStyle = PAL.ruleStrong;
  ctx.fillStyle = PAL.text;
  ctx.font = monoFont;
  ctx.textAlign = 'center';
  const ticks = Math.min(5, Math.max(2, chunks.length));
  for (let i = 0; i <= ticks; i++) {
    const t = (i / ticks) * tMax;
    const x = padL + (t / tMax) * plotW;
    ctx.beginPath(); ctx.moveTo(x, padT + plotH); ctx.lineTo(x, padT + plotH + 4); ctx.stroke();
    ctx.fillText(t.toFixed(1) + 's', x, h - padB + 18);
  }

  // Title in top-right
  ctx.fillStyle = PAL.textStrong;
  ctx.textAlign = 'right';
  ctx.fillText('P(FAKE) — TIME', w - padR, padT - 0);
}

function drawSpecOverlay(chunks, duration) {
  const overlay = $('specOverlay');
  overlay.innerHTML = '';
  if (!duration || !chunks.length) return;
  for (const c of chunks) {
    const div = document.createElement('div');
    div.className = 'spec-chunk';
    const left = (c.start / duration) * 100;
    const width = ((c.end - c.start) / duration) * 100;
    div.style.left = left + '%';
    div.style.width = width + '%';
    const intensity = Math.min(1, Math.max(0, c.fake_prob));
    div.style.background = c.fake_prob > 0.5
      ? `rgba(230,57,70,${0.18 + 0.45 * intensity})`
      : `rgba(200,216,210,${0.06 + 0.18 * (1 - intensity)})`;
    div.title = `${c.start.toFixed(1)}–${c.end.toFixed(1)}s · fake ${(c.fake_prob * 100).toFixed(1)}%`;
    overlay.appendChild(div);
  }
}

// ============================================================
// Augmentation
// ============================================================
document.querySelectorAll('.aug-btn').forEach(btn => {
  btn.addEventListener('click', async () => {
    if (!state.current?.srcToken) { showError('Augmentation needs a fresh upload — re-analyse to refresh.'); return; }
    const kind = btn.dataset.kind;
    btn.disabled = true; btn.classList.add('loading');
    try {
      const res = await fetch('/api/augment-predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ src_token: state.current.srcToken, kind }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Augment failed');
      addAugResult(kind, data);
    } catch (err) {
      addAugResult(kind, null, err.message);
    } finally {
      btn.disabled = false; btn.classList.remove('loading');
    }
  });
});

function addAugResult(kind, data, err) {
  const wrap = $('augResults');
  const card = document.createElement('div');
  card.className = 'aug-card';
  if (err) {
    card.innerHTML = `<span class="aug-kind">${kind}</span><span class="aug-err">${err}</span>`;
  } else {
    const baseFake = state.current.lastResult?.fake_prob ?? 0;
    const delta = (data.fake_prob - baseFake) * 100;
    const arrow = delta > 0 ? '▲' : (delta < 0 ? '▼' : '·');
    const cls = delta > 0 ? 'up' : (delta < 0 ? 'down' : '');
    card.innerHTML = `
      <span class="aug-kind">${kind}</span>
      <span class="aug-verdict ${data.label.toLowerCase()}">${data.label}</span>
      <span class="aug-conf">${data.confidence}%</span>
      <span class="aug-delta ${cls}">${arrow} ${Math.abs(delta).toFixed(1)} pp</span>`;
  }
  wrap.appendChild(card);
}

// ============================================================
// Export
// ============================================================
async function exportAs(fmt) {
  const data = state.current?.lastResult;
  if (!data) return;
  const res = await fetch(`/api/export/${fmt}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) { showError('Export failed'); return; }
  const blob = await res.blob();
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `truevoice_analysis.${fmt}`;
  a.click();
}
$('exportJsonBtn').addEventListener('click', () => exportAs('json'));
$('exportCsvBtn').addEventListener('click', () => exportAs('csv'));

// ============================================================
// Feedback / retraining queue
// ============================================================
async function refreshFeedbackStats() {
  try {
    const res = await fetch('/api/feedback/stats');
    const data = await res.json();
    const el = $('feedbackStats');
    if (el) el.textContent = `queue: ${data.counts.real} real · ${data.counts.fake} fake`;
  } catch {}
}
refreshFeedbackStats();

document.querySelectorAll('.flag-btn').forEach(btn => {
  btn.addEventListener('click', async () => {
    if (!state.current?.srcToken) {
      showFeedback('Source clip expired — re-analyse first.', true);
      return;
    }
    const claim = btn.dataset.claim;
    btn.disabled = true;
    try {
      const res = await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ src_token: state.current.srcToken, claimed_label: claim }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Flag failed');
      showFeedback(`Saved to feedback/${claim}/. Queue is now ${data.counts.real} real · ${data.counts.fake} fake.`, false);
      $('feedbackStats').textContent = `queue: ${data.counts.real} real · ${data.counts.fake} fake`;
    } catch (err) {
      showFeedback(err.message, true);
    } finally {
      btn.disabled = false;
    }
  });
});

function showFeedback(msg, isError) {
  const el = $('feedbackStatus');
  el.textContent = msg;
  el.classList.toggle('error', !!isError);
  el.hidden = false;
}

// ============================================================
// History drawer
// ============================================================
$('toggleHistoryBtn').addEventListener('click', () => {
  $('historyDrawer').hidden = false;
  loadHistory();
});
$('closeHistoryBtn').addEventListener('click', () => { $('historyDrawer').hidden = true; });

async function loadHistory() {
  try {
    const res = await fetch('/api/history');
    const items = await res.json();
    const list = $('historyList');
    if (!items.length) { list.innerHTML = '<div class="hint">No analyses yet.</div>'; return; }
    list.innerHTML = items.map(it => `
      <div class="hist-item ${it.label.toLowerCase()}">
        <div class="hist-row">
          <span class="hist-verdict">${it.label}</span>
          <span class="hist-conf">${it.confidence.toFixed(1)}%</span>
        </div>
        <div class="hist-name" title="${it.filename || ''}">${it.filename || '(unnamed)'}</div>
        <div class="hist-meta">${it.source} · ${it.duration?.toFixed?.(1) ?? '—'}s · ${new Date(it.ts*1000).toLocaleString()}</div>
      </div>`).join('');
  } catch (err) {
    $('historyList').innerHTML = `<div class="hint">Failed to load: ${err.message}</div>`;
  }
}
