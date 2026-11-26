/**
 * ═══════════════════════════════════════════════════
 *  SIGAP AI — app.js
 *  Sistem Intelijen Garda Ancaman Pekerjaan
 *
 *  Tanggung jawab file ini:
 *  ✅ Interaksi UI (input, loading, animasi, render)
 *  ✅ Mengirim laporan dan contoh knowledge ke Python backend
 * ═══════════════════════════════════════════════════
 */

'use strict';

// Saat disajikan Flask (port 5000), gunakan origin yang sama agar tidak
// bergantung pada CORS. Live Server tetap memakai backend Flask lokal.
const API_BASE_URL = window.SIGAP_API_BASE_URL
  || (window.location.port === '5000' ? '' : 'http://localhost:5000');

// ═══════════════════════════════════════════════════
//  UTILITIES
// ═══════════════════════════════════════════════════

function getTimestamp() {
  return new Date().toLocaleString('id-ID', {
    day: '2-digit', month: 'short', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}

function wordCount(text) {
  return text.trim().split(/\s+/).filter(Boolean).length;
}

function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ═══════════════════════════════════════════════════
//  UI EVENT HANDLERS
// ═══════════════════════════════════════════════════

function onInputChange() {
  const val = document.getElementById('report-input').value;
  const badge = document.getElementById('char-badge');
  badge.textContent = val.length + ' karakter';
  badge.classList.toggle('active', val.length >= 20);
  clearAnalysisError();
}

function clearInput() {
  document.getElementById('report-input').value = '';
  onInputChange();
  document.getElementById('report-input').focus();
}

function showAnalysisError(message) {
  const error = document.getElementById('analysis-error');
  error.textContent = message;
  error.hidden = false;
}

function clearAnalysisError() {
  const error = document.getElementById('analysis-error');
  error.textContent = '';
  error.hidden = true;
}

async function loadSample() {
  const button = document.getElementById('btn-sample');
  button.disabled = true;
  clearAnalysisError();

  try {
    const response = await fetch(`${API_BASE_URL}/api/sample`);
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || `Server error: ${response.status}`);
    }
    if (typeof data.laporan !== 'string' || !data.laporan.trim()) {
      throw new Error('Backend tidak mengirim contoh laporan yang valid.');
    }

    const textarea = document.getElementById('report-input');
    textarea.value = data.laporan.trim();
    onInputChange();
    textarea.focus();
  } catch (error) {
    console.error('[SIGAP AI] Gagal memuat contoh laporan:', error);
    showAnalysisError('Contoh laporan gagal dimuat. Pastikan backend SIGAP AI sedang berjalan.');
  } finally {
    button.disabled = false;
  }
}

// Ctrl+Enter shortcut
document.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') analyzeReport();
});

// ═══════════════════════════════════════════════════
//  LOADING ANIMATION
// ═══════════════════════════════════════════════════

const STEP_LABELS = [
  'Parsing teks laporan kerja...',
  'Mencari referensi K3 via RAG...',
  'Mendeteksi pola risiko...',
  'Menyusun rekomendasi tindakan...',
];

let loadingRunId = 0;

async function runLoadingSteps(runId) {
  const stepIds = ['step-1', 'step-2', 'step-3', 'step-4'];
  const loaderText = document.getElementById('loader-text');

  for (let i = 0; i < stepIds.length; i++) {
    await delay(420 + i * 480);
    if (runId !== loadingRunId) return;
    if (i > 0) document.getElementById(stepIds[i - 1]).className = 'pipeline-step done';
    document.getElementById(stepIds[i]).className = 'pipeline-step active';
    loaderText.textContent = STEP_LABELS[i];
  }
  await delay(380);
  if (runId !== loadingRunId) return;
  document.getElementById(stepIds[stepIds.length - 1]).className = 'pipeline-step done';
}

function resetSteps() {
  ['step-1', 'step-2', 'step-3', 'step-4'].forEach(id => {
    document.getElementById(id).className = 'pipeline-step';
  });
}

// ═══════════════════════════════════════════════════
//  MAIN: analyzeReport()
//  Fungsi utama — dieksekusi saat tombol diklik
// ═══════════════════════════════════════════════════

async function analyzeReport() {
  const input = document.getElementById('report-input').value.trim();

  // Validasi minimal karakter
  if (input.length < 20) {
    setUIState('idle');
    showAnalysisError('Masukkan laporan minimal 20 karakter agar dapat dianalisis.');
    shakeInputCard();
    return;
  }

  clearAnalysisError();

  // ── Set state: loading ──
  setUIState('loading');
  resetSteps();
  loadingRunId += 1;
  runLoadingSteps(loadingRunId);

  try {
    const interactionRequestId = typeof crypto.randomUUID === 'function'
      ? crypto.randomUUID()
      : `sigap-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const response = await fetch(`${API_BASE_URL}/api/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Interaction-Request-ID': interactionRequestId,
      },
      body: JSON.stringify({ laporan: input }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || `Server error: ${response.status}`);
    }
    await delay(500); // beri waktu animasi step selesai

    const mappedData = mapBackendResponse(data);
    renderResult(mappedData, input);
  } catch (error) {
    console.error('[SIGAP AI] Gagal menganalisis laporan:', error);
    setUIState('idle');
    showAnalysisError('Analisis gagal. Periksa koneksi backend, lalu coba lagi.');
  }
}

// ═══════════════════════════════════════════════════
//  BACKEND RESPONSE MAPPER
//  Konversi format response Python → format UI
// ═══════════════════════════════════════════════════

function mapBackendResponse(data) {
  return {
    level:           data.level,
    caption:         getCaptionForLevel(data.level),
    score:           getScoreForLevel(data.level),
    findings:        Array.isArray(data.temuan) ? data.temuan : ['Lihat analisis penyebab'],
    cause:           data.penyebab || '—',
    recommendations: Array.isArray(data.rekomendasi) ? data.rekomendasi : [data.rekomendasi],
    references:      data.referensi
      ? [{ text: data.referensi, source: 'Knowledge Base RAG' }]
      : [],
    retrievedDocuments: Array.isArray(data.retrieved_documents) ? data.retrieved_documents : [],
    answer: data.answer || '',
    similarity: data.similarity_evaluation || {},
  };
}

function getCaptionForLevel(level) {
  const map = {
    TINGGI: 'Risiko sangat tinggi — tindakan segera diperlukan',
    SEDANG: 'Risiko moderat — pengendalian segera dibutuhkan',
    RENDAH: 'Risiko rendah — pertahankan kondisi dan monitoring rutin',
    'TIDAK DAPAT DINILAI': 'Belum ada kecocokan yang cukup untuk menentukan tingkat risiko',
  };
  return map[level] || '—';
}

function getScoreForLevel(level) {
  return { TINGGI: 90, SEDANG: 58, RENDAH: 18, 'TIDAK DAPAT DINILAI': 0 }[level] ?? 18;
}

// ═══════════════════════════════════════════════════
//  RENDER RESULT
//  Mengisi semua card dengan data hasil analisis
// ═══════════════════════════════════════════════════

function renderResult(data, inputText) {
  const isUnassessed = data.level === 'TIDAK DAPAT DINILAI';
  const levelLower = isUnassessed ? 'unassessed' : data.level.toLowerCase();
  const colorMap   = { rendah: '#22C55E', sedang: '#F59E0B', tinggi: '#EF4444', unassessed: '#94A3B8' };
  const dotClass   = { rendah: 'finding-dot--green', sedang: 'finding-dot--amber', tinggi: 'finding-dot--red', unassessed: 'finding-dot--neutral' };

  // ── Risk Banner ──
  const badge = document.getElementById('risk-badge-large');
  badge.textContent  = isUnassessed ? 'BELUM DINILAI' : data.level;
  badge.className    = 'risk-badge-large ' + levelLower;

  document.querySelector('.risk-banner__label').textContent =
    isUnassessed ? 'Status Penilaian' : 'Level Risiko Terdeteksi';
  document.getElementById('risk-caption').textContent = data.caption;

  // Activate correct indicator
  document.querySelectorAll('.risk-indicator').forEach(el => {
    el.classList.toggle('active', !isUnassessed && el.dataset.level === levelLower);
  });

  // ── Gauge ──
  const track     = document.getElementById('gauge-track');
  const total     = 251; // full arc circumference
  const offset    = total - (data.score / 100) * total;
  track.style.stroke          = colorMap[levelLower];
  track.style.strokeDashoffset = total; // start from 0
  setTimeout(() => { track.style.strokeDashoffset = offset; }, 80);
  document.getElementById('gauge-score').textContent = data.score + '%';

  // ── Findings ──
  document.getElementById('finding-list').innerHTML = data.findings
    .map(f => `
      <li class="finding-item">
        <div class="finding-dot ${dotClass[levelLower]}"></div>
        <span>${f}</span>
      </li>`)
    .join('');

  // ── Cause ──
  document.getElementById('cause-text').textContent = data.cause;

  // ── Recommendations ──
  document.getElementById('reco-list').innerHTML = data.recommendations
    .map((r, i) => `
      <li class="reco-item">
        <div class="reco-num">${i + 1}</div>
        <span>${r}</span>
      </li>`)
    .join('');

  // ── References ──
  document.getElementById('ref-list').innerHTML = data.references
    .map(r => `
      <div class="ref-item">
        <span class="ref-item__icon">📄</span>
        <div>
          ${r.text}
          <span class="ref-item__source">${r.source}</span>
        </div>
      </div>`)
    .join('');

  // ── Evaluation dashboard ──
  const retrievedDocuments = data.retrievedDocuments;
  const average = values => values.length
    ? values.reduce((total, value) => total + value, 0) / values.length
    : null;
  const getScores = key => retrievedDocuments
    .map(item => item[key])
    .filter(value => typeof value === 'number' && Number.isFinite(value));
  const cosineScores = getScores('cosine_similarity');
  const relevanceScores = getScores('relevance_score');
  const lexicalCoverageScores = getScores('lexical_coverage');
  const metricCards = [
    {
      label: 'Rata-rata cosine similarity',
      value: average(cosineScores),
      detail: value => `Rata-rata dari ${cosineScores.length} dokumen`,
    },
    {
      label: 'Cosine similarity tertinggi',
      value: cosineScores.length ? Math.max(...cosineScores) : null,
      detail: () => 'Kecocokan semantik terbaik',
    },
    {
      label: 'Cosine similarity terendah',
      value: cosineScores.length ? Math.min(...cosineScores) : null,
      detail: () => 'Kecocokan semantik terendah',
    },
    {
      label: 'Rata-rata skor ranking',
      value: average(relevanceScores),
      detail: () => 'Skor gabungan per segmen laporan',
    },
    {
      label: 'Rata-rata cakupan kata',
      value: average(lexicalCoverageScores),
      detail: value => `Dihitung untuk ${lexicalCoverageScores.length} dokumen`,
    },
    {
      label: 'Dokumen ditemukan',
      count: retrievedDocuments.length,
      detail: () => 'Dari maksimal 2 konteks retrieval',
    },
  ];
  document.getElementById('evaluation-grid').innerHTML = metricCards.map(metric => {
    const available = typeof metric.value === 'number' && Number.isFinite(metric.value);
    const value = metric.count !== undefined
      ? `${metric.count} / 2`
      : available ? `${(metric.value * 100).toFixed(2)}%` : 'Belum ada data';
    const raw = metric.count !== undefined
      ? `${metric.count} dokumen`
      : available ? metric.value.toFixed(4) : 'Tidak ada konteks untuk dihitung';
    return `
      <div class="metric-card">
        <span class="metric-card__label">${metric.label}</span>
        <strong class="metric-card__value">${value}</strong>
        <span class="metric-card__raw">${raw}</span>
        <span class="metric-card__raw">${metric.detail(metric.value)}</span>
      </div>`;
  }).join('');

  document.getElementById('evaluation-status').textContent =
    'Semua skor dihitung lokal dari SentenceTransformer dan ranking FAISS. '
    + 'Skor ranking per segmen = 45% cosine similarity + 55% cakupan kata berbobot IDF. '
    + 'Cosine similarity pada tabel membandingkan laporan penuh dengan dokumen. '
    + 'Ini mengukur retrieval, bukan akurasi jawaban; metrik yang memerlukan ground truth '
    + 'atau evaluator generatif tidak ditampilkan.';

  document.getElementById('retrieval-table-body').innerHTML =
    data.retrievedDocuments.length
      ? data.retrievedDocuments.map(item => `
          <tr>
            <td>${item.rank}</td>
            <td>
              <strong>${item.document.kondisi || '—'}</strong><br>
              <small>Risiko: ${item.document.risiko || '—'}</small><br>
              <small>${item.document.penjelasan || ''}</small><br>
              <small>Solusi: ${item.document.solusi || '—'}</small>
            </td>
            <td>${(item.cosine_similarity * 100).toFixed(2)}%<br><small>${item.cosine_similarity.toFixed(4)}</small></td>
          </tr>`).join('')
      : '<tr><td colspan="3">Tidak ada dokumen yang relevan ditemukan.</td></tr>';

  // ── Meta bar ──
  document.getElementById('meta-time').textContent  = getTimestamp();
  document.getElementById('meta-words').textContent = wordCount(inputText) + ' kata';

  // ── Show result section ──
  setUIState('result');

  setTimeout(() => {
    document.getElementById('result-section').scrollIntoView({
      behavior: 'smooth',
      block: 'start',
    });
  }, 200);
}

// ═══════════════════════════════════════════════════
//  UI STATE MANAGER
// ═══════════════════════════════════════════════════

function setUIState(state) {
  if (state !== 'loading') loadingRunId += 1;
  const btnAnalyze     = document.getElementById('analyze-btn');
  const loadingSection = document.getElementById('loading-section');
  const resultSection  = document.getElementById('result-section');
  const statusBadge    = document.querySelector('.status-badge');
  const statusText     = document.querySelector('.status-text');

  switch (state) {
    case 'loading':
      btnAnalyze.disabled = true;
      loadingSection.classList.add('active');
      loadingSection.setAttribute('aria-hidden', 'false');
      resultSection.classList.remove('active');
      resultSection.setAttribute('aria-hidden', 'true');
      statusBadge.classList.add('analyzing');
      statusText.textContent = 'ANALYZING...';
      break;

    case 'result':
      btnAnalyze.disabled = false;
      loadingSection.classList.remove('active');
      loadingSection.setAttribute('aria-hidden', 'true');
      resultSection.classList.add('active');
      resultSection.setAttribute('aria-hidden', 'false');
      statusBadge.classList.remove('analyzing');
      statusText.textContent = 'ANALYSIS COMPLETE';
      break;

    case 'idle':
    default:
      btnAnalyze.disabled = false;
      loadingSection.classList.remove('active');
      loadingSection.setAttribute('aria-hidden', 'true');
      resultSection.classList.remove('active');
      resultSection.setAttribute('aria-hidden', 'true');
      statusBadge.classList.remove('analyzing');
      statusText.textContent = 'SYSTEM READY';
      break;
  }
}

function resetToInput() {
  setUIState('idle');
  document.getElementById('input-section').scrollIntoView({
    behavior: 'smooth',
    block: 'start',
  });
}

// ═══════════════════════════════════════════════════
//  UX: Shake animation saat validasi gagal
// ═══════════════════════════════════════════════════

function shakeInputCard() {
  const card = document.getElementById('input-card');
  card.style.borderColor = 'var(--red-border)';
  card.style.boxShadow   = '0 0 0 1px rgba(239,68,68,0.15)';

  card.animate([
    { transform: 'translateX(0)' },
    { transform: 'translateX(-7px)' },
    { transform: 'translateX(7px)' },
    { transform: 'translateX(-5px)' },
    { transform: 'translateX(5px)' },
    { transform: 'translateX(0)' },
  ], { duration: 380, easing: 'ease-in-out' });

  setTimeout(() => {
    card.style.borderColor = '';
    card.style.boxShadow   = '';
  }, 2200);
}
