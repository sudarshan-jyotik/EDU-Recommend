/**
 * frontend/static/utils.js
 * =========================
 * Shared JavaScript utilities for all EduRecommend pages.
 * Day 11 — created alongside profile.html and style.css
 *
 * Includes:
 *  - Profile helpers (read/write localStorage)
 *  - API fetch wrapper with error handling
 *  - Autocomplete helper
 *  - Feedback submission
 *  - Nav profile badge injection
 *  - Card rendering helpers
 */

'use strict';

/* ── API base URL ─────────────────────────────────────────────────── */
const API = '';  // same origin — FastAPI serves both API and frontend

/* ── Profile helpers ──────────────────────────────────────────────── */
const Profile = {
  get() {
    return {
      branch   : localStorage.getItem('userBranch')    || '',
      semester : localStorage.getItem('userSemester')  || '',
      goal     : localStorage.getItem('userGoal')      || '',
      interests: JSON.parse(localStorage.getItem('userInterests') || '[]'),
      isSet    : localStorage.getItem('profileSet') === 'true',
    };
  },

  clear() {
    ['userBranch','userSemester','userGoal','userInterests','profileSet']
      .forEach(k => localStorage.removeItem(k));
  },
};

/* ── Nav badge injection ──────────────────────────────────────────── */
/**
 * Injects a profile summary pill into the nav if the user has set their profile.
 * Call this at the top of each page's script section.
 * @param {string} activePage  - 'books' | 'courses' | 'plan' | 'profile'
 */
function initNav(activePage = '') {
  const profile = Profile.get();

  // Mark active nav link
  document.querySelectorAll('.nav-link').forEach(link => {
    const page = link.getAttribute('data-page');
    if (page === activePage) link.classList.add('active');
  });

  // Inject user badge if profile is set
  const badgeSlot = document.getElementById('nav-user-slot');
  if (badgeSlot && profile.isSet && profile.branch) {
    badgeSlot.innerHTML = `
      <span class="nav-user-badge">
        <span class="badge-dot"></span>
        ${escHtml(profile.branch)} · Sem ${escHtml(profile.semester)}
      </span>`;
  }
}

/* ── API fetch wrapper ────────────────────────────────────────────── */
/**
 * Wrapper around fetch() for JSON API calls.
 * Returns { data, ok, status, error }.
 *
 * @param {string} url
 * @param {object} options  fetch options
 * @returns {Promise<{data:any, ok:boolean, status:number, error:string|null}>}
 */
async function apiFetch(url, options = {}) {
  try {
    const res  = await fetch(API + url, options);
    const data = await res.json().catch(() => null);
    return {
      data,
      ok    : res.ok,
      status: res.status,
      error : res.ok ? null : (data?.detail?.message || data?.error || `HTTP ${res.status}`),
    };
  } catch (err) {
    return { data: null, ok: false, status: 0, error: 'Network error — is the server running?' };
  }
}

/* ── POST helper ──────────────────────────────────────────────────── */
async function apiPost(url, body) {
  return apiFetch(url, {
    method : 'POST',
    headers: { 'Content-Type': 'application/json' },
    body   : JSON.stringify(body),
  });
}

/* ── GET helper ───────────────────────────────────────────────────── */
async function apiGet(url) {
  return apiFetch(url);
}

/* ── Autocomplete ─────────────────────────────────────────────────── */
/**
 * Wire up an autocomplete input with a dropdown list.
 *
 * @param {HTMLInputElement} input        - search input element
 * @param {HTMLUListElement} list         - dropdown <ul> element
 * @param {string}           endpoint     - e.g. '/api/v1/books/autocomplete'
 * @param {function}         onSelect     - called with the selected value (string)
 * @param {number}           debounceMs   - debounce delay (default 200ms)
 */
function initAutocomplete(input, list, endpoint, onSelect, debounceMs = 200) {
  let timer;

  input.addEventListener('input', () => {
    clearTimeout(timer);
    const q = input.value.trim();
    if (q.length < 2) { hideList(); return; }

    timer = setTimeout(async () => {
      const { data, ok } = await apiGet(`${endpoint}?q=${encodeURIComponent(q)}`);
      if (!ok || !data) { hideList(); return; }

      const suggestions = data.suggestions || data;
      if (!suggestions.length) { hideList(); return; }

      list.innerHTML = suggestions.map(s => {
        const safe      = escHtml(s);
        const escaped_q = escHtml(q);
        const highlighted = safe.replace(
          new RegExp(`(${escaped_q})`, 'gi'),
          '<mark>$1</mark>',
        );
        return `<li data-value="${safe}">${highlighted}</li>`;
      }).join('');

      list.querySelectorAll('li').forEach(li => {
        li.addEventListener('click', () => {
          onSelect(li.getAttribute('data-value'));
          hideList();
        });
      });

      list.classList.add('show');
    }, debounceMs);
  });

  document.addEventListener('click', e => {
    if (!input.closest('.search-wrap')?.contains(e.target)) hideList();
  });

  function hideList() { list.classList.remove('show'); list.innerHTML = ''; }
}

/* ── Show error banner ────────────────────────────────────────────── */
/**
 * Renders an error banner with optional "did you mean?" suggestion chips.
 *
 * @param {HTMLElement} container    element to render into
 * @param {string}      message      error message text
 * @param {string[]}    suggestions  optional array of suggestion strings
 * @param {function}    onChipClick  called with the clicked suggestion string
 */
function showError(container, message, suggestions = [], onChipClick = null) {
  let chipsHtml = '';
  if (suggestions.length) {
    const chips = suggestions
      .map(s => `<span class="suggestion-chip" data-val="${escHtml(s)}">${escHtml(s)}</span>`)
      .join('');
    chipsHtml = `<div class="suggestion-chips">${chips}</div>`;
  }

  container.innerHTML = `
    <div class="error-banner">
      <div class="err-title">Not found</div>
      <div>${escHtml(message)}</div>
      ${chipsHtml}
    </div>`;

  if (onChipClick) {
    container.querySelectorAll('.suggestion-chip').forEach(chip => {
      chip.addEventListener('click', () => onChipClick(chip.getAttribute('data-val')));
    });
  }
}

/* ── Show spinner ─────────────────────────────────────────────────── */
function showSpinner(container) {
  container.innerHTML = `
    <div class="spinner-wrap">
      <div class="spinner"></div>
    </div>`;
}

/* ── Show empty state ─────────────────────────────────────────────── */
function showEmpty(container, message = 'No results found.') {
  container.innerHTML = `
    <div class="empty-state">
      <div class="empty-icon">◉</div>
      <p>${escHtml(message)}</p>
    </div>`;
}

/* ── Feedback submission ──────────────────────────────────────────── */
/**
 * Submit a feedback event to the API.
 *
 * @param {string}  query            - search query
 * @param {string}  recommendedItem  - the specific item recommended
 * @param {string}  itemType         - 'book' | 'course'
 * @param {boolean} relevant         - was it relevant?
 * @param {number}  rating           - 1-5
 */
async function submitFeedback(query, recommendedItem, itemType, relevant, rating = 5) {
  const profile = Profile.get();
  await apiPost('/api/v1/feedback/', {
    query           : query,
    recommended_item: recommendedItem,
    item_type       : itemType,
    rating          : relevant ? rating : 2,
    relevant        : relevant,
    user_branch     : profile.branch || null,
    user_semester   : profile.semester ? parseInt(profile.semester) : null,
  });
}

/* ── Book cover fallback ──────────────────────────────────────────── */
function coverOrPlaceholder(coverUrl, title) {
  if (coverUrl && coverUrl.startsWith('http')) {
    return `<img src="${escHtml(coverUrl)}" alt="${escHtml(title)}"
              loading="lazy"
              onerror="this.style.display='none';this.nextElementSibling.style.display='flex'" />
            <div class="cover-placeholder" style="display:none">&#9635;</div>`;
  }
  return `<div class="cover-placeholder">&#9635;</div>`;
}

/* ── Star rating display ──────────────────────────────────────────── */
function stars(rating) {
  const full  = Math.round(rating);
  const empty = 5 - full;
  return '★'.repeat(full) + '☆'.repeat(empty);
}

/* ── HTML escape ──────────────────────────────────────────────────── */
function escHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/* ── Truncate text ────────────────────────────────────────────────── */
function truncate(str, max = 60) {
  if (!str || str.length <= max) return str || '';
  return str.slice(0, max).trim() + '…';
}

/* ── Animate similarity bars ──────────────────────────────────────── */
function animateSimBars() {
  document.querySelectorAll('.sim-fill, .score-bar-fill').forEach(el => {
    const target = el.style.width;
    el.style.width = '0';
    setTimeout(() => { el.style.width = target; }, 100);
  });
}
