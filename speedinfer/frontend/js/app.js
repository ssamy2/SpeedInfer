/**
 * SpeedInfer Main Application Controller
 * Handles SPA navigation, views rendering, modals, chat streaming, and real-time backend synchronization.
 */

import { api } from './api.js?v=20260927-inception-release';
import { store } from './store.js?v=20260927-inception-release';
import { initContact, loadInbox, clearInbox } from './contact.js?v=20260927-inception-release';
import { Workspace, workspaceTitles } from './workspace.js?v=20260927-inception-release';
import { showPublicPolicies } from './policies.js?v=20260927-inception-release';

// Global Toast Dispatcher
export function showToast(message, type = 'info', duration = 4000) {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `
    <div style="flex:1;">${escapeHtml(message)}</div>
    <button class="btn-icon" style="padding:2px;" aria-label="Close">&times;</button>
  `;

  toast.querySelector('button').addEventListener('click', () => toast.remove());
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(20px)';
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

// Global Clipboard Copy Helper
export async function copyToClipboard(text, successMsg = 'Copied to clipboard!') {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
    } else {
      const textArea = document.createElement('textarea');
      textArea.value = text;
      textArea.style.position = 'fixed';
      textArea.style.left = '-999999px';
      document.body.appendChild(textArea);
      textArea.focus();
      textArea.select();
      document.execCommand('copy');
      textArea.remove();
    }
    showToast(successMsg, 'success');
  } catch (err) {
    showToast('Failed to copy to clipboard', 'error');
  }
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function fallbackDeterministicHash(str) {
  let h1 = 0xdeadbeef, h2 = 0x41c6ce57;
  for (let i = 0; i < str.length; i++) {
    const ch = str.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 2654435761);
    h2 = Math.imul(h2 ^ ch, 1597334677);
  }
  h1 = ((h1 ^ (h1 >>> 16)) >>> 0).toString(16).padStart(8, '0');
  h2 = ((h2 ^ (h2 >>> 16)) >>> 0).toString(16).padStart(8, '0');
  return `fp_${h1}${h2}`;
}

// Client-side Device Fingerprinting
export async function generateDeviceFingerprint() {
  try {
    const parts = [
      navigator.userAgent || '',
      navigator.language || '',
      screen.colorDepth || '',
      `${screen.width}x${screen.height}`,
      new Date().getTimezoneOffset(),
      navigator.hardwareConcurrency || '',
      navigator.deviceMemory || '',
    ];

    try {
      const canvas = document.createElement('canvas');
      canvas.width = 240;
      canvas.height = 60;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.textBaseline = 'top';
        ctx.font = "14px 'Arial'";
        ctx.fillStyle = '#f60';
        ctx.fillRect(125, 1, 62, 20);
        ctx.fillStyle = '#069';
        ctx.fillText('SpeedInfer AntiFraud <canvas> 1.0', 2, 15);
        ctx.fillStyle = 'rgba(102, 204, 0, 0.7)';
        ctx.fillText('SpeedInfer AntiFraud <canvas> 1.0', 4, 17);
        parts.push(canvas.toDataURL());
      }
    } catch {
      // Ignore if canvas fingerprinting is disabled or restricted
    }

    const rawString = parts.join('~~~');
    if (window.crypto?.subtle?.digest) {
      const encoder = new TextEncoder();
      const data = encoder.encode(rawString);
      const hashBuffer = await crypto.subtle.digest('SHA-256', data);
      const hashArray = Array.from(new Uint8Array(hashBuffer));
      return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
    }
    return fallbackDeterministicHash(rawString);
  } catch {
    return fallbackDeterministicHash(navigator.userAgent || 'unknown_device');
  }
}

// =========================================================================
// Modal Helpers
// =========================================================================
const modalReturnFocus = new WeakMap();

function openModal(modalId) {
  const modal = document.getElementById(modalId);
  if (modal) {
    modalReturnFocus.set(modal, document.activeElement);
    modal.setAttribute('role', 'dialog');
    modal.setAttribute('aria-modal', 'true');
    const heading = modal.querySelector('h3');
    if (heading) {
      heading.id ||= `${modalId}-title`;
      modal.setAttribute('aria-labelledby', heading.id);
    }
    modal.classList.add('active');
    document.getElementById('app-shell').inert = true;
    modal.querySelector('input, button, textarea, select')?.focus();
  }
}

function closeModal(modalId) {
  const modal = document.getElementById(modalId);
  if (modal) {
    modal.classList.remove('active');
    if (modalId === 'modal-secret-key-reveal') document.getElementById('modal-secret-key-val').textContent = '';
    document.getElementById('app-shell').inert = false;
    modalReturnFocus.get(modal)?.focus();
  }
}

// =========================================================================
// App Application Controller
// =========================================================================
class AppController {
  constructor() {
    this.currentView = 'dashboard';
    this.keyToRevoke = null;
    this.isStreaming = false;
    this.creditPackages = [];
    this.workspace = new Workspace(view => this.showAppView(view), showToast);
  }

  async init() {
    sessionStorage.removeItem('speedinfer_last_raw_key');
    if (location.pathname === '/legal' || location.pathname.startsWith('/legal/')) { showPublicPolicies(); return; }
    this.workspace.init();
    initContact();
    this.bindGlobalEvents();
    document.querySelectorAll('[data-workspace-view]').forEach(button => button.addEventListener('click', () => this.showAppView(button.dataset.workspaceView)));
    document.getElementById('copy-console-code').addEventListener('click', () => copyToClipboard(document.getElementById('console-code').textContent));
    document.getElementById('model-search').addEventListener('input', () => this.renderModelsView());
    document.getElementById('pricing-model').addEventListener('change', () => this.renderUsageView());
    document.querySelectorAll('[data-auth]').forEach(button => button.addEventListener('click', () => {
      if (api.isAuthenticated()) this.showAppView('dashboard');
      else this.showAuthView(button.dataset.auth);
      window.scrollTo(0, 0);
    }));
    document.querySelectorAll('[data-home]').forEach(button => button.addEventListener('click', () => this.showLandingView()));
    const pythonSnippet = `import os
from openai import OpenAI

# Select a model returned by client.models.list()
client = OpenAI(
    base_url="https://speedinfer.com/v1",
    api_key=os.environ["SPEEDINFER_API_KEY"],
)

stream = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[
        {"role": "user", "content": "Explain quantum teleportation in two sentences."}
    ],
    max_tokens=128,
    stream=True,
)

for chunk in stream:
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="", flush=True)`;

    const curlSnippet = `curl -X POST "https://speedinfer.com/v1/chat/completions" \\
  -H "Authorization: Bearer $SPEEDINFER_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "REPLACE_WITH_MODEL_FROM_CATALOG",
    "messages": [
      {"role": "user", "content": "Explain quantum teleportation in two sentences."}
    ],
    "stream": true,
    "max_tokens": 128
  }'`;

    const tabPy = document.getElementById('tab-python');
    const tabCurl = document.getElementById('tab-curl');
    const landingCode = document.getElementById('landing-code');
    const fileLabel = document.getElementById('code-file-label');
    const langLabel = document.getElementById('code-lang-label');

    if (tabPy && tabCurl && landingCode) {
      const switchTab = (toPy) => {
        if (toPy) {
          tabPy.classList.add('active');
          tabPy.setAttribute('aria-selected', 'true');
          tabCurl.classList.remove('active');
          tabCurl.setAttribute('aria-selected', 'false');
          if (fileLabel) fileLabel.textContent = 'inference.py';
          if (langLabel) langLabel.textContent = 'Python / OpenAI SDK (text chat)';
          landingCode.textContent = pythonSnippet;
          tabPy.focus();
        } else {
          tabCurl.classList.add('active');
          tabCurl.setAttribute('aria-selected', 'true');
          tabPy.classList.remove('active');
          tabPy.setAttribute('aria-selected', 'false');
          if (fileLabel) fileLabel.textContent = 'request.sh';
          if (langLabel) langLabel.textContent = 'cURL / Bash (text chat)';
          landingCode.textContent = curlSnippet;
          tabCurl.focus();
        }
      };

      tabPy.addEventListener('click', () => switchTab(true));
      tabCurl.addEventListener('click', () => switchTab(false));
      tabPy.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') { e.preventDefault(); switchTab(false); }
      });
      tabCurl.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') { e.preventDefault(); switchTab(true); }
      });
    }

    const copyBtn = document.getElementById('copy-example');
    if (copyBtn && landingCode) {
      copyBtn.addEventListener('click', async () => {
        await copyToClipboard(landingCode.textContent);
        const originalText = copyBtn.textContent;
        copyBtn.textContent = 'Copied ✓';
        setTimeout(() => {
          if (copyBtn.isConnected) copyBtn.textContent = originalText;
        }, 1800);
      });
    }
    document.querySelectorAll('.nav-item').forEach(item => {
      if (!item.dataset.view) return;
      item.tabIndex = 0;
      item.setAttribute('role', 'button');
      item.addEventListener('keydown', event => {
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); item.click(); }
      });
    });
    this.bindAuthEvents();
    this.bindModalEvents();
    this.bindPlaygroundEvents();

    // Listen to session expiration
    window.addEventListener('speedinfer:session_expired', () => {
      this.workspace.reset();
      document.getElementById('playground-custom-key').value = '';
      clearInbox();
      store.setState({user:null, keys:[], balance:0, models:[], chatHistory:[], selectedKey:null});
      showToast('Session expired. Please log in again.', 'error');
      this.showAuthView('login');
    });

    // Handle URL parameters for Referrals and OAuth Callbacks
    const urlParams = new URLSearchParams(window.location.search);
    const urlRef = urlParams.get('ref') || urlParams.get('r');
    const oauthToken = urlParams.get('token');

    if (urlRef) {
      const cleanRef = urlRef.trim().toUpperCase();
      sessionStorage.setItem('speedinfer_referral_code', cleanRef);
      const refInput = document.getElementById('register-referral');
      if (refInput) refInput.value = cleanRef;
    }

    generateDeviceFingerprint().then(fp => {
      const storedRef = sessionStorage.getItem('speedinfer_referral_code') || (urlRef ? urlRef.trim().toUpperCase() : '');
      const googleBtn = document.getElementById('oauth-google-btn');
      const githubBtn = document.getElementById('oauth-github-btn');
      const params = new URLSearchParams();
      if (storedRef) params.set('ref', storedRef);
      if (fp) params.set('fp', fp);
      const query = params.toString() ? `?${params.toString()}` : '';
      if (googleBtn) googleBtn.href = `/v1/auth/oauth/google${query}`;
      if (githubBtn) githubBtn.href = `/v1/auth/oauth/github${query}`;
    }).catch(() => {});

    if (oauthToken) {
      api.setToken(oauthToken);
      window.history.replaceState({}, document.title, window.location.pathname);
      showToast('Successfully signed in!', 'success');
    }

    // Check existing auth
    if (api.isAuthenticated()) {
      try {
        await this.loadInitialData();
        this.showAppView(this.routeFromLocation());
      } catch (err) {
        api.clearSession();
        this.showAuthView('login');
      }
    } else {
      if (location.pathname.startsWith('/app')) this.showAuthView('login');
      else this.showLandingView();
    }

    // Subscribe to store updates
    store.subscribe(() => {
      if (!api.isAuthenticated()) return;
      this.renderHeader();
      this.renderActiveView();
    });
    window.addEventListener('popstate', () => {
      if (api.isAuthenticated() && location.pathname.startsWith('/app')) this.showAppView(this.routeFromLocation(), false);
      else this.showLandingView();
    });

    // Start background health polling every 12 seconds
    setInterval(() => {
      if (api.isAuthenticated()) {
        this.pollHealth();
      }
    }, 12000);
  }

  async loadInitialData() {
    try {
      const [user, keysResp, modelsResp, healthResp] = await Promise.all([
        api.getMe(),
        api.listKeys(),
        api.listModels().catch(() => ({ data: [] })),
        api.checkHealth().catch(() => ({ status: 'unknown' })),
      ]);

      store.setUser(user);
      store.setKeys(keysResp.data || []);
      store.setModels(modelsResp.data || []);
      store.setHealth(healthResp);
    } catch (err) {
      console.error('Failed to load initial user data:', err);
      throw err;
    }
  }

  async pollHealth() {
    try {
      const health = await api.checkHealth();
      store.setHealth(health);
    } catch {
      store.setHealth({ status: 'offline', workers: [], models: [] });
    }
  }

  // =========================================================================
  // View Routing & UI Shell
  // =========================================================================
  showLandingView() {
    if (location.pathname.startsWith('/app')) history.pushState({}, '', '/');
    document.getElementById('landing-page').style.display = 'block';
    document.getElementById('auth-container').style.display = 'none';
    document.getElementById('app-shell').style.display = 'none';
    window.scrollTo(0, 0);
  }

  showAuthView(mode = 'login') {
    document.getElementById('landing-page').style.display = 'none';
    document.getElementById('auth-container').style.display = 'flex';
    document.getElementById('app-shell').style.display = 'none';
    this.switchAuthTab(mode);
  }

  routeFromLocation() {
    return location.pathname.replace(/^\/app\/?/, '') || location.hash.slice(1) || 'dashboard';
  }

  showAppView(viewName, updateHistory = true) {
    const aliases = {overview:'dashboard','api-keys':'keys',billing:'usage','model-registry':'registry',storage:'buckets',support:'help'};
    viewName = aliases[viewName] || viewName;
    if (!this.workspace.supports(viewName) && !['playground','keys','models','usage','help','inbox'].includes(viewName)) viewName = 'dashboard';
    document.getElementById('landing-page').style.display = 'none';
    document.getElementById('auth-container').style.display = 'none';
    document.getElementById('app-shell').style.display = 'flex';
    if (viewName === 'inbox' && !store.state.user?.is_admin) viewName = 'dashboard';
    if (this.currentView !== viewName) {
      this.workspace.route = null;
      document.querySelector('.view-viewport').scrollTop = 0;
    }
    this.currentView = viewName;
    const baseView = viewName.split('/')[0];
    document.querySelector('.header-title').textContent = workspaceTitles[baseView] || ({playground:'Playground', keys:'API keys', models:'Models', usage:'Credits & billing', help:'Help & contact', inbox:'Team inbox'})[viewName] || 'Workspace';
    if (updateHistory && location.pathname !== `/app/${viewName}`) history.pushState({}, '', `/app/${viewName}`);
    if (viewName === 'inbox') loadInbox();
    store.setCurrentTab(viewName);

    // Update nav links
    document.querySelectorAll('.nav-item').forEach(el => {
      if (el.dataset.view === baseView) {
        el.classList.add('active');
      } else {
        el.classList.remove('active');
      }
    });

    // Close mobile drawer if open
    document.querySelector('.app-sidebar')?.classList.remove('mobile-open');
    document.querySelector('.sidebar-backdrop')?.classList.remove('active');

    this.renderHeader();
    this.renderActiveView();
  }

  switchAuthTab(tab) {
    const loginTabBtn = document.getElementById('tab-btn-login');
    const registerTabBtn = document.getElementById('tab-btn-register');
    const loginForm = document.getElementById('login-form');
    const registerForm = document.getElementById('register-form');

    loginTabBtn.setAttribute('aria-selected', String(tab === 'login'));
    registerTabBtn.setAttribute('aria-selected', String(tab !== 'login'));
    if (tab === 'login') {
      loginTabBtn.classList.add('active');
      registerTabBtn.classList.remove('active');
      loginForm.style.display = 'flex';
      registerForm.style.display = 'none';
    } else {
      registerTabBtn.classList.add('active');
      loginTabBtn.classList.remove('active');
      registerForm.style.display = 'flex';
      loginForm.style.display = 'none';
    }
  }

  renderHeader() {
    const { user, balance, health } = store.state;
    document.getElementById('inbox-nav').hidden = !user?.is_admin;

    const avatarEl = document.getElementById('profile-avatar');
    if (avatarEl) {
      if (user?.avatar_url) {
        avatarEl.innerHTML = `<img src="${escapeHtml(user.avatar_url)}" alt="Avatar" style="width:100%; height:100%; border-radius:50%; object-fit:cover;">`;
      } else {
        avatarEl.textContent = (user?.name || user?.email || 'S').slice(0, 1).toUpperCase();
      }
    }

    const verifyBadge = document.getElementById('header-verify-badge');
    if (verifyBadge) {
      verifyBadge.style.display = (user && !user.is_verified) ? 'inline-flex' : 'none';
    }

    const balanceElem = document.getElementById('header-balance-val');
    if (balanceElem) {
      const pBal = parseFloat(store.state.paidBalance || user?.paid_balance || 0).toFixed(2);
      const tBal = parseFloat(store.state.trialBalance || user?.trial_balance || 0).toFixed(2);
      balanceElem.textContent = `$${parseFloat(balance || 0).toFixed(4)} USD`;
      balanceElem.title = `Paid: $${pBal} | Promo: $${tBal}`;
    }

    const emailElem = document.getElementById('header-user-email');
    if (emailElem && user) {
      emailElem.textContent = user.name || user.email;
    }

    const statusDot = document.getElementById('cluster-status-dot');
    const statusText = document.getElementById('cluster-status-text');
    if (statusDot && statusText) {
      if (health.status === 'healthy') {
        statusDot.className = 'status-dot';
        statusText.textContent = 'Gateway responding';
      } else {
        statusDot.className = 'status-dot offline';
        statusText.textContent = 'Gateway unavailable';
      }
    }
  }

  renderActiveView() {
    // Hide all view panels
    document.querySelectorAll('.view-panel').forEach(panel => {
      panel.style.display = 'none';
    });
    if (this.workspace.supports(this.currentView)) {
      this.workspace.panel.style.display = 'block';
      this.workspace.show(this.currentView);
      return;
    }

    const activePanel = document.getElementById(`view-${this.currentView}`);
    if (activePanel) {
      activePanel.style.display = 'flex';
    }

    switch (this.currentView) {
      case 'dashboard':
        this.renderDashboardView();
        break;
      case 'playground':
        this.renderPlaygroundView();
        break;
      case 'keys':
        this.renderKeysView();
        break;
      case 'models':
        this.renderModelsView();
        break;
      case 'usage':
        this.renderUsageView();
        break;
    }
  }

  // =========================================================================
  // View: Dashboard
  // =========================================================================
  renderDashboardView() {
    const { user, balance, keys, models, health } = store.state;

    document.getElementById('dashboard-greeting').textContent = user?.name ? `Welcome back, ${user.name.split(' ')[0]}.` : 'Welcome to your workspace.';
    const modelId = models[0]?.id || models[0]?.name || 'YOUR_DEPLOYED_MODEL';
    document.getElementById('console-code').textContent = `import os
from openai import OpenAI

client = OpenAI(
    base_url="${window.location.origin}/v1",
    api_key="YOUR_SPEEDINFER_KEY",
)

stream = client.chat.completions.create(
    model=${JSON.stringify(modelId)},
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)

for chunk in stream:
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="")`;
    // Metrics cards
    const dashBalance = document.getElementById('dash-balance-val');
    if (dashBalance) dashBalance.textContent = `$${parseFloat(balance || 0).toFixed(4)}`;

    const dashPaid = document.getElementById('dash-paid-val');
    if (dashPaid) dashPaid.textContent = `$${parseFloat(store.state.paidBalance || user?.paid_balance || 0).toFixed(4)}`;

    const dashTrial = document.getElementById('dash-trial-val');
    if (dashTrial) dashTrial.textContent = `$${parseFloat(store.state.trialBalance || user?.trial_balance || 0).toFixed(4)}`;

    const dashKeys = document.getElementById('dash-keys-val');
    const activeKeysCount = keys.filter(k => k.status === 'active' || k.is_active).length;
    if (dashKeys) dashKeys.textContent = activeKeysCount;

    const dashModels = document.getElementById('dash-models-val');
    if (dashModels) dashModels.textContent = models.length;

    const dashStatus = document.getElementById('dash-status-val');
    if (dashStatus) { dashStatus.textContent = health.status === 'healthy' ? 'Responding' : 'Unavailable'; dashStatus.style.color = health.status === 'healthy' ? 'var(--success)' : 'var(--text-muted)'; }

    // Referral Metrics Card
    this.renderReferralCard();

    // System Health details card
    const workersContainer = document.getElementById('dash-workers-list');
    if (workersContainer) {
      if (health.workers && health.workers.length > 0) {
        workersContainer.innerHTML = health.workers.map(w => `
          <div style="display:flex; justify-content:space-between; align-items:center; padding: 10px 0; border-bottom: 1px solid var(--border-subtle);">
            <div>
              <div style="font-weight:600; color:var(--text-primary);">${escapeHtml(w.worker_id)}</div>
              <div style="font-size:var(--text-2xs); color:var(--text-muted); font-family:var(--font-mono);">${escapeHtml(w.url)}</div>
            </div>
            <span class="badge">${escapeHtml(w.status || w.health || 'Unknown')}</span>
          </div>
        `).join('');
      } else {
        workersContainer.innerHTML = `
          <div style="display:flex; justify-content:space-between; align-items:center; padding: 10px 0;">
            <div>
              <div style="font-weight:600; color:var(--text-primary);">No workers reported</div>
              <div style="font-size:var(--text-2xs); color:var(--text-muted);">Connect a worker to see infrastructure status.</div>
            </div>
            <span class="badge">Unavailable</span>
          </div>
        `;
      }
    }
  }

  async renderReferralCard() {
    const { user } = store.state;
    if (!user) return;
    try {
      const stats = await api.getReferrals();
      const earningsEl = document.getElementById('dash-ref-earnings');
      if (earningsEl) earningsEl.textContent = `$${parseFloat(stats.total_earnings_usd || 0).toFixed(2)} USD`;
      const totalEl = document.getElementById('dash-ref-total');
      if (totalEl) totalEl.textContent = stats.total_referrals || 0;
      const pendingEl = document.getElementById('dash-ref-pending');
      if (pendingEl) pendingEl.textContent = stats.pending_referrals || 0;
      const convertedEl = document.getElementById('dash-ref-converted');
      if (convertedEl) convertedEl.textContent = stats.converted_referrals || 0;
      const linkInput = document.getElementById('dash-ref-link-input');
      const shareUrl = stats.referral_link || `${window.location.origin}/?ref=${stats.referral_code}`;
      if (linkInput) linkInput.value = shareUrl;

      const copyBtn = document.getElementById('dash-copy-ref-btn');
      if (copyBtn) {
        copyBtn.onclick = () => copyToClipboard(shareUrl, 'Referral link copied to clipboard!');
      }
    } catch {
      // In case referral stats fail or not loaded yet
    }
  }

  // =========================================================================
  // View: Keys Management
  // =========================================================================
  renderKeysView() {
    const { keys } = store.state;
    const tbody = document.getElementById('keys-table-body');
    const emptyState = document.getElementById('keys-empty-state');
    const tableContainer = document.getElementById('keys-table-container');

    if (!tbody) return;

    if (!keys || keys.length === 0) {
      if (emptyState) emptyState.style.display = 'flex';
      if (tableContainer) tableContainer.style.display = 'none';
      return;
    }

    if (emptyState) emptyState.style.display = 'none';
    if (tableContainer) tableContainer.style.display = 'block';

    tbody.innerHTML = keys.map(k => {
      const isKeyActive = k.status === 'active' || (k.is_active && k.status !== 'revoked' && k.status !== 'expired');
      const badgeClass = isKeyActive ? 'badge-active' : (k.status === 'expired' ? 'badge-expired' : 'badge-revoked');
      const statusLabel = isKeyActive ? 'Active' : (k.status === 'expired' ? 'Expired' : 'Revoked');

      const createdDate = k.created_at ? new Date(k.created_at).toLocaleDateString() : 'N/A';
      const lastUsed = k.last_used_at ? new Date(k.last_used_at).toLocaleDateString() : 'Never';

      let scopeLabel = 'Custom Scopes';
      const perms = k.permissions || '';
      if (perms.includes('admin') || perms.includes('*')) scopeLabel = 'Full Access';
      else if (perms.includes('chat') && !perms.includes('files') && !perms.includes('storage')) scopeLabel = 'Inference';
      else if (perms.includes('storage') || perms.includes('buckets')) scopeLabel = 'Storage';
      else if (perms.includes('fine_tuning') || perms.includes('training')) scopeLabel = 'Fine-Tuning';
      else if (!perms.includes('write')) scopeLabel = 'Read-Only';

      return `
        <tr>
          <td>
            <strong style="color:var(--text-primary);">${escapeHtml(k.name || 'default')}</strong>
            <div style="font-size:11px; color:#8cd867; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; margin-top:2px;">${scopeLabel}</div>
          </td>
          <td>
            <div class="masked-key-container">
              <span>${escapeHtml(k.prefix)}...</span>
              <button class="copy-btn" data-copy="${escapeHtml(k.prefix)}" title="Copy key prefix">
                <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>
              </button>
            </div>
          </td>
          <td>
            <span class="badge ${badgeClass}"><span class="badge-dot"></span>${statusLabel}</span>
          </td>
          <td><span style="font-family:var(--font-mono); color:var(--text-primary);">$${parseFloat(k.credit_balance || 0).toFixed(4)}</span></td>
          <td><span style="font-size:var(--text-2xs); color:var(--text-muted);">${k.rpm_limit} RPM / ${(k.tpm_limit/1000).toFixed(0)}k TPM</span></td>
          <td><span style="font-size:var(--text-xs); color:var(--text-muted);">${createdDate}</span></td>
          <td>
            ${isKeyActive ? `
              <button class="btn btn-danger btn-sm revoke-key-btn" data-id="${k.id}" data-name="${escapeHtml(k.name || k.prefix)}">
                Revoke
              </button>
            ` : `<span style="font-size:var(--text-2xs); color:var(--text-muted);">Disabled</span>`}
          </td>
        </tr>
      `;
    }).join('');

    // Attach copy & revoke events
    tbody.querySelectorAll('.copy-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        copyToClipboard(btn.dataset.copy, 'Key prefix copied!');
      });
    });

    tbody.querySelectorAll('.revoke-key-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        this.keyToRevoke = { id: btn.dataset.id, name: btn.dataset.name };
        document.getElementById('revoke-key-name-display').textContent = btn.dataset.name;
        openModal('modal-revoke-confirm');
      });
    });
  }

  // =========================================================================
  // View: Models Catalog
  // =========================================================================
  renderModelsView() {
    const { models } = store.state;
    const grid = document.getElementById('models-grid');
    if (!grid) return;

    if (!models || models.length === 0) {
      grid.innerHTML = `
        <div class="card" style="grid-column: 1 / -1; text-align:center; padding: var(--space-8);">
          <p style="color:var(--text-secondary);">No models registered yet.</p>
        </div>
      `;
      return;
    }

    const query = document.getElementById('model-search').value.trim().toLowerCase();
    const filtered = models.filter(m => (m.id || m.name || '').toLowerCase().includes(query));
    document.getElementById('model-count').textContent = `${filtered.length} of ${models.length} models`;
    const price = value => value == null ? 'Not listed' : `$${Number(value).toFixed(2)}`;
    grid.innerHTML = filtered.length ? filtered.map(m => {
      const modelId = m.id || m.name;
      const shortName = modelId.split('/').pop();
      return `<article class="card model-card"><div class="model-card-top"><span class="model-monogram">${escapeHtml(shortName.slice(0,1))}</span><span class="badge">Registered</span></div><h3>${escapeHtml(shortName)}</h3><p class="model-id">${escapeHtml(modelId)}</p><p class="model-description">Open-weight language model accessible through your deployment’s inference API.</p><dl class="model-specs"><div><dt>Context window</dt><dd>${m.context_length ? Number(m.context_length).toLocaleString() + ' tokens' : 'Not listed'}</dd></div><div><dt>Input / 1M tokens</dt><dd>${price(m.prompt_price_per_million)}</dd></div><div><dt>Output / 1M tokens</dt><dd>${price(m.completion_price_per_million)}</dd></div></dl><button class="btn btn-secondary test-model-btn" data-model="${escapeHtml(modelId)}">Try in playground <span>↗</span></button></article>`;
    }).join('') : '<div class="card empty-state"><h3>No matching models</h3><p>Try a different name or clear the search.</p></div>';

    grid.querySelectorAll('.test-model-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        store.setSelectedModel(btn.dataset.model);
        this.showAppView('playground');
      });
    });
  }

  // =========================================================================
  // View: Usage & Billing
  // =========================================================================
  renderUsageView() {
    const { balance, keys, models } = store.state;
    const select = document.getElementById('pricing-model');
    const previous = select.value;
    select.innerHTML = models.map(m => `<option value="${escapeHtml(m.id || m.name)}">${escapeHtml(m.id || m.name)}</option>`).join('');
    if (models.some(m => (m.id || m.name) === previous)) select.value = previous;
    const model = models.find(m => (m.id || m.name) === select.value);
    document.getElementById('usage-input-price').textContent = model?.prompt_price_per_million == null ? '—' : `$${Number(model.prompt_price_per_million).toFixed(2)}`;
    document.getElementById('usage-output-price').textContent = model?.completion_price_per_million == null ? '—' : `$${Number(model.completion_price_per_million).toFixed(2)}`;
    document.querySelectorAll('.usage-model-label').forEach(label => { label.textContent = model ? model.id || model.name : 'No model pricing available'; });
    const usageBalance = document.getElementById('usage-total-balance');
    if (usageBalance) {
      usageBalance.textContent = `$${parseFloat(balance || 0).toFixed(4)} USD`;
    }
    const usagePaid = document.getElementById('usage-paid-val');
    if (usagePaid) {
      usagePaid.textContent = `$${parseFloat(store.state.paidBalance || store.state.user?.paid_balance || 0).toFixed(4)}`;
    }
    const usageTrial = document.getElementById('usage-trial-val');
    if (usageTrial) {
      usageTrial.textContent = `$${parseFloat(store.state.trialBalance || store.state.user?.trial_balance || 0).toFixed(4)}`;
    }

    this.renderBillingPackages();

    const breakdownList = document.getElementById('usage-keys-breakdown');
    if (breakdownList) {
      if (!keys.length) {
        breakdownList.innerHTML = '<div class="empty-state"><h3>No credits to display yet.</h3><p>Create an API key to get started.</p></div>';
        return;
      }
      const total = keys.reduce((sum,k) => sum + Number(k.credit_balance || 0),0);
      breakdownList.innerHTML = keys.map(k => `
        <div style="display:flex; justify-content:space-between; align-items:center; padding: 12px 0; border-bottom: 1px solid var(--border-subtle);">
          <div>
            <div style="font-weight:600; color:var(--text-primary);">${escapeHtml(k.name || 'default')} (${escapeHtml(k.prefix)}...)</div>
            <div style="font-size:var(--text-xs); color:var(--text-muted);">Quotas: ${k.rpm_limit} RPM | ${k.tpm_limit} TPM</div><div class="credit-bar"><span style="width:${total > 0 ? Math.min(100, Math.max(0, Number(k.credit_balance || 0) / total * 100)) : 0}%"></span></div>
          </div>
          <div style="font-family:var(--font-mono); font-weight:700; color:var(--success);">
            $${parseFloat(k.credit_balance || 0).toFixed(4)}
          </div>
        </div>
      `).join('');
    }
  }

  async renderBillingPackages() {
    const container = document.getElementById('billing-packages');
    if (!container) return;
    if (!this.creditPackages.length) {
      try {
        const result = await api.getCreditPackages();
        this.creditPackages = Array.isArray(result?.packages) ? result.packages : [];
      } catch {
        container.innerHTML = '<span class="billing-loading">Payments are being prepared. Please try again shortly.</span>';
        return;
      }
    }
    if (!this.creditPackages.length) {
      container.innerHTML = '<span class="billing-loading">No credit packages are available right now.</span>';
      return;
    }
    container.innerHTML = this.creditPackages.map((amount, index) => `
      <button class="billing-package ${index === 1 ? 'featured' : ''}" type="button" data-checkout-amount="${Number(amount)}">
        <span>ONE-TIME TOP-UP</span><strong>$${Number(amount).toFixed(0)}</strong><small>${Number(amount).toFixed(2)} API credit</small><b>Continue ↗</b>
      </button>`).join('') + '<p class="billing-policy-note">Prepaid API credits are generally non-refundable, subject to mandatory rights. <a href="/legal/billing">Read billing &amp; refund terms before purchase ↗</a></p>';
    container.querySelectorAll('[data-checkout-amount]').forEach(button => button.addEventListener('click', async () => {
      const amount = Number(button.dataset.checkoutAmount);
      const activeKey = store.state.selectedKey || store.state.keys.find(key => key.is_active);
      if (!activeKey?.id) {
        showToast('Create an active API key before adding credits.', 'error');
        this.showAppView('keys');
        return;
      }
      const original = button.innerHTML;
      button.disabled = true;
      button.innerHTML = '<span>PREPARING</span><strong>…</strong><small>Opening secure checkout</small>';
      try {
        const checkout = await api.createCheckout(amount, activeKey.id);
        window.location.assign(checkout.checkout_url);
      } catch (err) {
        showToast(err.message || 'Unable to start checkout.', 'error');
        button.disabled = false;
        button.innerHTML = original;
      }
    }));
  }

  // =========================================================================
  // View: Playground
  // =========================================================================
  renderPlaygroundView() {
    const { models, selectedModel, keys, selectedKey } = store.state;

    // Populate Models Select
    const modelSelect = document.getElementById('playground-model-select');
    if (modelSelect) {
      modelSelect.innerHTML = models.map(m => {
        const id = m.id || m.name;
        return `<option value="${escapeHtml(id)}" ${id === selectedModel ? 'selected' : ''}>${escapeHtml(id)}</option>`;
      }).join('');
    }

    // Populate API Keys Select
    const keySelect = document.getElementById('playground-key-select');
    const activeKeys = keys.filter(k => k.status === 'active' || k.is_active);
    if (keySelect) {
      if (activeKeys.length > 0) {
        keySelect.innerHTML = activeKeys.map(k => `
          <option value="${escapeHtml(k.prefix)}" ${selectedKey && selectedKey.id === k.id ? 'selected' : ''}>
            ${escapeHtml(k.name || 'Key')} (${escapeHtml(k.prefix)}... - $${parseFloat(k.credit_balance || 0).toFixed(2)})
          </option>
        `).join('');
      } else {
        keySelect.innerHTML = `<option value="">No active keys - Create one in API Keys</option>`;
      }
    }

    this.renderChatMessages();
  }

  renderChatMessages() {
    const thread = document.getElementById('chat-thread');
    if (!thread) return;

    const { chatHistory } = store.state;

    if (chatHistory.length === 0) {
      thread.innerHTML = `
        <div class="empty-state">
          <div class="empty-state-icon">
            <svg width="48" height="48" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"/></svg>
          </div>
          <h3 style="font-weight:700; color:var(--text-primary);">A blank canvas for your next idea.</h3>
          <p style="font-size:var(--text-sm); max-width: 440px;">
            Choose a model, connect your API key, and start a conversation. Response timing and token counts appear as you experiment.
          </p>
        </div>
      `;
      return;
    }

    thread.innerHTML = chatHistory.map(msg => {
      const isUser = msg.role === 'user';
      const roleLabel = isUser ? 'You' : 'SpeedInfer';
      const avatarClass = isUser ? 'user-avatar' : 'assistant-avatar';

      let metricsHtml = '';
      if (!isUser && msg.metrics) {
        const { ttft, totalTimeMs, promptTokens, completionTokens, totalTokens } = msg.metrics;
        metricsHtml = `
          <div class="message-metrics-tag">
            <span class="metric-pill">TTFT: <strong>${ttft}ms</strong></span>
            <span class="metric-pill">Latency: <strong>${totalTimeMs}ms</strong></span>
            <span class="metric-pill">Tokens: <strong>${promptTokens} in / ${completionTokens} out</strong></span>
          </div>
        `;
      }

      return `
        <div class="chat-message ${msg.role}">
          <div class="chat-avatar ${avatarClass}">${isUser ? 'U' : 'AI'}</div>
          <div class="chat-bubble-container">
            <div class="chat-bubble">${escapeHtml(msg.content)}${msg.isStreaming ? '<span class="stream-cursor"></span>' : ''}</div>
            ${metricsHtml}
          </div>
        </div>
      `;
    }).join('');

    thread.scrollTop = thread.scrollHeight;
  }

  // =========================================================================
  // Chat Execution & SSE Streaming
  // =========================================================================
  async sendChatMessage() {
    if (this.isStreaming) return;

    const input = document.getElementById('chat-input-textarea');
    const promptText = input.value.trim();
    if (!promptText) return;

    // Resolve API key to use
    let rawApiKey = null;
    const customKeyInput = document.getElementById('playground-custom-key');
    if (customKeyInput && customKeyInput.value.trim()) {
      rawApiKey = customKeyInput.value.trim();
    } else {
      showToast('Enter an API key you created to use live inference.', 'info');
      customKeyInput?.focus();
      return;
    }

    const modelSelect = document.getElementById('playground-model-select');
    const model = modelSelect ? modelSelect.value : 'Qwen/Qwen2.5-7B-Instruct';
    const tempInput = document.getElementById('param-temperature');
    const temperature = tempInput ? parseFloat(tempInput.value) : 0.7;
    const maxTokensInput = document.getElementById('param-max-tokens');
    const maxTokens = maxTokensInput ? parseInt(maxTokensInput.value, 10) : 1024;
    const systemPromptInput = document.getElementById('param-system-prompt');
    const systemPrompt = systemPromptInput ? systemPromptInput.value.trim() : '';

    // Add user message
    store.addChatMessage({ role: 'user', content: promptText });
    input.value = '';

    // Add placeholder assistant message
    store.addChatMessage({ role: 'assistant', content: '', isStreaming: true });
    this.isStreaming = true;

    const sendBtn = document.getElementById('chat-send-btn');
    if (sendBtn) sendBtn.disabled = true;

    // Build OpenAI messages array
    const messages = [];
    if (systemPrompt) {
      messages.push({ role: 'system', content: systemPrompt });
    }
    messages.push({ role: 'user', content: promptText });

    await api.streamChatCompletion({
      model,
      messages,
      temperature,
      max_tokens: maxTokens,
      apiKey: rawApiKey,
      onChunk: ({ delta, fullText, ttft }) => {
        store.updateLastAssistantMessage(fullText);
        this.renderChatMessages();
      },
      onDone: (metrics) => {
        this.isStreaming = false;
        if (sendBtn) sendBtn.disabled = false;
        const history = [...store.state.chatHistory];
        const lastMsg = history[history.length - 1];
        if (lastMsg) {
          lastMsg.isStreaming = false;
          lastMsg.metrics = metrics;
          store.setState({ chatHistory: history });
        }
        this.renderChatMessages();

        // Refresh balance after completion
        api.getMe().then(user => store.setUser(user)).catch(() => {});
      },
      onError: (err) => {
        this.isStreaming = false;
        if (sendBtn) sendBtn.disabled = false;
        const history = [...store.state.chatHistory];
        const lastMsg = history[history.length - 1];
        if (lastMsg) {
          lastMsg.isStreaming = false;
          lastMsg.content = `Error: ${err.message}`;
          store.setState({ chatHistory: history });
        }
        this.renderChatMessages();
        showToast(err.message, 'error');
      },
    });
  }

  // =========================================================================
  // Event Bindings
  // =========================================================================
  bindGlobalEvents() {
    // Navigation items click
    document.querySelectorAll('.nav-item').forEach(el => {
      el.addEventListener('click', () => {
        const view = el.dataset.view;
        if (view) this.showAppView(view);
      });
    });

    // Mobile Hamburger button
    document.querySelector('.mobile-menu-btn')?.addEventListener('click', () => {
      document.querySelector('.app-sidebar')?.classList.add('mobile-open');
      document.querySelector('.sidebar-backdrop')?.classList.add('active');
    });

    // Sidebar Backdrop click
    document.querySelector('.sidebar-backdrop')?.addEventListener('click', () => {
      document.querySelector('.app-sidebar')?.classList.remove('mobile-open');
      document.querySelector('.sidebar-backdrop')?.classList.remove('active');
    });

    // Logout button
    document.getElementById('logout-btn')?.addEventListener('click', () => {
      api.clearSession();
      this.workspace.reset();
      document.getElementById('playground-custom-key').value = '';
      clearInbox();
      store.setState({user:null, keys:[], balance:0, models:[], chatHistory:[], selectedKey:null});
      sessionStorage.removeItem('speedinfer_last_raw_key');
      showToast('Logged out successfully', 'info');
      this.showAuthView('login');
    });
  }

  bindAuthEvents() {
    // Auth Tab Switchers
    document.getElementById('tab-btn-login')?.addEventListener('click', () => this.switchAuthTab('login'));
    document.getElementById('tab-btn-register')?.addEventListener('click', () => this.switchAuthTab('register'));

    // Forgot password link in login form
    document.getElementById('btn-forgot-password')?.addEventListener('click', () => {
      openModal('modal-forgot-password');
    });

    // Login Form Submit
    document.getElementById('login-form')?.addEventListener('submit', async (e) => {
      e.preventDefault();
      const email = document.getElementById('login-email').value.trim();
      const password = document.getElementById('login-password').value;
      const submitBtn = document.getElementById('login-submit-btn');

      const formData = new FormData(document.getElementById('login-form'));
      const turnstileToken = formData.get('cf-turnstile-response') || (window.turnstile?.getResponse ? window.turnstile.getResponse('#turnstile-login') : null);

      try {
        submitBtn.disabled = true;
        submitBtn.textContent = 'Signing in...';
        await api.login(email, password, turnstileToken);
        await this.loadInitialData();
        showToast('Login successful!', 'success');
        this.showAppView('dashboard');

        if (store.state.user && !store.state.user.is_verified) {
          const targetEmail = document.getElementById('verify-email-target');
          if (targetEmail) targetEmail.textContent = email;
          openModal('modal-verify-email');
        }
      } catch (err) {
        showToast(err.message, 'error');
      } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Sign In';
        if (window.turnstile) {
          try { window.turnstile.reset('#turnstile-login'); } catch {}
        }
      }
    });

    // Register Form Submit
    document.getElementById('register-form')?.addEventListener('submit', async (e) => {
      e.preventDefault();
      const name = document.getElementById('register-name').value.trim();
      const email = document.getElementById('register-email').value.trim();
      const password = document.getElementById('register-password').value;
      const referralCode = document.getElementById('register-referral')?.value.trim() || sessionStorage.getItem('speedinfer_referral_code') || null;
      const submitBtn = document.getElementById('register-submit-btn');

      const formData = new FormData(document.getElementById('register-form'));
      const turnstileToken = formData.get('cf-turnstile-response') || (window.turnstile?.getResponse ? window.turnstile.getResponse('#turnstile-register') : null);

      try {
        submitBtn.disabled = true;
        submitBtn.textContent = 'Creating Account...';
        const deviceFp = await generateDeviceFingerprint();

        await api.register({
          name,
          email,
          password,
          initial_balance: 0.0,
          create_api_key: false,
          accepted_policy_version: '2026-09-27-r2',
          referral_code: referralCode,
          device_fingerprint: deviceFp,
          turnstile_token: turnstileToken,
        });

        await this.loadInitialData();
        showToast('Account created! Please enter the 6-digit code sent to your email to verify.', 'success');
        this.showAppView('dashboard');

        const targetEmail = document.getElementById('verify-email-target');
        if (targetEmail) targetEmail.textContent = email;
        openModal('modal-verify-email');
      } catch (err) {
        showToast(err.message, 'error');
      } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Create workspace';
        if (window.turnstile) {
          try { window.turnstile.reset('#turnstile-register'); } catch {}
        }
      }
    });
  }

  openProfileModal() {
    const { user } = store.state;
    if (!user) return;

    document.getElementById('profile-modal-display-name').textContent = user.name || 'Developer';
    document.getElementById('profile-modal-display-email').textContent = user.email;
    document.getElementById('profile-name-input').value = user.name || '';
    document.getElementById('profile-avatar-input').value = user.avatar_url || '';
    document.getElementById('profile-location-input').value = user.location || '';
    document.getElementById('profile-org-input').value = user.organization || '';
    document.getElementById('profile-referral-code').value = user.referral_code || '';

    const badgeEl = document.getElementById('profile-modal-badge');
    if (badgeEl) {
      if (user.is_verified) {
        badgeEl.textContent = 'Verified ✓';
        badgeEl.style.background = 'rgba(16,185,129,0.1)';
        badgeEl.style.color = 'var(--success)';
      } else {
        badgeEl.textContent = 'Unverified ⚠️';
        badgeEl.style.background = '#fff3cd';
        badgeEl.style.color = '#856404';
      }
    }

    this.updateAvatarPreview(user.avatar_url, user.name || user.email);
    openModal('modal-profile');
  }

  updateAvatarPreview(url, fallbackText) {
    const previewEl = document.getElementById('profile-modal-avatar-preview');
    if (!previewEl) return;
    if (url && url.trim()) {
      previewEl.innerHTML = `<img src="${escapeHtml(url.trim())}" alt="Avatar" style="width:100%; height:100%; border-radius:50%; object-fit:cover;">`;
    } else {
      previewEl.textContent = (fallbackText || 'S').slice(0, 1).toUpperCase();
    }
  }

  bindModalEvents() {
    document.addEventListener('keydown', event => {
      const modal = document.querySelector('.modal-overlay.active');
      if (!modal) return;
      if (event.key === 'Escape') { event.preventDefault(); closeModal(modal.id); }
      if (event.key !== 'Tab') return;
      const focusable = [...modal.querySelectorAll('button, input, textarea, select, a[href], [tabindex="0"]')].filter(el => !el.disabled && el.getClientRects().length);
      const first = focusable[0], last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    });

    // Header Profile & Verify Badge Click Handlers
    document.getElementById('header-profile-btn')?.addEventListener('click', () => {
      this.openProfileModal();
    });
    document.getElementById('nav-profile-btn')?.addEventListener('click', () => {
      this.openProfileModal();
    });
    document.addEventListener('click', (e) => {
      if (e.target.closest('#ws-edit-profile-btn')) {
        this.openProfileModal();
      }
    });
    document.getElementById('header-verify-badge')?.addEventListener('click', () => {
      const { user } = store.state;
      if (user) {
        const targetEmail = document.getElementById('verify-email-target');
        if (targetEmail) targetEmail.textContent = user.email;
        openModal('modal-verify-email');
      }
    });

    // Email Verification Form Submit
    document.getElementById('verify-email-form')?.addEventListener('submit', async (e) => {
      e.preventDefault();
      const code = document.getElementById('verify-code-input').value.trim();
      const submitBtn = document.getElementById('btn-submit-verify');
      const { user } = store.state;
      if (!user) return;

      try {
        submitBtn.disabled = true;
        submitBtn.textContent = 'Verifying...';
        const res = await api.verifyEmail({ email: user.email, code });

        showToast('Email verified successfully!', 'success');

        closeModal('modal-verify-email');
        const updatedUser = await api.getMe();
        store.setUser(updatedUser);
        this.renderHeader();
      } catch (err) {
        showToast(err.message, 'error');
      } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Verify Email';
      }
    });

    // Resend Verification Code Button
    document.getElementById('btn-resend-verification')?.addEventListener('click', async () => {
      const resendBtn = document.getElementById('btn-resend-verification');
      const { user } = store.state;
      if (!user) return;

      try {
        resendBtn.disabled = true;
        await api.resendCode({ email: user.email, purpose: 'registration' });
        showToast('A new 6-digit verification code has been sent to your email.', 'info');

        let cooldown = 60;
        resendBtn.textContent = `Resend in ${cooldown}s`;
        const interval = setInterval(() => {
          cooldown -= 1;
          if (cooldown <= 0) {
            clearInterval(interval);
            resendBtn.disabled = false;
            resendBtn.textContent = 'Resend code';
          } else {
            resendBtn.textContent = `Resend in ${cooldown}s`;
          }
        }, 1000);
      } catch (err) {
        resendBtn.disabled = false;
        showToast(err.message, 'error');
      }
    });

    // Forgot Password Form Submit
    document.getElementById('forgot-password-form')?.addEventListener('submit', async (e) => {
      e.preventDefault();
      const email = document.getElementById('forgot-email-input').value.trim();
      const submitBtn = document.getElementById('btn-submit-forgot');
      const formData = new FormData(document.getElementById('forgot-password-form'));
      const turnstileToken = formData.get('cf-turnstile-response') || (window.turnstile?.getResponse ? window.turnstile.getResponse('#turnstile-forgot') : null);

      try {
        submitBtn.disabled = true;
        submitBtn.textContent = 'Sending...';
        await api.forgotPassword({ email, turnstile_token: turnstileToken });
        showToast('If your email is registered, a 6-digit reset code has been sent.', 'info');
        closeModal('modal-forgot-password');

        // Pre-fill email in reset password modal and open it
        document.getElementById('reset-email-input').value = email;
        document.getElementById('reset-code-input').value = '';
        document.getElementById('reset-new-password').value = '';
        openModal('modal-reset-password');
      } catch (err) {
        showToast(err.message, 'error');
      } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Send Reset Code';
        if (window.turnstile) {
          try { window.turnstile.reset('#turnstile-forgot'); } catch {}
        }
      }
    });

    // Reset Password Form Submit
    document.getElementById('reset-password-form')?.addEventListener('submit', async (e) => {
      e.preventDefault();
      const email = document.getElementById('reset-email-input').value.trim();
      const code = document.getElementById('reset-code-input').value.trim();
      const newPassword = document.getElementById('reset-new-password').value;
      const submitBtn = document.getElementById('btn-submit-reset');

      try {
        submitBtn.disabled = true;
        submitBtn.textContent = 'Updating...';
        await api.resetPassword({ email, code, new_password: newPassword });
        showToast('Password reset successfully! Please sign in with your new password.', 'success');
        closeModal('modal-reset-password');
        this.showAuthView('login');
      } catch (err) {
        showToast(err.message, 'error');
      } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Update Password';
      }
    });

    // Profile Avatar live preview listener
    document.getElementById('profile-avatar-input')?.addEventListener('input', (e) => {
      const { user } = store.state;
      this.updateAvatarPreview(e.target.value, user?.name || user?.email);
    });

    // Profile Avatar file upload
    document.getElementById('profile-upload-file-btn')?.addEventListener('click', () => {
      document.getElementById('profile-avatar-file')?.click();
    });

    document.getElementById('profile-avatar-file')?.addEventListener('change', (e) => {
      const file = e.target.files?.[0];
      if (!file) return;
      if (file.size > 2 * 1024 * 1024) {
        showToast('Avatar file size must be less than 2MB', 'error');
        return;
      }
      const reader = new FileReader();
      reader.onload = (event) => {
        const base64Data = event.target.result;
        const avatarInput = document.getElementById('profile-avatar-input');
        if (avatarInput) avatarInput.value = base64Data;
        const { user } = store.state;
        this.updateAvatarPreview(base64Data, user?.name || user?.email);
      };
      reader.readAsDataURL(file);
    });

    // Copy referral link in profile modal
    document.getElementById('profile-copy-ref-btn')?.addEventListener('click', () => {
      const code = document.getElementById('profile-referral-code').value;
      if (code) {
        const url = `${window.location.origin}/?ref=${code}`;
        copyToClipboard(url, 'Referral link copied to clipboard!');
      }
    });

    // Profile Form Submit
    document.getElementById('profile-form')?.addEventListener('submit', async (e) => {
      e.preventDefault();
      const name = document.getElementById('profile-name-input').value.trim();
      const avatar_url = document.getElementById('profile-avatar-input').value.trim();
      const location = document.getElementById('profile-location-input').value.trim();
      const organization = document.getElementById('profile-org-input').value.trim();
      const submitBtn = document.getElementById('profile-save-btn');

      try {
        submitBtn.disabled = true;
        submitBtn.textContent = 'Saving...';
        const updated = await api.updateProfile({ name, avatar_url, location, organization });
        store.setUser(updated);
        this.renderHeader();
        showToast('Profile updated successfully!', 'success');
        closeModal('modal-profile');
      } catch (err) {
        showToast(err.message, 'error');
      } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Save Changes';
      }
    });

    // Open Create Key Modal
    document.getElementById('open-create-key-modal-btn')?.addEventListener('click', () => {
      openModal('modal-create-key');
    });

    // Close Modals
    document.querySelectorAll('.modal-close-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const modal = btn.closest('.modal-overlay');
        if (modal) closeModal(modal.id);
      });
    });

    // Create Key Preset Change
    const presetEl = document.getElementById('new-key-preset');
    const customContainer = document.getElementById('custom-permissions-container');
    presetEl?.addEventListener('change', () => {
      if (presetEl.value === 'custom') {
        if (customContainer) customContainer.style.display = 'block';
      } else {
        if (customContainer) customContainer.style.display = 'none';
      }
    });

    // Create Key Form Submit
    document.getElementById('create-key-form')?.addEventListener('submit', async (e) => {
      e.preventDefault();
      const name = document.getElementById('new-key-name').value.trim();
      const rpm = parseInt(document.getElementById('new-key-rpm').value, 10) || 60;
      const tpm = parseInt(document.getElementById('new-key-tpm').value, 10) || 60000;
      const submitBtn = document.getElementById('create-key-submit-btn');

      let permissions = 'admin,chat:completions,completions,models:read,usage:read,files:*,storage:*,fine_tuning:*,keys:*';
      const preset = presetEl ? presetEl.value : 'all';
      if (preset === 'inference') {
        permissions = 'chat:completions,completions,models:read';
      } else if (preset === 'training') {
        permissions = 'fine_tuning:read,fine_tuning:write,files:read,files:write,models:read';
      } else if (preset === 'storage') {
        permissions = 'storage:read,storage:write,files:read,files:write';
      } else if (preset === 'readonly') {
        permissions = 'models:read,usage:read,files:read,storage:read,fine_tuning:read,keys:read';
      } else if (preset === 'custom') {
        const checked = Array.from(document.querySelectorAll('.scope-checkbox:checked')).map(cb => cb.value);
        permissions = checked.join(',') || 'models:read';
      }

      try {
        submitBtn.disabled = true;
        submitBtn.textContent = 'Generating...';
        const newKey = await api.createKey({
          name,
          rpm_limit: rpm,
          tpm_limit: tpm,
          permissions,
        });

        closeModal('modal-create-key');
        // Refresh keys
        const keysResp = await api.listKeys();
        store.setKeys(keysResp.data || []);
        store.setUser(await api.getMe());

        // Reveal only in this dialog; never persist the raw key in browser storage.
        if (newKey.key) {
          document.getElementById('modal-secret-key-val').textContent = newKey.key;
          openModal('modal-secret-key-reveal');
        }

        showToast(`Key "${name}" created successfully!`, 'success');
      } catch (err) {
        showToast(err.message, 'error');
      } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Generate API Key';
      }
    });

    // Copy Secret Key from Reveal Modal
    document.getElementById('copy-revealed-key-btn')?.addEventListener('click', () => {
      const keyVal = document.getElementById('modal-secret-key-val').textContent;
      copyToClipboard(keyVal, 'API Key copied to clipboard! Save it securely.');
    });

    // Revoke Key Confirmation Submit
    document.getElementById('confirm-revoke-btn')?.addEventListener('click', async () => {
      if (!this.keyToRevoke) return;
      const btn = document.getElementById('confirm-revoke-btn');

      try {
        btn.disabled = true;
        btn.textContent = 'Revoking...';
        await api.revokeKey(this.keyToRevoke.id, false);
        closeModal('modal-revoke-confirm');
        showToast(`Key "${this.keyToRevoke.name}" has been revoked.`, 'info');

        // Refresh keys
        const keysResp = await api.listKeys();
        store.setKeys(keysResp.data || []);
      } catch (err) {
        showToast(err.message, 'error');
      } finally {
        btn.disabled = false;
        btn.textContent = 'Revoke Key';
        this.keyToRevoke = null;
      }
    });
  }

  bindPlaygroundEvents() {
    // Quick prompt chips
    document.querySelectorAll('.quick-prompt-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        const textarea = document.getElementById('chat-input-textarea');
        if (textarea) {
          textarea.value = chip.dataset.prompt;
          textarea.focus();
        }
      });
    });

    // Textarea enter key submit (shift+enter for newline)
    const textarea = document.getElementById('chat-input-textarea');
    textarea?.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.sendChatMessage();
      }
    });

    // Send Button click
    document.getElementById('chat-send-btn')?.addEventListener('click', () => {
      this.sendChatMessage();
    });

    // Clear Chat
    document.getElementById('clear-chat-btn')?.addEventListener('click', () => {
      store.clearChat();
      this.renderChatMessages();
    });

    // Temperature slider sync
    const tempSlider = document.getElementById('param-temperature');
    const tempVal = document.getElementById('param-temperature-val');
    tempSlider?.addEventListener('input', () => {
      if (tempVal) tempVal.textContent = tempSlider.value;
    });

    // Max tokens slider sync
    const maxSlider = document.getElementById('param-max-tokens');
    const maxVal = document.getElementById('param-max-tokens-val');
    maxSlider?.addEventListener('input', () => {
      if (maxVal) maxVal.textContent = maxSlider.value;
    });
  }
}

// Bootstrap Application
document.addEventListener('DOMContentLoaded', () => {
  const app = new AppController();
  app.init();
});
