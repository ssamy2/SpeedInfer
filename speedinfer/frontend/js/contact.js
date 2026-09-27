/** Shared, accessible sales/support intake and private administrator inbox. */
import { api } from './api.js?v=20260927-inception-release';
import { store } from './store.js?v=20260927-inception-release';

let inboxOffset = 0;
let inboxLoading = false;
let inboxGeneration = 0;
const inboxPageSize = 50;
const addresses = { sales: 'Sales@speedinfer.com', support: 'Support@speedinfer.com' };
const text = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));

export function initContact() {
  const dialog = document.getElementById('contact-dialog');
  const privacy = document.getElementById('privacy-dialog');
  const form = document.getElementById('contact-form');
  const result = document.getElementById('contact-result');
  const submit = document.getElementById('contact-submit');
  let requestId = null;
  let sending = false;
  let completed = false;

  const updateEmail = category => {
    const link = document.getElementById('contact-direct-email');
    link.textContent = addresses[category];
    link.href = `mailto:${addresses[category]}`;
  };
  document.querySelectorAll('[data-contact]').forEach(button => button.addEventListener('click', () => {
    const category = button.dataset.contact;
    if (!sending && (completed || form.elements.category.value !== category)) {
      form.reset(); requestId = null; completed = false; result.textContent = '';
    }
    form.elements.category.value = category;
    form.elements.name.value ||= store.state.user?.name || '';
    form.elements.email.value ||= store.state.user?.email || '';
    document.getElementById('contact-dialog-eyebrow').textContent = category === 'sales' ? 'LET’S BUILD TOGETHER' : 'WE’RE HERE TO HELP';
    document.getElementById('contact-dialog-title').textContent = category === 'sales' ? 'Tell us what you’re building.' : 'Let’s work through it.';
    document.getElementById('contact-dialog-description').textContent = category === 'sales' ? 'Share your requirements. Our team will follow up by email.' : 'Describe the issue or ask a question. We’ll reply to your email.';
    updateEmail(category);
    submit.disabled = sending;
    if (!sending) submit.innerHTML = 'Send request <span>↗</span>';
    dialog.showModal();
  }));
  document.querySelector('[data-close-contact]').addEventListener('click', () => dialog.close());
  document.querySelectorAll('[data-privacy]').forEach(button => button.addEventListener('click', () => privacy.showModal()));
  document.querySelector('[data-close-privacy]').addEventListener('click', () => privacy.close());

  form.addEventListener('submit', async event => {
    event.preventDefault();
    if (sending || completed || !form.reportValidity()) return;
    requestId ||= crypto.randomUUID();
    const payload = Object.fromEntries(new FormData(form));
    payload.request_id = requestId;
    payload.consent = form.elements.consent.checked;
    sending = true;
    submit.disabled = true;
    submit.textContent = 'Sending…';
    result.className = '';
    result.textContent = '';
    try {
      const response = await api.request('/v1/contact/requests', { method:'POST', body:JSON.stringify(payload), signal:AbortSignal.timeout(30000) });
      result.className = 'request-success';
      result.textContent = `Request received. Our team will follow up at ${payload.email}. Your reference: ${response.reference}.`;
      completed = true;
      form.reset();
      result.focus();
      submit.textContent = 'Request received ✓';
    } catch (error) {
      result.className = 'request-error';
      result.textContent = error.name === 'TimeoutError' || error.name === 'TypeError' ? 'The connection was interrupted. Please try again; retries won’t create a duplicate request. You can also email us directly.' : error.message;
      submit.textContent = 'Try again ↗';
      result.focus();
    } finally {
      sending = false;
      submit.disabled = completed;
    }
  });
  api.request('/v1/contact/options').then(options => {
    for (const category of ['sales','support']) {
      const address = options[`${category}_email`];
      if (typeof address !== 'string' || !/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(address)) continue;
      addresses[category] = address;
      document.querySelectorAll(`[data-${category}-email]`).forEach(link => {
        link.href = `mailto:${address}`;
        link.textContent = `${address} ↗`;
      });
    }
    updateEmail(form.elements.category.value);
  }).catch(() => { /* Direct email links remain available if configuration cannot load. */ });
  document.getElementById('refresh-inbox').addEventListener('click', () => loadInbox());
  document.getElementById('inbox-previous').addEventListener('click', () => { inboxOffset = Math.max(0, inboxOffset - inboxPageSize); loadInbox(); });
  document.getElementById('inbox-next').addEventListener('click', () => { inboxOffset += inboxPageSize; loadInbox(); });
}

export function clearInbox() {
  inboxGeneration++;
  inboxOffset = 0;
  document.getElementById('inbox-list').replaceChildren();
}

export async function loadInbox() {
  const container = document.getElementById('inbox-list');
  if (!store.state.user?.is_admin || inboxLoading) return;
  const generation = inboxGeneration;
  inboxLoading = true;
  const controls = ['inbox-previous','inbox-next','refresh-inbox'].map(id => document.getElementById(id));
  controls.forEach(button => { button.disabled = true; });
  container.textContent = 'Loading requests…';
  try {
    const response = await api.request(`/v1/contact/requests?offset=${inboxOffset}`);
    if (generation !== inboxGeneration || !store.state.user?.is_admin) return;
    const items = response.data;
    container.innerHTML = items.length ? items.map(item => {
      const reply = `mailto:${encodeURIComponent(item.email)}?subject=${encodeURIComponent(`Re: ${item.subject} [${item.id}]`)}`;
      return `<article class="inbox-card card"><div class="inbox-card-top"><span class="badge">${text(item.category)}</span><span>${text(new Date(item.created_at).toLocaleString())}</span><span class="badge ${item.delivery_status === 'sent' ? 'badge-active' : 'badge-expired'}">${item.delivery_status === 'sent' ? 'Team notified' : 'Email notification pending'}</span></div><h3>${text(item.subject)}</h3><p>${text(item.name)} · ${text(item.email)}${item.company ? ` · ${text(item.company)}` : ''}</p><div class="inbox-message">${text(item.message)}</div><div class="inbox-actions"><a class="btn btn-primary" href="${text(reply)}">Reply by email ↗</a>${item.delivery_status !== 'sent' ? `<button class="btn btn-secondary" data-notify="${text(item.id)}">Retry team notification</button>` : ''}<small>Reference: ${text(item.id)}</small></div></article>`;
    }).join('') : '<div class="card empty-state"><h3>You’re all caught up.</h3><p>New sales and support requests will appear here.</p></div>';
    document.getElementById('inbox-page').textContent = `Page ${Math.floor(inboxOffset / inboxPageSize) + 1}`;
    controls[0].disabled = inboxOffset === 0;
    controls[1].disabled = items.length < inboxPageSize;
    container.querySelectorAll('[data-notify]').forEach(button => button.addEventListener('click', async () => {
      button.disabled = true;
      try {
        const result = await api.request(`/v1/contact/requests/${button.dataset.notify}/notify`, {method:'POST'});
        if (result.delivery_status === 'sent') { await loadInbox(); return; }
        button.textContent = 'Email unavailable — reply directly';
      } catch { button.textContent = 'Could not notify — try again'; }
      button.disabled = false;
    }));
  } catch {
    container.textContent = 'Unable to load requests. Check your connection and administrator access, then refresh.';
    controls[0].disabled = inboxOffset === 0;
  } finally {
    controls[2].disabled = false;
    inboxLoading = false;
  }
}
