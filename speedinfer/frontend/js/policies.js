const policies = {
  terms: {title:'Terms of Service', intro:'Legal agreement for developers and enterprises using SpeedInfer.', sections:[
    ['Service Overview','SpeedInfer Technologies provides ultra-low latency, OpenAI-compatible AI model inference, weights distribution, file storage, and fine-tuning orchestration via secure API and developer console.'],
    ['Customer Data & Intellectual Property','You retain 100% of all intellectual property rights in your prompts, datasets, fine-tuned weights, and generated completions. SpeedInfer claims zero ownership over your inputs or outputs.'],
    ['Acceptable Use & Compliance','Users agree to utilize SpeedInfer services in accordance with all applicable international laws and model licenses. Reverse engineering, abuse, Denial of Service attempts, or unauthorized access to other accounts are strictly prohibited.'],
    ['Account & Key Security','You are responsible for safeguarding your API keys and credentials. API keys should be granted least-privilege permissions and stored in secure environment variables. Stored API keys are cryptographically hashed using SHA-256 with secret salt.'],
    ['Service Availability & SLA','SpeedInfer strives for 99.9% service uptime for production workloads with automated health monitoring, load balancing, and multi-region failover. Enterprise custom SLAs are available upon request.'],
    ['Commercial Entity & Contact','Megsy for Digital Platforms Development and E-Commerce L.L.C (Operating SpeedInfer) · CR No. 284191 (Cairo Investment Commercial Registry) · Tax ID: 785-034-774 · Cairo, Egypt. For legal notices, enterprise contracts, or billing inquiries: legal@speedinfer.com or support@speedinfer.com. Review verified leadership and corporate backing at /company/team.'],
  ]},
  privacy: {title:'Privacy Policy & Data Protection',intro:'Enterprise-grade privacy, Zero Data Retention, and compliance standards.',sections:[
    ['Zero Data Retention (ZDR) for Inference','SpeedInfer enforces a strict Zero Data Retention policy for standard API inference. Your prompts and completions are processed ephemerally in volatile memory and are NEVER logged, NEVER stored on disk, and NEVER used to train or fine-tune foundation models.'],
    ['Encryption & Transit Security','All API requests and web sessions are encrypted in transit using industry-standard TLS 1.3 / HTTPS. Internal database records and storage buckets are protected with AES-256 encryption at rest.'],
    ['GDPR & CCPA Compliance','SpeedInfer fully adheres to the EU General Data Protection Regulation (GDPR) and California Consumer Privacy Act (CCPA). Users have the statutory right to request access, export, or permanent deletion of their account data at any time by contacting support@speedinfer.com.'],
    ['Usage & Billing Records','The platform only stores aggregated, non-sensitive usage metrics (token counts, model ID, latency, and calculated fee) in immutable ledger tables for accurate billing verification and audit compliance.'],
    ['Third-Party Subprocessors','Payment transactions are processed securely through certified PCI-DSS compliant providers (Whop). No payment card numbers or sensitive banking details ever touch SpeedInfer servers.'],
    ['Data Deletion & Retention Rights','You may delete your API keys, uploaded datasets, and storage objects at any time through the dashboard or API. Account closure requests are processed with immediate data purging.'],
  ]},
  security: {title:'Security & Infrastructure Architecture',intro:'Defense-in-depth security controls protecting your AI workloads.',sections:[
    ['Data Center & Infrastructure Security','SpeedInfer is hosted in Tier-III/IV enterprise datacenter facilities equipped with 24/7 physical security, biometric access, redundant power, and automated DDoS mitigation.'],
    ['Authentication & Granular RBAC','API requests authenticate using cryptographically salted SHA-256 key hashing with constant-time verification. Granular scopes (e.g. chat:completions, storage:read, training:write) enforce least-privilege security.'],
    ['Isolation & Network Boundaries','Inference workloads, file buckets, and database sessions are strictly isolated between accounts. Network perimeters are guarded with rate limiting, IP clustering defense, and automated anomaly detection.'],
    ['Responsible Disclosure','We welcome security researchers. If you identify a potential vulnerability, please report it immediately to security@speedinfer.com or support@speedinfer.com. We acknowledge and address validated reports promptly.'],
  ]},
  billing: {title:'Billing & Refund Policy',intro:'Transparent, metered pricing with no hidden charges.',sections:[
    ['Prepaid Balance & Metering','Inference is metered per million tokens based on publicly published rates. Credits are purchased via prepaid top-up packages with zero hidden fees or recurring lock-in.'],
    ['Exact Token Settlement','Requests reserve a conservative estimate before dispatch, and settle authoritatively against exact token consumption reported by the inference engine upon completion. Unused reservations are refunded immediately.'],
    ['Refunds & Dispute Resolution','Prepaid credits are applicable to all models. If you encounter an unserved request or technical outage, our support team reviews logs and credits balances promptly upon notice to support@speedinfer.com.'],
  ]},
  'acceptable-use': {title:'Acceptable Use Policy',intro:'Standards for maintaining platform integrity and safety.',sections:[
    ['Lawful & Ethical Use','The API must not be used for illegal activities, generating malicious software, infringing copyright, or distributing harmful content.'],
    ['Rate Limits & Fair Use','Token bucket rate limiters (RPM and TPM) ensure platform stability and protect all users from noisy-neighbor interference.'],
  ]},
};

export function renderPolicies(key='overview', publicMode=false) {
  const list=Object.entries(policies);
  const navigation=list.map(([id,p])=>publicMode?`<a href="/legal/${id}">${p.title} ↗</a>`:`<button class="ws-text-btn" data-policy="${id}">${p.title} →</button>`).join('');
  const item=policies[key];
  return `<div class="ws-heading"><div><div class="ws-eyebrow">SPEEDINFER / TRUST &amp; COMPLIANCE</div><h1>${item?.title||'Enterprise AI Infrastructure Trust'}</h1><p>${item?.intro||'Understand our zero-retention guarantee, security controls, and terms.'}</p></div></div><div class="ws-notice"><span>✓</span><p>Enterprise Compliance: SpeedInfer operates under a strict Zero Data Retention (ZDR) policy. Prompts and completions are never stored or used for model training.</p></div><div class="ws-policy-layout"><nav class="ws-card ws-policy-nav" aria-label="Policies">${navigation}</nav><article class="ws-card ws-policy-body">${item?item.sections.map(([h,p])=>`<section><h2>${h}</h2><p>${p}</p></section>`).join(''):'<h2>Enterprise AI Infrastructure & Data Trust</h2><p>SpeedInfer is built from the ground up for privacy-first developers and enterprise organizations. Each policy details our security controls, zero-retention commitments, and operational guarantees.</p><h2>Get in touch</h2><p>For enterprise agreements, compliance inquiries, or security reports: <a href="mailto:support@speedinfer.com">support@speedinfer.com</a> or <a href="mailto:sales@speedinfer.com">sales@speedinfer.com</a>.</p>'}</article></div>`;
}

export function showPublicPolicies(targetKey) {
  const landing = document.getElementById('landing-page');
  const auth = document.getElementById('auth-container');
  const app = document.getElementById('app-shell');
  if (landing) landing.style.display='none';
  if (auth) auth.style.display='none';
  if (app) app.style.display='none';
  
  const existing = document.querySelector('.ws-public');
  if (existing) existing.remove();

  const main=document.createElement('main'); 
  main.className='ws-public';
  const pathPart = location.pathname.split('/')[2] || location.pathname.replace(/^\//, '');
  const resolvedKey = targetKey || (policies[pathPart] ? pathPart : (pathPart === 'terms' ? 'terms' : (pathPart === 'privacy' ? 'privacy' : (pathPart === 'security' ? 'security' : 'terms'))));
  main.innerHTML='<a class="ws-public-brand" href="/">SpeedInfer. <span>← Back to platform</span></a>'+renderPolicies(resolvedKey, true);
  document.body.append(main);
}
