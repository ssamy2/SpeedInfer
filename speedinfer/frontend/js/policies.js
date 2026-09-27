const policies = {
  terms: {title:'Terms of service', intro:'Rules for using the SpeedInfer preview.', sections:[
    ['Service scope','SpeedInfer provides an inference gateway and a preview workspace for managed model workflows. The availability of live inference depends on the deployment and its connected workers. Training, evaluation and deployment workflows marked Simulation do not allocate hardware or produce real model results.'],
    ['Your account','Provide accurate account information, protect your credentials and use only data and models you are authorized to use. API keys grant access to billable inference where configured. Revoke a key if it is exposed. You remain responsible for activity authorized by your credentials.'],
    ['Files and models','You retain your rights in uploaded content. Upload only material you own or have permission to process. Licenses for base models and datasets continue to apply. Storing an uploaded weights file does not verify its compatibility, licensing or safety.'],
    ['Preview limits','Preview features may change and are not offered with an uptime or durability guarantee. Keep an independent copy of important files. The local storage preview accepts up to 20 MiB per file, 100 MiB and 100 files per account. Large production weight files require a future storage integration.'],
    ['Charges and suspension','API credit purchases and metered inference are separate from free simulations. See the billing policy before making a purchase. Access may be limited to address abuse, security incidents or resource limits. Contact support to review an access issue.'],
    ['Commercial terms','SpeedInfer AI Technologies (headquartered in Cairo, Egypt) provides high-performance inference acceleration software and developer infrastructure. Enterprise agreements, custom SLAs, liability provisions and dedicated private capacity contracts are confirmed via enterprise sales. Questions: Support@speedinfer.com.'],
  ]},
  privacy: {title:'Privacy & data handling',intro:'What this deployment stores and how it is used.',sections:[
    ['Account information','The service stores your email, optional name, password hash, account status and timestamps to provide account access. API key hashes and metadata support authentication; the raw key is revealed when explicitly created. The current browser session uses local storage for its access token and profile.'],
    ['Workspace content','Projects, bucket metadata, uploaded file contents, workflow settings and activity events are stored in this deployment’s database and associated with your account. They are used to provide the workspace features you request. Application access is scoped to the owning account; system operators with database access can administer stored information.'],
    ['Inference and payments','SpeedInfer operates under a strict enterprise Zero Data Retention (ZDR) guarantee: all prompts and completions are processed ephemerally in GPU memory (VRAM) and are never logged, stored on persistent media, or used for model training. Where payment is enabled, Whop hosts checkout and sends payment events used to credit the selected API key.'],
    ['Support requests','Contact requests store the details you provide, consent, timestamps and an abuse-prevention connection fingerprint. Authorized administrators can read requests. If SMTP is configured, the deployment sends a notification through its configured mail provider.'],
    ['Retention and requests','There is no automated account-wide retention schedule in this preview. Uploaded files can be deleted when not referenced by a workflow; other records remain until removed by the operator. Request access, correction, export or account closure through Support@speedinfer.com. No automated completion deadline is promised.'],
    ['Providers and locations','Infrastructure, backup location and inference processors depend on the operator’s deployment. Contact support for the applicable provider list and location before uploading sensitive or regulated information. No certification or cross-border compliance claim is made by this preview.'],
  ]},
  billing: {title:'Billing & refunds',intro:'Know what is charged and what is only an estimate.',sections:[
    ['API credits','Published credit packages are one-time prepaid top-ups, not recurring subscriptions. The server supplies package amounts and validates checkout. Credits currently belong to the API key selected for the purchase. An account with no keys has no spendable key balance.'],
    ['Metered inference','Input and output token rates are shown separately per million tokens in the model catalog. Inference charges depend on the model and token usage. The amount displayed in a compute simulation is not an inference price or a payment request.'],
    ['Simulations','Dedicated endpoint and training capacity estimates are illustrative values in a server-side demo catalog. Estimated cost reflects dedicated endpoint runtime across replicas. Actual charges are zero in simulation mode. Storage and network costs are not included or priced. No compute resource is reserved.'],
    ['Payment confirmation','A checkout redirect alone does not establish that a balance has been credited. The backend verifies payment notifications and applies supported successful payments. If the amount is missing or incorrect, contact support with the transaction reference; never send card details or API secrets.'],
    ['Refund requests','Contact Support@speedinfer.com with the transaction reference, amount and reason. Requests require operator review. This preview does not publish a universal refund window or promise refunds for consumed usage. Binding commercial refund and tax terms must be confirmed before a production purchase.'],
  ]},
  'acceptable-use': {title:'Acceptable use',intro:'Use shared services responsibly.',sections:[
    ['Authorized workloads','Use only accounts, datasets, models and systems that you have permission to access. Respect licenses and intellectual property rights. Do not use the service to distribute illegal content or facilitate harm.'],
    ['Service integrity','Do not bypass authentication, attempt to access another account’s files, evade quotas, upload malicious payloads for execution, attack the service or interfere with other users. API rate limits and storage limits apply.'],
    ['Sensitive information','Do not upload production secrets or regulated personal information into the preview. Remove passwords, private keys and unrelated personal data from datasets and support requests.'],
    ['Enforcement and reports','Suspected abuse can result in access restrictions and investigation. Report incidents to Support@speedinfer.com with relevant timestamps and non-sensitive references. The operator reviews disputed restrictions.'],
  ]},
  retention: {title:'Storage & retention',intro:'Private buckets for a bounded local preview.',sections:[
    ['Where files live','Uploads are stored as database objects in this deployment, not in Google Cloud Storage or a connected S3 service. Bucket names are organizational labels; no public URL is created. Authenticated download is required.'],
    ['File categories','Datasets and weights are categorized independently. Dataset uploads accept UTF-8 JSONL with text or messages fields. Weights accept safetensors, GGUF or bin filenames and are never executed by the upload service. Extension checks are not a security scan or model validation.'],
    ['Deletion','Unreferenced files can be permanently deleted through the bucket interface. Deletion is blocked while a saved workflow references the file to preserve its record. Ask support to remove dependent records or close the account.'],
    ['Backups and lifecycle','No automatic expiration, version recovery or backup retention commitment is implemented by the preview. Database file cleanup and backups are operator responsibilities. Keep your source files independently and request deployment-specific retention details if needed.'],
  ]},
  security: {title:'Security & service status',intro:'Describe the controls that exist, without implying certifications.',sections:[
    ['Account boundaries','Workspace API requests require an authenticated account and enforce ownership checks for referenced projects, files and jobs. Raw API keys are shown only after creation; stored key hashes are used for verification.'],
    ['Operational limits','Live inference availability is separate from simulated workflow states. A Running simulation is not evidence of a GPU worker or live endpoint. The preview has no commercial uptime SLA or guaranteed support response time.'],
    ['Report a vulnerability','Send a concise report to Support@speedinfer.com. Include reproduction steps using your own account and avoid exposing another user’s data. Do not include passwords or raw API keys.'],
    ['Production readiness','A production rollout requires verified infrastructure security, backup and recovery, provider integration and operational monitoring. This preview does not claim SOC 2, ISO 27001, NVIDIA membership or an NVIDIA endorsement.'],
  ]},
  cookies: {title:'Browser storage',intro:'Storage used for account access and interface state.',sections:[
    ['Essential local storage','The current client stores the access token and profile in browser local storage to maintain login. Signing out removes these values and clears the visible workspace. Anyone with access to an unlocked browser session may be able to use that session.'],
    ['API secrets','API keys are not automatically generated on registration and are not persisted by the updated client. A Playground key you enter is kept in the page for the current session and cleared on logout. Copy a newly created key to your own secure storage.'],
    ['Optional services','The workspace does not require an advertising consent choice. Any future optional analytics or marketing tools need an updated disclosure reflecting what is actually deployed. External checkout and linked sites use their own policies.'],
  ]},
};

export function renderPolicies(key='overview', publicMode=false) {
  const list=Object.entries(policies);
  const navigation=list.map(([id,p])=>publicMode?`<a href="/legal/${id}">${p.title} ↗</a>`:`<button class="ws-text-btn" data-policy="${id}">${p.title} →</button>`).join('');
  const item=policies[key];
  return `<div class="ws-heading"><div><div class="ws-eyebrow">SPEEDINFER / TRUST CENTER</div><h1>${item?.title||'Built on transparency.'}</h1><p>${item?.intro||'Understand your files, your costs and the limits of this preview.'}</p></div></div><div class="ws-notice"><span>ⓘ</span><p>Preview policies · Version 2026-09-27. Operational disclosures for this release. Company-specific commercial terms require completion before production contracting.</p></div><div class="ws-policy-layout"><nav class="ws-card ws-policy-nav" aria-label="Policies">${navigation}</nav><article class="ws-card ws-policy-body">${item?item.sections.map(([h,p])=>`<section><h2>${h}</h2><p>${p}</p></section>`).join(''):'<h2>Your model workflow, clearly explained.</h2><p>Private uploads are real. Training, evaluation and deployment workflows are simulations. API inference depends on the connected backend. Each policy explains these boundaries and the data involved.</p><h2>Get in touch</h2><p>For data access, billing questions or security reports, contact <a href="mailto:Support@speedinfer.com">Support@speedinfer.com</a>.</p>'}</article></div>`;
}

export function showPublicPolicies() {
  document.getElementById('landing-page').style.display='none';
  document.getElementById('auth-container').style.display='none';
  document.getElementById('app-shell').style.display='none';
  const main=document.createElement('main'); main.className='ws-public';
  main.innerHTML='<a class="ws-public-brand" href="/">SpeedInfer. <span>← Back to platform</span></a>'+renderPolicies(location.pathname.split('/')[2]||'overview',true);
  document.body.append(main);
}
