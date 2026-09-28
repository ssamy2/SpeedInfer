import { api } from './api.js?v=20260927-inception-release';
import { store } from './store.js?v=20260927-inception-release';
import { renderPolicies } from './policies.js?v=20260927-inception-release';

const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const money = value => `$${Number(value || 0).toFixed(2)}`;
const bytes = value => value < 1024 ? `${value} B` : value < 1048576 ? `${(value / 1024).toFixed(1)} KiB` : `${(value / 1048576).toFixed(1)} MiB`;
const icons = {dashboard:'◫', projects:'▦', buckets:'▱', datasets:'≋', registry:'◈', models:'⬡', playground:'⌘', training:'↗', evaluations:'◎', deployments:'◇', monitoring:'⌁', pricing:'＄', usage:'▥', keys:'⚿', policies:'▤', help:'?', settings:'⚙'};
export const workspaceTitles = {dashboard:'Overview', projects:'Projects', buckets:'Buckets & files', datasets:'Datasets', registry:'Model weights & versions', training:'Training', evaluations:'Evaluations', deployments:'Deployments', monitoring:'Activity & monitoring', pricing:'Pricing', policies:'Trust & policies', settings:'Account settings'};
const groups = [['WORKSPACE',['dashboard','projects']],['BUILD',['buckets','datasets','registry','models','training','evaluations']],['SERVE',['deployments','playground','monitoring']],['MANAGE',['keys','pricing','usage','settings','policies','help']]];
const names = {...workspaceTitles, models:'Model catalog', playground:'Playground', usage:'Credits & billing', keys:'API keys', help:'Help & support'};
const descriptions = {
  projects:'Give every model a home. Organize your training, evaluations and deployments.',
  buckets:'Your files, organized. Upload datasets and model weights into private buckets.',
  datasets:'Prepare your training data independently from your model weights.',
  registry:'Manage uploaded weights, fine-tuned adapters, and model artifacts.',
  training:'Adapt an open model to your data. Configure supervised fine-tuning with LoRA or QLoRA.',
  evaluations:'Evaluate model accuracy, benchmark metrics, and perplexity on test sets.',
  deployments:'Configure a managed model endpoint as part of your inference workflow.',
  monitoring:'A clear record of your model workflows and execution logs.',
  pricing:'Understand token-metered inference and optional managed model endpoint planning.',
  settings:'Manage your account profile, security credentials, and organization preferences.',
};

export class Workspace {
  constructor(navigate, toast) {
    this.navigate = navigate;
    this.toast = toast;
    this.resources = [];
    this.objects = [];
    this.catalog = null;
    this.serial = 0;
    this.route = null;
  }
  init() {
    const footer = document.querySelector('.site-footer > div:nth-child(2)');
    if (footer) {
      const policies = document.createElement('a');
      policies.href = '/legal/overview'; policies.textContent = 'Policies & trust'; footer.append(policies);
    }
    const nav = document.querySelector('.sidebar-nav');
    const inbox = document.getElementById('inbox-nav');
    nav.innerHTML = groups.map(([label, views]) => `<div class="ws-nav-group"><span class="nav-group-label">${label}</span>${views.map(v => `<button class="nav-item" data-view="${v}"><span class="ws-nav-icon" aria-hidden="true">${icons[v]}</span><span>${names[v]}</span></button>`).join('')}</div>`).join('');
    nav.append(inbox);
    const docs = document.createElement('a');
    docs.className = 'nav-item'; docs.href = '/docs'; docs.textContent = '↗ API documentation';
    nav.append(docs);
    document.querySelector('.sidebar-callout').innerHTML = '<span>From model to API.</span><p>Your managed AI workspace.</p><a href="/legal/overview">Trust & transparency ↗</a>';
    document.querySelector('.brand-logo span').innerHTML = 'SpeedInfer<span class="ws-brand-sub">DEVELOPER PLATFORM</span>';
    const panel = document.createElement('section');
    panel.id = 'workspace-panel'; panel.className = 'view-panel ws-panel'; panel.style.display = 'none';
    document.querySelector('.view-viewport').append(panel); this.panel = panel;
    panel.addEventListener('click', event => this.click(event));
    panel.addEventListener('submit', event => this.submit(event));
    panel.addEventListener('change', event => {
      if (event.target.closest('[data-workflow-form]')) {
        if (event.target.name === 'project_id') {
          const select = panel.querySelector('select[name="artifact_id"]');
          if (select) select.innerHTML = this.options(this.items('training').filter(r => r.status === 'succeeded' && r.data.project_id === event.target.value), 'Choose a result from this project');
        }
        this.updateEstimate();
      }
    });
    panel.addEventListener('input', event => {
      if (event.target.matches('[data-file-search]')) this.filterFiles(event.target.value);
      if (event.target.closest('[data-workflow-form]') && (event.target.name === 'epochs' || event.target.name === 'hours' || event.target.name === 'replicas')) {
        this.updateEstimate();
      }
    });
  }
  reset() {
    this.serial++; this.route = null; this.resources = []; this.objects = []; this.catalog = null;
    this.panel?.replaceChildren();
  }
  supports(view) { return Object.hasOwn(workspaceTitles, view.split('/')[0]); }
  async show(route, force = false) {
    if (this.route === route && !force) return;
    this.route = route;
    const serial = ++this.serial;
    this.panel.innerHTML = '<div class="ws-loading" role="status">Loading your workspace…</div>';
    try {
      const [result, catalog] = await Promise.all([api.request('/v1/workspace/resources'), api.request('/v1/workspace/pricing')]);
      if (serial !== this.serial) return;
      this.resources = result.resources; this.objects = result.objects; this.catalog = catalog;
      this.draw();
    } catch (err) {
      if (serial !== this.serial) return;
      this.panel.innerHTML = this.empty('Unable to load workspace', err.message, 'Try again', 'refresh');
    }
  }
  items(kind) { return this.resources.filter(r => r.kind === kind); }
  link(view, text, primary = false) { return `<button class="btn ${primary ? 'btn-primary' : 'btn-secondary'}" data-go="${view}">${text}</button>`; }
  empty(title, description, label, go) { return `<div class="ws-empty"><span class="ws-empty-symbol">◇</span><h3>${esc(title)}</h3><p>${esc(description)}</p>${label ? this.link(go, label, true) : ''}</div>`; }
  heading(title, description, action = '') { return `<div class="ws-heading"><div><div class="ws-eyebrow">SPEEDINFER / WORKSPACE</div><h1>${esc(title)}</h1><p>${esc(description)}</p></div>${action}</div>`; }
  badge(value) { return `<span class="ws-badge ${esc(value)}">${value === 'queued' ? 'Pending' : esc(value)}</span>`; }
  draw() {
    const [view, id] = this.route.split('/');
    let body = '';
    if (view === 'dashboard') body = this.overview();
    else if (view === 'policies') body = renderPolicies(id || 'overview');
    else if (view === 'pricing') body = this.pricing();
    else if (view === 'settings') body = this.settings();
    else if (view === 'monitoring') body = this.monitoring();
    else if (view === 'datasets' || view === 'registry') body = this.fileLibrary(view);
    else if (view === 'buckets') body = this.buckets(id);
    else if (view === 'projects') body = this.projects(id);
    else if (['training','deployments','evaluations'].includes(view)) {
      body = id === 'new' ? this.workflowForm(view) : id ? this.detail(id) : this.workflows(view);
    }
    this.panel.innerHTML = `<div class="ws-content">${body}</div>`;
    if (id === 'new' && view !== 'projects' && view !== 'buckets') this.updateEstimate();
  }
  overview() {
    const first = store.state.user?.name?.split(' ')[0] || 'builder';
    return this.heading(`Welcome, ${first}.`, 'Build, refine and serve your models. One connected workspace.', this.link('projects','New project ↗',true)) +
      `<div class="ws-hero"><div><span class="ws-eyebrow">YOUR MODEL. YOUR WORKFLOW.</span><h2>From first dataset<br>to your next API.</h2><p>Bring your files, configure a training run, and launch a managed model deployment.</p><div class="ws-inline">${this.link('training/new','Launch training →',true)}${this.link('buckets','Upload files')}</div></div><div class="ws-pipeline" aria-label="Model workflow"><div><span>01</span>Private bucket<small>Datasets & weights</small></div><i>↓</i><div><span>02</span>Train & evaluate<small>LoRA & QLoRA</small></div><i>↓</i><div><span>03</span>Managed endpoint<small>Your model, through an API</small></div></div></div>` +
      `<div class="ws-stats">${[['Projects',this.items('project').length,'projects'],['Private buckets',this.items('bucket').length,'buckets'],['Training runs',this.items('training').length,'training'],['API keys',store.state.keys.length,'keys']].map(([title,value,go]) => `<button data-go="${go}" class="ws-stat"><span>${title} ↗</span><strong>${value}</strong><small>View ${title.toLowerCase()}</small></button>`).join('')}</div>` +
      `<div class="ws-section-title"><h2>Start building</h2><span>Three steps, at your pace</span></div><div class="ws-three">${[['01','Organize your files','Create a bucket and upload datasets or model weights.','buckets'],['02','Adapt a model','Select your data and base weights in separate steps.','training/new'],['03','Deploy a model','Configure a managed endpoint and estimate its cost.','deployments/new']].map(([n,t,d,v])=>`<button class="ws-step" data-go="${v}"><span>${n} →</span><h3>${t}</h3><p>${d}</p></button>`).join('')}</div>` +
      `<div class="ws-section-title"><h2>Workspace status</h2></div><div class="ws-card ws-inline ws-between"><div><strong>Model workflow workspace</strong><p>Manage model-serving and workflow configuration from one place. Execution availability depends on connected workers.</p></div>${this.link('policies','Read our policies ↗')}</div>`;
  }
  projects(id) {
    if (id) {
      const project = this.resources.find(r=>r.id===id && r.kind==='project');
      if (!project) return this.empty('Project not found','This project may belong to another workspace.');
      return this.heading(project.name,'Project resources and model workflows.',this.link('projects','← All projects')) + this.resourceTable(this.resources.filter(r=>r.data.project_id===id));
    }
    return this.heading('Projects', descriptions.projects) + this.simpleForm('project','Create a project','e.g. Customer support model') +
      (this.items('project').length ? `<div class="ws-three">${this.items('project').map(r=>`<button class="ws-card ws-project" data-go="projects/${r.id}"><span class="ws-folder">▦</span><h3>${esc(r.name)}</h3><p>${this.resources.filter(x=>x.data.project_id===r.id).length} linked workflows</p><span>Open project →</span></button>`).join('')}</div>` : this.empty('A place for your next model','Create a project to keep related training runs, evaluations and deployments together.'));
  }
  simpleForm(kind, label, placeholder) { return `<form class="ws-card ws-inline ws-create" data-simple="${kind}"><label>${label}<input class="form-input" name="name" maxlength="100" required placeholder="${placeholder}"></label><button class="btn btn-primary" type="submit">Create ${kind} +</button></form>`; }
  buckets(id) {
    if (id) {
      const bucket = this.resources.find(r=>r.id===id && r.kind==='bucket');
      if (!bucket) return this.empty('Bucket not found','Choose an available bucket.');
      const objects = this.objects.filter(o=>o.bucket_id===id);
      return this.heading(bucket.name,'Private bucket · Files are stored in this deployment.',this.link('buckets','← All buckets')) +
        `<div class="ws-notice"><span>▱</span><div><strong>Private Encrypted Storage</strong><p>Store training datasets (.jsonl) and model adapter weights securely in your private workspace.</p></div></div>` +
        `<form class="ws-card ws-upload" data-upload="${id}"><div class="ws-upload-symbol">↑</div><h2>Upload to this bucket</h2><p>Keep datasets and model weights in separate categories.</p><div class="ws-form-grid"><label>File category<select name="purpose" class="form-select"><option value="dataset">Training dataset (.jsonl)</option><option value="weights">Model weights (.safetensors, .gguf, .bin)</option><option value="other">Other file</option></select></label><label>Choose file<input class="form-input" type="file" name="file" required></label></div><button class="btn btn-primary" type="submit">Upload file ↑</button><div role="status" data-upload-status></div></form>` + this.filesTable(objects);
    }
    return this.heading('Buckets & files',descriptions.buckets) + this.simpleForm('bucket','Create a private bucket','e.g. research-assets') +
      `<div class="ws-section-title"><h2>Your buckets</h2><span>${bytes(this.objects.reduce((n,o)=>n+o.size,0))} / 100 MiB used</span></div>` +
      (this.items('bucket').length ? `<div class="ws-three">${this.items('bucket').map(b=>`<button class="ws-card ws-project" data-go="buckets/${b.id}"><span class="ws-folder">▱</span>${this.badge('private')}<h3>${esc(b.name)}</h3><p>${this.objects.filter(o=>o.bucket_id===b.id).length} objects · ${bytes(this.objects.filter(o=>o.bucket_id===b.id).reduce((n,o)=>n+o.size,0))}</p><span>Browse files →</span></button>`).join('')}</div>` : this.empty('Your first bucket starts here','A bucket is a private container for your datasets, model weights and supporting files.'));
  }
  filesTable(objects) {
    if (!objects.length) return this.empty('No files yet','Upload a file to see it here. Datasets and weights remain separate when you configure a workflow.');
    return `<div class="ws-card ws-table-wrap"><div class="ws-table-toolbar"><h2>Files</h2><input class="form-input" data-file-search aria-label="Search files" placeholder="Search files…"></div><table class="ws-table"><thead><tr><th>Name</th><th>Category</th><th>Size</th><th>Actions</th></tr></thead><tbody>${objects.map(o=>`<tr data-file-name="${esc(o.name.toLowerCase())}"><td><strong>${esc(o.name)}</strong><small>${new Date(o.created_at).toLocaleDateString()}</small></td><td>${this.badge(o.purpose)}</td><td>${bytes(o.size)}</td><td><button class="ws-text-btn" data-download="${o.id}">Download</button><button class="ws-text-btn ws-danger" data-delete="${o.id}">Delete</button></td></tr>`).join('')}</tbody></table></div>`;
  }
  filterFiles(value) { this.panel.querySelectorAll('[data-file-name]').forEach(row=>{row.hidden=!row.dataset.fileName.includes(value.toLowerCase());}); }
  fileLibrary(view) {
    const isWeights = view==='registry';
    return this.heading(workspaceTitles[view],descriptions[view],this.link('buckets','Upload files ↑',true)) +
      `<div class="ws-notice"><span>ⓘ</span><p>${isWeights ? 'Uploaded weights and adapters can be selected during managed endpoint planning and private model deployment.' : 'JSONL datasets are validated for structure, valid JSON, and standard prompt/completion schemas.'}</p></div>` +
      this.filesTable(this.objects.filter(o=>o.purpose===(isWeights?'weights':'dataset'))) +
      (isWeights ? '<div class="ws-section-title"><h2>Training run checkpoints &amp; results</h2></div>'+this.resourceTable(this.items('training').filter(r=>r.status==='succeeded')) : '');
  }
  options(items, placeholder, label = 'name') { return `<option value="">${placeholder}</option>`+items.map(o=>`<option value="${esc(o.id)}">${esc(o[label])}</option>`).join(''); }
  workflowForm(view) {
    const kind = {training:'training',deployments:'deployment',evaluations:'evaluation'}[view];
    if (!this.items('project').length) return this.heading(`New ${kind}`, descriptions[view])+this.empty('Start with a project','Create a project to organize this workflow.','Create project','projects');
    const models = store.state.models.map(m=>m.id||m.name);
    if (!models.length) models.push('Qwen/Qwen2.5-7B-Instruct');
    return this.heading(`New ${kind}`,descriptions[view],this.link(view,'← Back'))+
      `<form class="ws-workflow" data-workflow-form="${kind}"><div class="ws-form-main"><section class="ws-card"><h2><span class="ws-number">01</span> Project & identity</h2><div class="ws-form-grid"><label>Name<input name="name" class="form-input" required maxlength="100" placeholder="${kind}-experiment"></label><label>Project<select name="project_id" class="form-select" required>${this.options(this.items('project'),'Choose a project')}</select></label></div></section>`+
      `<section class="ws-card"><h2><span class="ws-number">02</span> Model & weights</h2><label>Base model<select name="model" class="form-select">${models.map(m=>`<option>${esc(m)}</option>`).join('')}</select></label><label>Optional weights from a bucket<select name="weights_id" class="form-select">${this.options(this.objects.filter(o=>o.purpose==='weights'),'Use base model weights')}</select></label><p class="ws-hint">Weights are separate from your dataset. Upload them in Buckets & files first.</p>${kind==='training'?'<label>Training method<select name="method" class="form-select"><option value="qlora">SFT · QLoRA (4-bit)</option><option value="lora">SFT · LoRA</option></select></label>':''}${kind!=='training'?`<label>${kind==='evaluation'?'Training result to evaluate':'Optional training result'}<select name="artifact_id" class="form-select" ${kind==='evaluation'?'required':''}>${this.options(this.items('training').filter(r=>r.status==='succeeded'),'Choose a result from this project')}</select></label>`:''}</section>`+
      (kind==='training'?`<section class="ws-card"><h2><span class="ws-number">03</span> Training data &amp; parameters</h2><label>Dataset from a bucket<select name="dataset_id" class="form-select" required>${this.options(this.objects.filter(o=>o.purpose==='dataset'),'Select a JSONL dataset')}</select></label><p class="ws-hint">Choose your uploaded JSONL dataset. File format and token counts are verified automatically.</p><div class="ws-form-grid" style="margin-top:12px;"><label>Training Epochs<input class="form-input" name="epochs" type="number" min="1" max="50" step="1" value="3" required></label></div>${this.link('buckets','Manage datasets ↗')}</section>`:'')+
      (kind!=='training'?`<section class="ws-card"><h2><span class="ws-number">03</span> Managed endpoint estimate</h2><div class="ws-form-grid"><label>Endpoint tier<select name="sku" class="form-select">${this.catalog.items.map(p=>`<option value="${p.sku}">${p.name} · ${p.memory_gb} GB VRAM · Model endpoint tier</option>`).join('')}</select></label><label>Estimated hours<input class="form-input" name="hours" type="number" min="0.1" max="168" step="0.1" value="1" required></label><label>Replicas<input class="form-input" name="replicas" type="number" min="1" max="4" value="1" required></label></div><p class="ws-hint">Resource tiers for managed private model endpoints and continuous workloads.</p></section>`:'')+
      `</div><aside class="ws-card ws-review"><span class="ws-eyebrow">REVIEW YOUR WORKFLOW</span><h2>Ready when you are.</h2><p>Your configuration is saved privately to your account.</p><div class="ws-estimate" data-estimate aria-live="polite">Calculating…</div><hr><input type="hidden" name="require_balance" value="true"><label class="ws-check"><input type="checkbox" required>I authorize deployment of this workload to the managed compute queue.</label><button class="btn btn-primary" id="btn-submit-workflow" type="submit">Start ${kind==='training'?'Training Run':kind==='deployment'?'Deployment Run':'Evaluation Run'} →</button></aside></form>`;
  }
  async updateEstimate() {
    const form = this.panel.querySelector('[data-workflow-form]');
    if (!form) return;
    const kind = form.dataset.workflowForm;
    const data = new FormData(form); 
    const target = form.querySelector('[data-estimate]');
    const submitBtn = form.querySelector('#btn-submit-workflow') || form.querySelector('button[type="submit"]');
    const sequence = this.estimateSequence = (this.estimateSequence || 0)+1;

    if (kind === 'training') {
      const datasetId = data.get('dataset_id');
      const epochs = Number(data.get('epochs')) || 3;
      if (!datasetId) {
        if (target) target.innerHTML = `<small>Select a dataset to calculate token count and pricing.</small>`;
        return;
      }
      try {
        const modelName = data.get('model') || 'Qwen/Qwen2.5-7B-Instruct';
        const result = await api.request('/v1/workspace/estimate', {
          method: 'POST',
          body: JSON.stringify({
            kind: 'training',
            dataset_id: datasetId,
            model: modelName,
            epochs: epochs,
            sku: 'demo-small'
          })
        });
        if (sequence === this.estimateSequence && target.isConnected) {
          if (!result.is_valid_format) {
            target.innerHTML = `<div class="ws-estimate-error" style="color:var(--danger, #d32f2f); font-size:13px; line-height:1.4;">
              <strong>⚠️ Invalid Dataset Format</strong>
              <p style="margin:4px 0 0 0;">${esc(result.format_error || 'File must be valid UTF-8 JSONL with messages, prompt/completion, or text.')}</p>
            </div>`;
      if (submitBtn) {
              submitBtn.disabled = true;
              submitBtn.textContent = 'Invalid Dataset Format';
            }
            return;
          }

          const hasSufficient = result.has_sufficient_balance;
          const rateFormatted = (Number(result.rate_per_million) || 0.75).toFixed(2);
          target.innerHTML = `
            <div style="margin-bottom:8px;">
              <small>Training cost estimate · ${esc(result.tier || 'Standard Tier')}</small>
              <div style="font-size:26px; font-weight:800; color:var(--primary); margin:2px 0;">${money(result.total_usd)}</div>
              <div style="font-size:12px; color:var(--text-secondary); line-height:1.4;">
                <div>Tokens: <strong>${Number(result.tokens).toLocaleString()}</strong> (${Number(result.tokens * result.epochs).toLocaleString()} total trained)</div>
                <div>Model: <strong>${esc(result.model || modelName)}</strong></div>
                <div>Rate: <strong>$${rateFormatted} / 1M tokens</strong> · Epochs: <strong>${result.epochs}</strong></div>
                <div style="color:var(--text-muted); font-size:11px; margin-top:2px;">${esc(result.formula || `$${rateFormatted} × (${Number(result.tokens).toLocaleString()} / 1M) × ${result.epochs}`)}</div>
              </div>
            </div>
            <div style="padding:8px 10px; border-radius:6px; font-size:12px; margin-top:8px; ${hasSufficient ? 'background:rgba(46,125,50,0.1); border:1px solid rgba(46,125,50,0.3); color:#2e7d32;' : 'background:rgba(211,47,47,0.1); border:1px solid rgba(211,47,47,0.3); color:#d32f2f;'}">
              <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
                <span>Available Balance:</span>
                <strong>${money(result.user_balance)}</strong>
              </div>
              <div>${hasSufficient ? '✓ Sufficient balance available' : '⚠️ Insufficient balance! Please top up your balance.'}</div>
            </div>
          `;

          if (submitBtn) {
            if (!hasSufficient) {
              submitBtn.disabled = true;
              submitBtn.textContent = `Insufficient Balance (${money(result.user_balance)})`;
            } else {
              submitBtn.disabled = false;
              submitBtn.textContent = `Start Training Run (${money(result.total_usd)}) →`;
            }
          }
        }
      } catch (err) {
        if (sequence === this.estimateSequence) target.textContent = err.message || 'Could not calculate training estimate.';
      }
      return;
    }

    try {
      const result = await api.request('/v1/workspace/estimate',{method:'POST',body:JSON.stringify({sku:data.get('sku'),hours:Number(data.get('hours')),replicas:Number(data.get('replicas'))})});
      if (sequence === this.estimateSequence && target.isConnected) {
        target.innerHTML=`<small>Dedicated endpoint estimate</small><strong>${money(result.total_usd)}</strong><small>Dedicated compute · ${result.hours} h runtime × ${result.replicas} replica(s)</small>`;
        if (submitBtn) {
          submitBtn.disabled = false;
          submitBtn.textContent = 'Provision Endpoint →';
        }
      }
    } catch { 
      if (sequence === this.estimateSequence) target.textContent='Enter valid hours (0.1–168) and replicas (1–4).'; 
    }
  }
  workflows(view) {
    const kind={training:'training',deployments:'deployment',evaluations:'evaluation'}[view];
    return this.heading(workspaceTitles[view],descriptions[view],this.link(`${view}/new`,`New ${kind} +`,true))+this.resourceTable(this.items(kind));
  }
  resourceTable(items) {
    if (!items.length) return this.empty('Nothing here yet','Create a workflow to see its status, configuration and history.');
    return `<div class="ws-card ws-table-wrap"><table class="ws-table"><thead><tr><th>Name</th><th>Type</th><th>Status</th><th>Compute Cost</th><th></th></tr></thead><tbody>${items.map(r=>`<tr><td><strong>${esc(r.name)}</strong><small>${new Date(r.created_at).toLocaleDateString()}</small></td><td>${esc(r.kind)}</td><td>${this.badge(r.status)}</td><td>${money(r.data.estimate?.total_usd)}</td><td>${this.link(`${{training:'training',deployment:'deployments',evaluation:'evaluations',bucket:'buckets',project:'projects'}[r.kind]}/${r.id}`,'Open →')}</td></tr>`).join('')}</tbody></table></div>`;
  }
  detail(id) {
    const item=this.resources.find(r=>r.id===id);
    if (!item) return this.empty('Workflow not found','Choose a resource from this workspace.');
    const actions=[];
    if (['queued','stopped'].includes(item.status)) actions.push(['start','Start run']);
    if (item.status==='running') actions.push(item.kind==='deployment'?['stop','Stop endpoint']:['complete','Complete run']);
    if (['queued','running','stopped'].includes(item.status)) actions.push(['cancel','Cancel']);
    const data=item.data;
    return this.heading(item.name,`${item.kind} · ${item.id}`,this.badge(item.status))+
      `<div class="ws-inline ws-action-bar">${actions.map(([action,label])=>`<button class="btn btn-secondary" data-action="${action}" data-id="${id}">${label}</button>`).join('')}</div><div class="ws-two"><section class="ws-card"><h2>Configuration</h2><dl class="ws-dl">${[['Model',data.model],['Method',item.kind==='training'?data.method:'Managed workflow'],['Dataset',this.objects.find(o=>o.id===data.dataset_id)?.name||'Not selected'],['Weights',this.objects.find(o=>o.id===data.weights_id)?.name||'Base model'],['Compute',data.sku],['Compute cost',money(data.estimate?.total_usd)]].map(([k,v])=>`<div><dt>${k}</dt><dd>${esc(v)}</dd></div>`).join('')}</dl></section><section class="ws-card"><h2>Execution log</h2><div class="ws-log">${data.events.map(e=>`<div><time>${new Date(e.at).toLocaleTimeString()}</time><span>${esc(e.message)}</span></div>`).join('')}</div>${data.result?`<p class="ws-notice">${esc(data.result)}</p>`:''}</section></div>`;
  }
  pricing() {
    return this.heading('Transparent pricing',descriptions.pricing)+`<div class="ws-section-title"><h2>Inference API</h2><span>USD / 1 million tokens · from the model catalog</span></div><div class="ws-three">${store.state.models.map(m=>`<article class="ws-card ws-price"><span class="ws-eyebrow">TOKEN USAGE</span><h2>${esc((m.id||m.name).split('/').pop())}</h2><dl class="ws-dl"><div><dt>Input / 1M</dt><dd>${m.prompt_price_per_million==null?'Not listed':money(m.prompt_price_per_million)}</dd></div><div><dt>Output / 1M</dt><dd>${m.completion_price_per_million==null?'Not listed':money(m.completion_price_per_million)}</dd></div></dl><p>Prepaid usage. Real-time token billing with sub-second accuracy.</p>${this.link('usage','Manage credits →')}</article>`).join('')||this.empty('No model prices available','The model catalog has not returned a price.')}</div><div class="ws-section-title"><h2>Dedicated Inference Capacity</h2><span>Dedicated private model endpoint tiers</span></div><div class="ws-three">${this.catalog.items.map(p=>`<article class="ws-card ws-price"><span class="ws-eyebrow">${p.memory_gb} GB VRAM · DEDICATED CAPACITY</span><h2>${p.name}</h2><div class="ws-price-value">${money(p.hourly_usd)}<small>/h dedicated capacity</small></div><p>Dedicated private inference endpoint.<br>Isolated single-tenant throughput.<br>Retention terms are disclosed for the selected deployment.</p>${this.link('deployments/new','Provision endpoint →')}</article>`).join('')}</div><p class="ws-hint">Catalog ${esc(this.catalog.version)}. API credits are one-time top-ups, not monthly subscriptions.</p>`;
  }
  monitoring() {
    const events=this.resources.flatMap(r=>r.data.events.map(e=>({...e,name:r.name}))).sort((a,b)=>b.at.localeCompare(a.at));
    return this.heading(workspaceTitles.monitoring,descriptions.monitoring)+`<div class="ws-notice"><span>⌁</span><p>Connected-worker metrics and telemetry appear during active processing.</p></div><section class="ws-card"><h2>Activity log</h2><div class="ws-log">${events.map(e=>`<div><time>${new Date(e.at).toLocaleTimeString()}</time><span><strong>${esc(e.name)}</strong><br>${esc(e.message)}</span></div>`).join('')||'<p>No activity yet. Start by creating a project or a bucket.</p>'}</div></section>`;
  }
  settings() { return this.heading('Account settings',descriptions.settings)+`<div class="ws-two"><section class="ws-card"><h2>Profile</h2><dl class="ws-dl"><div><dt>Name</dt><dd>${esc(store.state.user?.name||'Not provided')}</dd></div><div><dt>Email</dt><dd>${esc(store.state.user?.email)}</dd></div><div><dt>Location</dt><dd>${esc(store.state.user?.location||'Not provided')}</dd></div><div><dt>Organization</dt><dd>${esc(store.state.user?.organization||'Not provided')}</dd></div><div><dt>Workspace</dt><dd>Personal</dd></div></dl><button class="btn btn-secondary btn-sm" id="ws-edit-profile-btn" style="margin-top:12px;" type="button">Edit Profile &amp; Settings ↗</button></section><section class="ws-card"><h2>Data & account requests</h2><p>Contact support to request account access, corrections, export or closure. Uploaded files can be deleted from a bucket when no workflow references them.</p>${this.link('help','Contact support →')}${this.link('policies/privacy','Privacy policy')}</section></div>`; }
  async submit(event) {
    const form=event.target; event.preventDefault();
    const button=form.querySelector('button[type="submit"]');
    if (!button || button.disabled) return;
    button.disabled=true; const original=button.textContent; button.textContent='Saving…';
    try {
      if (form.dataset.upload) {
        const data=new FormData(form), file=data.get('file');
        if (!file?.size || file.size>this.catalog.max_object_bytes) throw new Error('Choose a non-empty file up to 20 MiB.');
        const response=await fetch(`/v1/workspace/buckets/${form.dataset.upload}/objects?${new URLSearchParams({name:file.name,purpose:data.get('purpose')})}`,{method:'POST',headers:{Authorization:`Bearer ${api.getToken()}`,'Content-Type':'application/octet-stream'},body:file});
        const result=await response.json();
        if (!response.ok) throw new Error(typeof result.detail==='string'?result.detail:'Upload failed');
        this.toast('File uploaded to your private bucket.','success');
        await this.show(this.route,true);
      } else {
        const values=Object.fromEntries(new FormData(form));
        for (const key of ['project_id','dataset_id','weights_id','artifact_id']) if (!values[key]) delete values[key];
        for (const key of ['hours','replicas','epochs']) if (values[key]) values[key]=Number(values[key]);
        if (values.require_balance) values.require_balance = (values.require_balance === 'true' || values.require_balance === true);
        const kind=form.dataset.simple||form.dataset.workflowForm;
        const result=await api.request('/v1/workspace/resources',{method:'POST',body:JSON.stringify({...values,kind,request_id:form.dataset.requestId||(form.dataset.requestId=crypto.randomUUID())})});
        this.toast(`${kind==='bucket'?'Bucket':kind==='project'?'Project':kind==='training'?'Training job queued':kind==='deployment'?'Deployment provisioned':'Workflow initiated'}.`,'success');
      this.navigate(`${{project:'projects',bucket:'buckets',training:'training',deployment:'deployments',evaluation:'evaluations'}[kind]}/${result.id}`);
      }
    } catch(err) { this.toast(err.message,'error'); }
    finally { if (button.isConnected) {button.disabled=false;button.textContent=original;} }
  }
  async click(event) {
    const button=event.target.closest('button');
    if (!button || button.disabled) return;
    if (button.dataset.go) {
      if (button.dataset.go==='refresh') this.show(this.route,true);
      else this.navigate(button.dataset.go);
      return;
    }
    if (button.dataset.policy) { this.navigate(`policies/${button.dataset.policy}`); return; }
    button.disabled=true;
    try {
      if (button.dataset.action) {
        await api.request(`/v1/workspace/resources/${button.dataset.id}/actions`,{method:'POST',body:JSON.stringify({action:button.dataset.action})});
        await this.show(this.route,true);
      } else if (button.dataset.download) {
        const response=await fetch(`/v1/workspace/objects/${button.dataset.download}`,{headers:{Authorization:`Bearer ${api.getToken()}`}});
        if (!response.ok) throw new Error('Could not download this file.');
        const url=URL.createObjectURL(await response.blob()); const anchor=document.createElement('a');
        anchor.href=url; anchor.download=this.objects.find(o=>o.id===button.dataset.download)?.name||'download'; anchor.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
      } else if (button.dataset.delete) {
        if (!window.confirm('Permanently delete this uploaded file? This cannot be undone.')) return;
        await api.request(`/v1/workspace/objects/${button.dataset.delete}`,{method:'DELETE'}); await this.show(this.route,true);
      }
    } catch(err) {this.toast(err.message,'error');}
    finally {if (button.isConnected) button.disabled=false;}
  }
}
