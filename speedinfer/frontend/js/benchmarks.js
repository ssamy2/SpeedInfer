(() => {
  const models = [
    {
      id: 'gpt-5-6-sol',
      name: 'GPT-5.6 Sol',
      maker: 'OpenAI',
      logo: 'openai.svg',
      tone: 'openai',
      access: 'External API',
      cohort: false,
      params: 'Undisclosed',
      context: '1.05M',
      type: 'Frontier · multimodal',
      best: 'Complex reasoning & code',
      sourceUrl: 'https://platform.openai.com/docs/models',
      scores: {
        reasoning: [89.2, 94.6], // MMLU-Pro, GPQA Diamond
        code: [82.6],            // LiveCodeBench
        math: [98.4, 96.8],      // AIME, MATH-500
        instruction: [81.7]      // IF-Eval
      }
    },
    {
      id: 'gpt-5-6-terra',
      name: 'GPT-5.6 Terra',
      maker: 'OpenAI',
      logo: 'openai.svg',
      tone: 'openai',
      access: 'External API',
      cohort: false,
      params: 'Undisclosed',
      context: '1.05M',
      type: 'Multimodal · reasoning',
      best: 'Balanced workloads',
      sourceUrl: 'https://platform.openai.com/docs/models',
      scores: {
        reasoning: [82.5, 84.2],
        code: [77.0],
        math: [76.0, 92.4],
        instruction: [84.6]
      }
    },
    {
      id: 'gpt-5-6-luna',
      name: 'GPT-5.6 Luna',
      maker: 'OpenAI',
      logo: 'openai.svg',
      tone: 'openai',
      access: 'External API',
      cohort: false,
      params: 'Undisclosed',
      context: '1.05M',
      type: 'Multimodal · efficient',
      best: 'High-volume inference',
      sourceUrl: 'https://platform.openai.com/docs/models',
      scores: {
        reasoning: [74.2, 68.4],
        code: [58.2],
        math: [52.0, 84.5],
        instruction: [82.0]
      }
    },
    {
      id: 'claude-fable-5',
      name: 'Claude Fable 5',
      maker: 'Anthropic',
      logo: 'anthropic.svg',
      tone: 'anthropic',
      access: 'External API',
      cohort: false,
      params: 'Undisclosed',
      context: '1M',
      type: 'Adaptive reasoning · vision',
      best: 'Frontier intelligence',
      sourceUrl: 'https://docs.anthropic.com/en/docs/about-claude/models',
      scores: {
        reasoning: [88.5, 89.4],
        code: [80.8],
        math: [86.7, 95.2],
        instruction: [91.2]
      }
    },
    {
      id: 'claude-opus-5',
      name: 'Claude Opus 5',
      maker: 'Anthropic',
      logo: 'anthropic.svg',
      tone: 'anthropic',
      access: 'External API',
      cohort: false,
      params: 'Undisclosed',
      context: '1M',
      type: 'Adaptive reasoning · vision',
      best: 'Deep agents & coding',
      sourceUrl: 'https://docs.anthropic.com/en/docs/about-claude/models',
      scores: {
        reasoning: [90.4, 92.1],
        code: [84.5],
        math: [91.3, 96.0],
        instruction: [92.8]
      }
    },
    {
      id: 'claude-sonnet-5',
      name: 'Claude Sonnet 5',
      maker: 'Anthropic',
      logo: 'anthropic.svg',
      tone: 'anthropic',
      access: 'External API',
      cohort: false,
      params: 'Undisclosed',
      context: '1M',
      type: 'Adaptive reasoning · vision',
      best: 'Speed & intelligence',
      sourceUrl: 'https://docs.anthropic.com/en/docs/about-claude/models',
      scores: {
        reasoning: [84.1, 82.6],
        code: [76.4],
        math: [74.0, 91.5],
        instruction: [90.6]
      }
    },
    {
      id: 'claude-haiku-4-5',
      name: 'Claude Haiku 4.5',
      maker: 'Anthropic',
      logo: 'anthropic.svg',
      tone: 'anthropic',
      access: 'External API',
      cohort: false,
      params: 'Undisclosed',
      context: '200K',
      type: 'Fast · multimodal',
      best: 'Low-latency workloads',
      sourceUrl: 'https://docs.anthropic.com/en/docs/about-claude/models',
      scores: {
        reasoning: [74.6, 71.8],
        code: [54.2],
        math: [56.7, 82.4],
        instruction: [88.9]
      }
    },
    {
      id: 'deepseek',
      name: 'DeepSeek V3',
      maker: 'DeepSeek AI',
      logo: 'deepseek-icon.svg',
      tone: 'mint',
      cohort: true,
      params: '671B / 37B active',
      context: '128K',
      type: 'MoE',
      best: 'Math & code',
      sourceUrl: 'https://github.com/deepseek-ai/DeepSeek-V3#4-evaluation-results',
      scores: {
        reasoning: [75.9, 59.1],
        code: [37.6],
        math: [39.2, 90.2],
        instruction: [86.1]
      }
    },
    {
      id: 'deepseek-r1',
      name: 'DeepSeek R1',
      maker: 'DeepSeek AI',
      logo: 'deepseek-icon.svg',
      tone: 'mint',
      cohort: false,
      params: '671B / 37B active',
      context: '128K',
      type: 'MoE · reasoning',
      best: 'Deep reasoning',
      sourceUrl: 'https://github.com/deepseek-ai/DeepSeek-R1#4-evaluation-results',
      scores: {
        reasoning: [84.0, 71.5],
        code: [65.9],
        math: [79.8, 97.3],
        instruction: [83.3]
      }
    },
    {
      id: 'qwen3-235',
      name: 'Qwen3 235B A22B',
      maker: 'Qwen',
      logo: 'qwen.png',
      tone: 'blue',
      cohort: false,
      params: '235B / 22B active',
      context: '256K',
      type: 'MoE · hybrid thinking',
      best: 'Reasoning at scale',
      sourceUrl: 'https://github.com/QwenLM/Qwen3',
      scores: {
        reasoning: [82.4, 84.0],
        code: [70.7],
        math: [81.5, 94.8],
        instruction: [88.2]
      }
    },
    {
      id: 'qwen3-30',
      name: 'Qwen3 30B A3B',
      maker: 'Qwen',
      logo: 'qwen.png',
      tone: 'blue',
      cohort: false,
      params: '30B / 3B active',
      context: '256K',
      type: 'MoE · hybrid thinking',
      best: 'Efficient agents',
      sourceUrl: 'https://github.com/QwenLM/Qwen3',
      scores: {
        reasoning: [74.8, 73.4],
        code: [66.0],
        math: [85.0, 91.2],
        instruction: [85.4]
      }
    },
    {
      id: 'qwen',
      name: 'Qwen2.5 72B',
      maker: 'Alibaba Cloud',
      logo: 'qwen.png',
      tone: 'blue',
      cohort: true,
      params: '72B',
      context: '128K',
      type: 'Dense',
      best: 'Multilingual',
      sourceUrl: 'https://github.com/QwenLM/Qwen2.5',
      scores: {
        reasoning: [71.6, 49.0],
        code: [28.7],
        math: [23.3, 80.0],
        instruction: [84.1]
      }
    },
    {
      id: 'llama4-maverick',
      name: 'Llama 4 Maverick',
      maker: 'Meta',
      logo: 'meta.svg',
      tone: 'violet',
      cohort: false,
      params: '400B / 17B active',
      context: '1M',
      type: 'MoE · multimodal',
      best: 'Image & text',
      sourceUrl: 'https://ai.meta.com/llama/',
      scores: {
        reasoning: [78.4, 64.2],
        code: [56.8],
        math: [48.0, 88.6],
        instruction: [89.5]
      }
    },
    {
      id: 'llama4-scout',
      name: 'Llama 4 Scout',
      maker: 'Meta',
      logo: 'meta.svg',
      tone: 'violet',
      cohort: false,
      params: '109B / 17B active',
      context: '10M',
      type: 'MoE · multimodal',
      best: 'Long context',
      sourceUrl: 'https://ai.meta.com/llama/',
      scores: {
        reasoning: [72.1, 53.8],
        code: [46.2],
        math: [34.0, 81.2],
        instruction: [87.8]
      }
    },
    {
      id: 'llama',
      name: 'Llama 3.1 405B',
      maker: 'Meta',
      logo: 'meta.svg',
      tone: 'violet',
      cohort: true,
      params: '405B',
      context: '128K',
      type: 'Dense',
      best: 'General knowledge',
      sourceUrl: 'https://github.com/meta-llama/llama-models',
      scores: {
        reasoning: [73.3, 51.1],
        code: [30.1],
        math: [23.3, 73.8],
        instruction: [86.0]
      }
    },
    {
      id: 'gemma',
      name: 'Gemma 3 27B',
      maker: 'Google DeepMind',
      logo: 'gemma.svg',
      tone: 'coral',
      cohort: false,
      params: '27B',
      context: '128K',
      type: 'Dense · multimodal',
      best: 'Efficient deployment',
      sourceUrl: 'https://ai.google.dev/gemma/docs/core/model_card_3',
      scores: {
        reasoning: [67.5, 42.4],
        code: [29.7],
        math: [26.7, 89.0],
        instruction: [90.4]
      }
    },
    {
      id: 'mistral-small',
      name: 'Mistral Small 3.1',
      maker: 'Mistral AI',
      logo: 'mistral.svg',
      tone: 'coral',
      cohort: false,
      params: '24B',
      context: '128K',
      type: 'Dense · multimodal',
      best: 'Small multimodal',
      sourceUrl: 'https://mistral.ai/news/mistral-small-3-1/',
      scores: {
        reasoning: [58.2, 41.1],
        code: [34.6],
        math: [20.0, 70.6],
        instruction: [82.9]
      }
    },
    {
      id: 'nemotron-ultra',
      name: 'Nemotron 3 Ultra',
      maker: 'NVIDIA',
      logo: 'nvidia.svg',
      tone: 'mint',
      cohort: false,
      params: '550B / 55B active',
      context: '1M',
      type: 'Hybrid MoE',
      best: 'Agentic reasoning',
      sourceUrl: 'https://github.com/NVIDIA-NeMo/Nemotron',
      scores: {
        reasoning: [86.8, 87.0],
        code: [89.0],
        math: [78.5, 93.4],
        instruction: [85.0]
      }
    },
    {
      id: 'llama-nemotron',
      name: 'Llama Nemotron Ultra',
      maker: 'NVIDIA',
      logo: 'nvidia.svg',
      tone: 'mint',
      cohort: false,
      params: '253B',
      context: '128K',
      type: 'Dense · reasoning',
      best: 'Scientific reasoning',
      sourceUrl: 'https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Instruct',
      scores: {
        reasoning: [54.2, 76.0],
        code: [38.0],
        math: [28.0, 68.5],
        instruction: [89.2]
      }
    },
    {
      id: 'phi4',
      name: 'Phi-4',
      maker: 'Microsoft',
      logo: 'microsoft.svg',
      tone: 'coral',
      cohort: false,
      params: '14B',
      context: '16K',
      type: 'Dense',
      best: 'Compact reasoning',
      sourceUrl: 'https://huggingface.co/microsoft/phi-4',
      scores: {
        reasoning: [70.4, 56.1],
        code: [43.8],
        math: [24.0, 80.4],
        instruction: [78.5]
      }
    }
  ];

  const body = document.getElementById('benchmark-body');
  const profiles = document.getElementById('benchmark-profiles');
  const page = document.getElementById('benchmarks-page');
  if (!body || !page) return;

  let category = 'all';
  let sortMetricIdx = null;
  let sortAscending = false;

  const flat = model => [
    ...model.scores.reasoning,
    ...model.scores.code,
    ...model.scores.math,
    ...model.scores.instruction
  ];

  const cell = (value, best, comparable) => {
    if (value == null) return '<td class="score-na">—</td>';
    const isBest = comparable && value === best;
    return `<td><b class="score ${isBest ? 'is-best' : ''}">${value.toFixed(1)}</b></td>`;
  };

  const bests = [0, 1, 2, 3, 4, 5].map(i =>
    Math.max(
      ...models
        .filter(m => m.cohort)
        .map(m => flat(m)[i])
        .filter(Number.isFinite)
    )
  );

  function updateSpotlight(m) {
    const spotlight = document.querySelector('.benchmark-spotlight');
    if (!spotlight || !m) return;
    const initial = m.name.charAt(0);
    spotlight.innerHTML = `
      <span class="spotlight-rank">${m.cohort ? '01 / PICK' : 'PROFILE SPOTLIGHT'}</span>
      <div class="spotlight-mark">${initial}</div>
      <h3>${m.name}</h3>
      <p>${m.maker} · ${m.type}. Optimized for ${m.best.toLowerCase()}.</p>
      <dl>
        <div><dt>Architecture</dt><dd>${m.type}</dd></div>
        <div><dt>Parameters</dt><dd>${m.params}</dd></div>
        <div><dt>Context</dt><dd>${m.context}</dd></div>
        <div><dt>Best at</dt><dd>${m.best}</dd></div>
      </dl>
      <button class="btn lime-button" data-auth="register">Run with SpeedInfer <span>↗</span></button>
    `;
    spotlight.querySelectorAll('[data-auth]').forEach(btn => {
      btn.addEventListener('click', () => { page.hidden = true; });
    });
  }

  function renderTable() {
    const query = (document.getElementById('benchmark-search')?.value || '').trim().toLowerCase();
    let visible = models.filter(m => `${m.name} ${m.maker}`.toLowerCase().includes(query));

    if (sortMetricIdx !== null) {
      visible = [...visible].sort((a, b) => {
        if (sortMetricIdx === -1) {
          const comp = a.name.localeCompare(b.name);
          return sortAscending ? comp : -comp;
        }
        const valA = flat(a)[sortMetricIdx] ?? -Infinity;
        const valB = flat(b)[sortMetricIdx] ?? -Infinity;
        return sortAscending ? valA - valB : valB - valA;
      });
    }

    body.innerHTML = visible.map(m => {
      const scores = flat(m);
      const access = m.access || 'Hosted model';
      return `
        <tr data-model="${m.id}" style="cursor: pointer;">
          <td>
            <span class="table-model-mark ${m.tone}">
              <img src="/static/assets/models/${m.logo}" alt="${m.name} logo">
            </span>
            <span>
              <strong>${m.name}</strong>
              <small>${m.maker} · ${access}${m.cohort ? '' : ' · separate protocol'}</small>
            </span>
          </td>
          ${scores.map((v, i) => cell(v, bests[i], m.cohort)).join('')}
        </tr>
      `;
    }).join('');

    const empty = document.getElementById('benchmark-empty');
    if (empty) empty.hidden = visible.length > 0;

    body.querySelectorAll('tr[data-model]').forEach(row => {
      row.addEventListener('click', () => {
        const modelId = row.dataset.model;
        const found = models.find(m => m.id === modelId);
        if (found) {
          body.querySelectorAll('tr').forEach(r => r.classList.remove('is-selected'));
          row.classList.add('is-selected');
          updateSpotlight(found);
        }
      });
    });

    document.querySelectorAll('.benchmark-table th[data-metric], .benchmark-table td').forEach(el => el.classList.remove('metric-muted'));
    if (category !== 'all') {
      document.querySelectorAll('.benchmark-table th[data-metric]').forEach((th, index) => {
        if (th.dataset.metric !== category) {
          th.classList.add('metric-muted');
          body.querySelectorAll('tr').forEach(row => row.children[index + 1]?.classList.add('metric-muted'));
        }
      });
    }
  }

  profiles.innerHTML = models.map((m, i) => `
    <article class="profile-card ${m.tone}">
      <div class="profile-top">
        <span>${String(i + 1).padStart(2, '0')}</span>
        <i><img src="/static/assets/models/${m.logo}" alt="${m.name} logo"></i>
      </div>
      <h3>${m.name}</h3>
      <p>${m.maker} · ${m.access || 'Hosted model'}</p>
      <dl>
        <div><dt>Model</dt><dd>${m.type}</dd></div>
        <div><dt>Parameters</dt><dd>${m.params}</dd></div>
        <div><dt>Context</dt><dd>${m.context}</dd></div>
        <div><dt>Profile</dt><dd>${m.best}</dd></div>
      </dl>
      <a href="${m.sourceUrl}" target="_blank" rel="noopener" class="profile-source" style="text-decoration:none;">
        ${m.cohort ? 'COMPARABLE SET' : 'OFFICIAL PRIMARY SOURCE'} ↗
      </a>
    </article>
  `).join('');

  document.querySelectorAll('[data-benchmark]').forEach(button => {
    button.addEventListener('click', () => {
      category = button.dataset.benchmark;
      document.querySelectorAll('[data-benchmark]').forEach(b => b.classList.toggle('active', b === button));
      renderTable();
    });
  });

  const searchInput = document.getElementById('benchmark-search');
  if (searchInput) searchInput.addEventListener('input', renderTable);

  const headers = document.querySelectorAll('.benchmark-table thead th');
  headers.forEach((th, index) => {
    th.style.cursor = 'pointer';
    th.title = 'Click to sort';
    th.addEventListener('click', () => {
      const metricIndex = index === 0 ? -1 : index - 1;
      if (sortMetricIdx === metricIndex) {
        sortAscending = !sortAscending;
      } else {
        sortMetricIdx = metricIndex;
        sortAscending = false;
      }
      renderTable();
    });
  });

  function route() {
    const show = location.hash === '#benchmarks' || location.hash.startsWith('#benchmark-');
    if (show) {
      document.getElementById('landing-page').style.display = 'none';
      document.getElementById('auth-container').style.display = 'none';
      document.getElementById('app-shell').style.display = 'none';
      page.hidden = false;
      if (location.hash === '#benchmarks') window.scrollTo(0, 0);
    } else {
      const wasOpen = !page.hidden;
      page.hidden = true;
      if (
        wasOpen &&
        document.getElementById('auth-container').style.display === 'none' &&
        document.getElementById('app-shell').style.display === 'none'
      ) {
        document.getElementById('landing-page').style.display = 'block';
        window.scrollTo(0, 0);
      }
    }
  }

  page.querySelectorAll('[data-auth]').forEach(button => {
    button.addEventListener('click', () => { page.hidden = true; });
  });

  window.addEventListener('hashchange', route);
  document.addEventListener('DOMContentLoaded', () => setTimeout(route, 0));
  renderTable();
})();
