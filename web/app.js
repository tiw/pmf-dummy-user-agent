/* ───────────────────────────────────────────────
   PersonaForge Web UI — Frontend Logic
   ─────────────────────────────────────────────── */

const API_BASE = '';

// ─── DOM refs ───
const modeBtns = document.querySelectorAll('.mode-btn');
const modePanels = document.querySelectorAll('.mode-panel');
const btnGenerateLlm = document.getElementById('btn-generate-llm');
const btnGenerateManual = document.getElementById('btn-generate-manual');
const btnCopyPrompt = document.getElementById('btn-copy-prompt');
const btnDownloadYaml = document.getElementById('btn-download-yaml');
const btnDownloadMd = document.getElementById('btn-download-md');
const results = document.getElementById('results');
const toast = document.getElementById('toast');

let currentResult = null;

// ─── Mode Switcher ───
modeBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    const mode = btn.dataset.mode;
    modeBtns.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    modePanels.forEach(p => p.classList.remove('active'));
    document.getElementById(`mode-${mode}`).classList.add('active');
  });
});

// ─── Accordion ───
document.querySelectorAll('.accordion-header').forEach(header => {
  header.addEventListener('click', () => {
    const body = header.nextElementSibling;
    const isOpen = body.style.display === 'block';
    document.querySelectorAll('.accordion-body').forEach(b => b.style.display = 'none');
    document.querySelectorAll('.accordion-header').forEach(h => h.classList.remove('active'));
    if (!isOpen) {
      body.style.display = 'block';
      header.classList.add('active');
    }
  });
});

// ─── Helpers ───
function showToast(msg, duration = 4000) {
  toast.textContent = msg;
  toast.classList.remove('hidden');
  setTimeout(() => toast.classList.add('hidden'), duration);
}

function showError(msg) {
  // 持久化错误展示在结果区域
  results.classList.remove('hidden');
  document.getElementById('system-prompt').textContent = '';
  document.getElementById('layer-1').innerHTML = '';
  document.getElementById('layer-3').innerHTML = '';
  document.getElementById('layer-4').innerHTML = '';
  document.getElementById('layer-5').innerHTML = '';
  document.getElementById('critique-card').classList.add('hidden');
  document.querySelector('.download-bar').classList.add('hidden');

  const errorHtml = `
    <div style="text-align:center;padding:40px 20px;">
      <div style="font-size:48px;margin-bottom:16px;">⚠️</div>
      <div style="font-size:18px;font-weight:600;color:var(--ink-strong);margin-bottom:12px;">生成失败</div>
      <div style="color:var(--body);max-width:480px;margin:0 auto;line-height:1.6;">${msg}</div>
      <div style="margin-top:24px;padding:16px;background:var(--canvas-soft);border:1px solid var(--hairline);border-radius:var(--radius-md);text-align:left;font-size:13px;color:var(--mute);">
        <strong style="color:var(--ink);">排查步骤：</strong><br>
        1. 确认已设置 DEEPSEEK_API_KEY 环境变量<br>
        2. 确认 API Key 有效且未过期<br>
        3. 检查后端服务是否正常运行
      </div>
    </div>
  `;
  document.querySelector('.card-header').innerHTML = '<h3 class="card-title">错误信息</h3>';
  document.getElementById('system-prompt').innerHTML = errorHtml;
  results.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function setLoading(btn, isLoading, stageText) {
  const text = btn.querySelector('.btn-text');
  const loading = btn.querySelector('.btn-loading');
  btn.disabled = isLoading;
  text.classList.toggle('hidden', isLoading);
  loading.classList.toggle('hidden', !isLoading);
  if (stageText && loading) {
    loading.textContent = stageText;
  }
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function renderKvPairs(obj, keys) {
  return keys.map(k => {
    const val = obj[k];
    let display = val;
    if (Array.isArray(val)) display = val.join(', ') || '—';
    else if (val === undefined || val === null || val === '') display = '—';
    return `<div class="kv"><span class="kv-key">${k}</span><span class="kv-val">${escapeHtml(String(display))}</span></div>`;
  }).join('');
}

// ─── Render Results ───
function renderResult(data) {
  currentResult = data;
  results.classList.remove('hidden');

  // System Prompt
  document.getElementById('system-prompt').textContent = data.system_prompt || '';

  // Critique
  const critiqueCard = document.getElementById('critique-card');
  if (data.critique && Object.keys(data.critique).length > 0) {
    critiqueCard.classList.remove('hidden');
    document.getElementById('critique-score').textContent = data.critique.score ?? '—';
    const body = document.getElementById('critique-body');
    body.innerHTML = '';

    const sections = [
      { key: 'strengths', title: '优点' },
      { key: 'issues', title: '问题' },
      { key: 'suggestions', title: '建议' },
    ];
    sections.forEach(sec => {
      const items = data.critique[sec.key];
      if (Array.isArray(items) && items.length > 0) {
        body.innerHTML += `
          <div class="critique-section">
            <div class="critique-section-title">${sec.title}</div>
            <ul class="critique-list">${items.map(i => `<li>${escapeHtml(i)}</li>`).join('')}</ul>
          </div>`;
      }
    });
  } else {
    critiqueCard.classList.add('hidden');
  }

  // Layer 1: Persona
  const p = data.persona;
  document.getElementById('layer-1').innerHTML = renderKvPairs(p.demographics, ['role', 'age', 'industry', 'company_size', 'location', 'years_experience'])
    + renderKvPairs(p.psychographics, ['goals', 'frustrations', 'decision_style', 'tech_stack', 'budget_authority'])
    + renderKvPairs(p.behavioral_traits, ['communication', 'skepticism_level', 'price_sensitivity', 'risk_tolerance'])
    + renderKvPairs(p.context, ['current_problem', 'recent_changes', 'team_pressure', 'competitive_exposure']);

  // Layer 3: Scene
  document.getElementById('layer-3').innerHTML = renderKvPairs(data.scene, ['scene_description', 'initial_attitude', 'time_pressure', 'participation_motivation', 'prior_exposure']);

  // Layer 4: Memory
  document.getElementById('layer-4').innerHTML = renderKvPairs(data.memory, ['trust_level', 'emotional_state', 'exposure_count']);

  // Layer 5: Behavior
  document.getElementById('layer-5').innerHTML = renderKvPairs(data.behavior, ['attention_keywords', 'skepticism_triggers', 'knowledge_boundary']);

  // Scroll to results
  results.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ─── API Calls ───
async function generateLlm() {
  const productName = document.getElementById('product-name').value.trim();
  const productType = document.getElementById('product-type').value.trim();
  const userDesc = document.getElementById('user-desc').value.trim();

  if (!userDesc) {
    showToast('请输入用户描述');
    return;
  }

  setLoading(btnGenerateLlm, true, '正在扩展 persona 细节...');
  try {
    const res = await fetch(`${API_BASE}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        mode: 'llm',
        product_name: productName || '产品',
        product_type: productType || 'SaaS',
        user_description: userDesc,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || '生成失败');
    renderResult(data);
    showToast('生成成功！');
  } catch (err) {
    showError(err.message);
    console.error(err);
  } finally {
    setLoading(btnGenerateLlm, false);
  }
}

async function generateManual() {
  const payload = {
    mode: 'manual',
    product_name: document.getElementById('m-product-name').value.trim() || '产品',
    product_type: document.getElementById('m-product-type').value.trim() || 'SaaS',
    persona: {
      demographics: {
        role: document.getElementById('m-role').value.trim() || '用户',
        industry: document.getElementById('m-industry').value.trim() || '互联网',
        company_size: document.getElementById('m-company-size').value.trim() || '50-500人',
        location: document.getElementById('m-location').value.trim() || '北京',
        age: parseInt(document.getElementById('m-age').value) || 30,
        years_experience: parseInt(document.getElementById('m-years').value) || 5,
      },
      psychographics: {
        goals: (document.getElementById('m-goals').value || '提高效率').split(',').map(s => s.trim()).filter(Boolean),
        frustrations: (document.getElementById('m-frustrations').value || '工具复杂').split(',').map(s => s.trim()).filter(Boolean),
        decision_style: document.getElementById('m-decision').value.trim() || '数据驱动',
        tech_stack: (document.getElementById('m-tech').value || '').split(',').map(s => s.trim()).filter(Boolean),
        budget_authority: document.getElementById('m-budget').value.trim() || '有建议权',
      },
      behavioral_traits: {
        communication: document.getElementById('m-communication').value.trim() || '直接',
        skepticism_level: parseFloat(document.getElementById('m-skepticism').value) || 0.5,
        price_sensitivity: parseFloat(document.getElementById('m-price').value) || 0.5,
        risk_tolerance: document.getElementById('m-risk').value.trim() || '中',
      },
      context: {
        current_problem: document.getElementById('m-frustrations').value.split(',')[0]?.trim() || '效率低',
        recent_changes: '无重大变化',
        team_pressure: '一般',
        competitive_exposure: [],
      },
    },
    scene: {
      scene_description: document.getElementById('m-scene').value.trim() || '评估产品',
      initial_attitude: document.getElementById('m-attitude').value.trim() || '中立',
      time_pressure: document.getElementById('m-time').value.trim() || '一般',
      participation_motivation: document.getElementById('m-motivation').value.trim() || '解决痛点',
      prior_exposure: document.getElementById('m-exposure').value.trim() || '未接触',
    },
    behavior: {
      attention_keywords: (document.getElementById('m-attention').value || '效率,价格').split(',').map(s => s.trim()).filter(Boolean),
      skepticism_triggers: (document.getElementById('m-triggers').value || '赋能,生态').split(',').map(s => s.trim()).filter(Boolean),
      knowledge_boundary: '知道自己擅长什么，不会假装懂不熟悉的领域',
    },
  };

  setLoading(btnGenerateManual, true, '正在生成并优化...');
  try {
    const res = await fetch(`${API_BASE}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || '生成失败');
    renderResult(data);
    showToast('生成成功！');
  } catch (err) {
    showError(err.message);
    console.error(err);
  } finally {
    setLoading(btnGenerateManual, false);
  }
}

// ─── Downloads ───
function downloadFile(content, filename, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function buildYaml(data) {
  const p = data.persona;
  const yamlStr = (s) => JSON.stringify(s); // simple quoting

  let lines = [
    `# 虚拟用户定义：${p.demographics.role}`,
    `# 框架版本：五层设计框架 v2.0 (LLM增强版)`,
    ``,
    `persona_id: "${p.persona_id}"`,
    ``,
    `demographics:`,
    `  age: ${p.demographics.age}`,
    `  role: ${yamlStr(p.demographics.role)}`,
    `  company_size: ${yamlStr(p.demographics.company_size)}`,
    `  industry: ${yamlStr(p.demographics.industry)}`,
    `  location: ${yamlStr(p.demographics.location)}`,
    `  years_experience: ${p.demographics.years_experience}`,
    ``,
    `psychographics:`,
    `  goals: ${JSON.stringify(p.psychographics.goals)}`,
    `  frustrations: ${JSON.stringify(p.psychographics.frustrations)}`,
    `  decision_style: ${yamlStr(p.psychographics.decision_style)}`,
    `  tech_stack: ${JSON.stringify(p.psychographics.tech_stack)}`,
    `  budget_authority: ${yamlStr(p.psychographics.budget_authority)}`,
    ``,
    `behavioral_traits:`,
    `  communication: ${yamlStr(p.behavioral_traits.communication)}`,
    `  skepticism_level: ${p.behavioral_traits.skepticism_level}`,
    `  price_sensitivity: ${p.behavioral_traits.price_sensitivity}`,
    `  risk_tolerance: ${yamlStr(p.behavioral_traits.risk_tolerance)}`,
    ``,
    `context:`,
    `  current_problem: ${yamlStr(p.context.current_problem)}`,
    `  recent_changes: ${yamlStr(p.context.recent_changes)}`,
    `  team_pressure: ${yamlStr(p.context.team_pressure)}`,
    `  competitive_exposure: ${JSON.stringify(p.context.competitive_exposure)}`,
    ``,
    `system_prompt: |`,
  ];
  data.system_prompt.split('\n').forEach(line => {
    lines.push(`  ${line}`);
  });
  lines.push('');
  lines.push('scene_context:');
  lines.push(`  scene_description: ${yamlStr(data.scene.scene_description)}`);
  lines.push(`  initial_attitude: ${yamlStr(data.scene.initial_attitude)}`);
  lines.push(`  time_pressure: ${yamlStr(data.scene.time_pressure)}`);
  lines.push(`  participation_motivation: ${yamlStr(data.scene.participation_motivation)}`);
  lines.push(`  prior_exposure: ${yamlStr(data.scene.prior_exposure)}`);
  lines.push('');
  lines.push('memory_state:');
  lines.push(`  trust_level: ${data.memory.trust_level}`);
  lines.push(`  emotional_state: ${yamlStr(data.memory.emotional_state)}`);
  lines.push(`  exposure_count: ${data.memory.exposure_count}`);
  lines.push('');
  lines.push('behavior_engine:');
  lines.push(`  attention_keywords: ${JSON.stringify(data.behavior.attention_keywords)}`);
  lines.push(`  skepticism_triggers: ${JSON.stringify(data.behavior.skepticism_triggers)}`);
  lines.push(`  knowledge_boundary: ${yamlStr(data.behavior.knowledge_boundary)}`);

  if (data.critique) {
    lines.push('');
    lines.push('llm_critique:');
    lines.push(`  score: ${data.critique.score ?? 'N/A'}`);
    lines.push(`  strengths: ${JSON.stringify(data.critique.strengths || [])}`);
    lines.push(`  issues: ${JSON.stringify(data.critique.issues || [])}`);
    lines.push(`  suggestions: ${JSON.stringify(data.critique.suggestions || [])}`);
  }

  return lines.join('\n');
}

// ─── Event Listeners ───
btnGenerateLlm.addEventListener('click', generateLlm);
btnGenerateManual.addEventListener('click', generateManual);

btnCopyPrompt.addEventListener('click', () => {
  if (!currentResult) return;
  navigator.clipboard.writeText(currentResult.system_prompt).then(() => {
    showToast('System Prompt 已复制');
  });
});

btnDownloadYaml.addEventListener('click', () => {
  if (!currentResult) return;
  const yaml = buildYaml(currentResult);
  const name = currentResult.persona.persona_id || 'persona';
  downloadFile(yaml, `${name}.yaml`, 'text/yaml');
  showToast('YAML 已下载');
});

btnDownloadMd.addEventListener('click', () => {
  if (!currentResult) return;
  const md = `# 虚拟用户：${currentResult.persona.demographics.role}\n\n## System Prompt\n\n\`\`\`markdown\n${currentResult.system_prompt}\n\`\`\`\n`;
  const name = currentResult.persona.persona_id || 'persona';
  downloadFile(md, `${name}.md`, 'text/markdown');
  showToast('Markdown 已下载');
});
