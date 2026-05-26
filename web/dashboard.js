/* ───────────────────────────────────────────────
   VMU Dashboard — Frontend Logic
   ─────────────────────────────────────────────── */

const API = '/api/v1';

// ═══ State ═══
let state = {
  types: [],
  instances: [],
  scenes: [],
  currentPage: 'dashboard',
  chatInstanceId: null,
  chatHistory: [],
};

// ═══ DOM refs ═══
const $ = id => document.getElementById(id);
const $$ = sel => document.querySelectorAll(sel);

// ═══ API ═══
async function apiGet(path) {
  const r = await fetch(API + path);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
async function apiPost(path, body) {
  const r = await fetch(API + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
async function apiDelete(path) {
  const r = await fetch(API + path, { method: 'DELETE' });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

// ═══ Toast ═══
function toast(msg, type = 'default') {
  const el = $('toast');
  el.textContent = msg;
  el.className = 'toast show ' + type;
  setTimeout(() => el.classList.remove('show'), 3000);
}

// ═══ Navigation ═══
$$('.nav-item').forEach(el => {
  el.addEventListener('click', () => {
    const page = el.dataset.page;
    switchPage(page);
  });
});

function switchPage(page) {
  state.currentPage = page;
  $$('.nav-item').forEach(n => n.classList.toggle('active', n.dataset.page === page));
  $$('.page').forEach(p => p.classList.toggle('active', p.id === 'page-' + page));
  $('page-title').textContent = {
    dashboard: '仪表盘',
    types: '人格类型',
    scenes: '场景管理',
    instances: '实例管理',
    chat: '对话测试',
  }[page];
  // Refresh data
  if (page === 'dashboard') loadDashboard();
  if (page === 'types') loadTypes();
  if (page === 'scenes') loadScenes();
  if (page === 'instances') loadInstances();
  if (page === 'chat') loadChatInstances();
}

// ═══ Dashboard ═══
async function loadDashboard() {
  try {
    const data = await apiGet('/stats');
    $('stat-types').textContent = data.types;
    $('stat-instances').textContent = data.instances;
    $('stat-scenes').textContent = data.scenes;
  } catch (e) {
    console.error(e);
  }
}

$('btn-init-presets').addEventListener('click', async () => {
  try {
    $('btn-init-presets').disabled = true;
    await apiPost('/presets');
    toast('预设类型创建成功', 'success');
    loadDashboard();
    await loadTypes();  // 同步更新 state.types，解除后续页面 Guard
  } catch (e) {
    toast('创建失败: ' + e.message, 'error');
  } finally {
    $('btn-init-presets').disabled = false;
  }
});

// ═══ Types ═══
async function loadTypes() {
  try {
    const data = await apiGet('/types');
    state.types = data.types || [];
    renderTypesTable();
  } catch (e) {
    $('types-table-container').innerHTML = `<div class="empty-state"><p>加载失败: ${e.message}</p></div>`;
  }
}

function renderTypesTable() {
  const container = $('types-table-container');
  if (state.types.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="icon">◉</div>
        <h3>暂无类型</h3>
        <p>点击「创建预设类型」或「新建类型」开始</p>
      </div>`;
    return;
  }
  const rows = state.types.map(t => `
    <tr>
      <td><span class="tag tag-blue">${t.type_id}</span></td>
      <td><strong>${t.name}</strong></td>
      <td>${t.description || '-'}</td>
      <td>${t.demographics?.role || '-'}</td>
      <td>${t.demographics?.age || '-'}</td>
      <td>${t.behavioral_traits?.skepticism_level ?? '-'}</td>
      <td>${t.behavioral_traits?.price_sensitivity ?? '-'}</td>
      <td>
        <button class="btn btn-danger btn-sm" onclick="deleteType('${t.type_id}')">删除</button>
      </td>
    </tr>
  `).join('');
  container.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>ID</th><th>名称</th><th>描述</th><th>角色</th><th>年龄</th><th>怀疑度</th><th>价格敏感</th><th>操作</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>`;
}

$('btn-new-type').addEventListener('click', () => openModal('modal-type'));

$('btn-save-type').addEventListener('click', async () => {
  const payload = {
    type_id: $('type-id').value.trim(),
    name: $('type-name').value.trim(),
    description: $('type-desc').value.trim(),
    demographics: {
      age: parseInt($('type-age').value) || 30,
      role: $('type-role').value.trim() || '用户',
      company_size: '未知',
      industry: '未知',
      location: '未知',
      years_experience: 3,
    },
    behavioral_traits: {
      communication: '直接',
      skepticism_level: parseFloat($('type-skepticism').value) || 0.5,
      price_sensitivity: parseFloat($('type-price').value) || 0.5,
      risk_tolerance: '中',
    },
    variation_config: {},
  };
  if (!payload.type_id || !payload.name) {
    toast('请填写 ID 和名称', 'error'); return;
  }
  try {
    await apiPost('/types', payload);
    closeModal('modal-type');
    toast('类型创建成功', 'success');
    loadTypes();
  } catch (e) {
    toast('创建失败: ' + e.message, 'error');
  }
});

window.deleteType = async function(typeId) {
  if (!confirm(`确定删除类型 ${typeId} 吗？`)) return;
  try {
    await apiDelete('/types/' + typeId);
    toast('已删除', 'success');
    loadTypes();
  } catch (e) {
    toast('删除失败: ' + e.message, 'error');
  }
};

// ═══ Scenes ═══
async function loadScenes() {
  try {
    const data = await apiGet('/scenes');
    state.scenes = data.scenes || [];
    renderScenesTable();
  } catch (e) {
    $('scenes-table-container').innerHTML = `<div class="empty-state"><p>加载失败: ${e.message}</p></div>`;
  }
}

function renderScenesTable() {
  const container = $('scenes-table-container');
  if (state.scenes.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="icon">▥</div>
        <h3>暂无场景</h3>
        <p>创建场景来组织虚拟人交互</p>
      </div>`;
    return;
  }
  const rows = state.scenes.map(s => {
    const hasInstances = (s.participant_instance_ids || []).length > 0;
    return `
      <tr>
        <td><strong>${s.name}</strong></td>
        <td>${s.description || '-'}</td>
        <td>${s.participant_configs?.length || 0} 种类型</td>
        <td>${s.participant_instance_ids?.length || 0} 个实例</td>
        <td>
          ${hasInstances
            ? '<span class="tag tag-green">已实例化</span>'
            : '<span class="tag tag-orange">未实例化</span>'}
        </td>
        <td>
          ${!hasInstances
            ? `<button class="btn btn-primary btn-sm" onclick="instantiateScene('${s.scene_id}')">实例化</button>`
            : `<button class="btn btn-secondary btn-sm" onclick="viewScene('${s.scene_id}')">查看</button>`}
          <button class="btn btn-danger btn-sm" onclick="deleteScene('${s.scene_id}')">删除</button>
        </td>
      </tr>
    `;
  }).join('');
  container.innerHTML = `<table><thead><tr>
    <th>名称</th><th>描述</th><th>配置</th><th>实例</th><th>状态</th><th>操作</th>
  </tr></thead><tbody>${rows}</tbody></table>`;
}

$('btn-new-scene').addEventListener('click', () => {
  if (state.types.length === 0) {
    toast('请先创建人格类型（仪表盘 → 创建预设类型）', 'error');
    return;
  }
  openModal('modal-scene');
  renderParticipantSelectors();
});

function renderParticipantSelectors() {
  const container = $('scene-participants-list');
  if (state.types.length === 0) {
    container.innerHTML = '<p style="color:var(--ink-tertiary);font-size:12px;">请先创建人格类型</p>';
    return;
  }
  const options = state.types.map(t => `<option value="${t.type_id}">${t.name} (${t.type_id})</option>`).join('');
  container.innerHTML = `
    <div class="participant-row" style="display:flex;gap:10px;margin-bottom:8px;align-items:center;">
      <select class="form-select participant-type" style="flex:1;">${options}</select>
      <input type="number" class="form-input participant-count" value="1" min="1" max="10" style="width:70px;">
      <button class="btn btn-danger btn-sm" onclick="this.parentElement.remove()">✕</button>
    </div>`;
}

$('btn-add-participant').addEventListener('click', () => {
  const container = $('scene-participants-list');
  if (state.types.length === 0) return;
  const options = state.types.map(t => `<option value="${t.type_id}">${t.name} (${t.type_id})</option>`).join('');
  const div = document.createElement('div');
  div.className = 'participant-row';
  div.style.cssText = 'display:flex;gap:10px;margin-bottom:8px;align-items:center;';
  div.innerHTML = `
    <select class="form-select participant-type" style="flex:1;">${options}</select>
    <input type="number" class="form-input participant-count" value="1" min="1" max="10" style="width:70px;">
    <button class="btn btn-danger btn-sm" onclick="this.parentElement.remove()">✕</button>
  `;
  container.appendChild(div);
});

$('btn-save-scene').addEventListener('click', async () => {
  const rows = $$('.participant-row');
  const configs = [];
  rows.forEach(row => {
    const typeId = row.querySelector('.participant-type').value;
    const count = parseInt(row.querySelector('.participant-count').value) || 1;
    configs.push({ type_id: typeId, count });
  });
  const payload = {
    name: $('scene-name').value.trim(),
    description: $('scene-desc').value.trim(),
    scenario: $('scene-scenario').value.trim(),
    participant_configs: configs,
  };
  if (!payload.name) { toast('请填写场景名称', 'error'); return; }
  try {
    await apiPost('/scenes', payload);
    closeModal('modal-scene');
    toast('场景创建成功', 'success');
    loadScenes();
  } catch (e) {
    toast('创建失败: ' + e.message, 'error');
  }
});

window.instantiateScene = async function(sceneId) {
  try {
    await apiPost('/scenes/' + sceneId + '/instantiate');
    toast('场景实例化成功', 'success');
    loadScenes();
    loadInstances();
  } catch (e) {
    toast('实例化失败: ' + e.message, 'error');
  }
};

window.viewScene = async function(sceneId) {
  try {
    const data = await apiGet('/scenes/' + sceneId);
    const scene = data.scene;
    const participants = scene.participants || [];
    let html = `<div class="detail-section"><div class="detail-section-title">基本信息</div>
      <div class="detail-grid">
        <div class="detail-item"><div class="detail-item-label">ID</div><div class="detail-item-value">${scene.scene_id}</div></div>
        <div class="detail-item"><div class="detail-item-label">名称</div><div class="detail-item-value">${scene.name}</div></div>
        <div class="detail-item"><div class="detail-item-label">描述</div><div class="detail-item-value">${scene.description || '-'}</div></div>
        <div class="detail-item"><div class="detail-item-label">实例数</div><div class="detail-item-value">${participants.length}</div></div>
      </div></div>`;
    if (participants.length) {
      html += `<div class="detail-section"><div class="detail-section-title">参与者</div><div class="detail-grid">`;
      participants.forEach(p => {
        html += `<div class="detail-item">
          <div class="detail-item-label">${p.name} <span class="tag tag-blue">${p.type_id}</span></div>
          <div class="detail-item-value">${p.demographics?.role}, ${p.demographics?.age}岁, 怀疑度${p.behavioral_traits?.skepticism_level}</div>
        </div>`;
      });
      html += '</div></div>';
    }
    $('detail-title').textContent = '场景详情：' + scene.name;
    $('detail-body').innerHTML = html;
    openModal('modal-detail');
  } catch (e) {
    toast('加载失败: ' + e.message, 'error');
  }
};

window.deleteScene = async function(sceneId) {
  if (!confirm('确定删除此场景吗？')) return;
  try {
    await apiDelete('/scenes/' + sceneId);
    toast('已删除', 'success');
    loadScenes();
  } catch (e) {
    toast('删除失败: ' + e.message, 'error');
  }
};

// ═══ Instances ═══
async function loadInstances() {
  try {
    const data = await apiGet('/instances');
    state.instances = data.instances || [];
    renderInstancesTable();
  } catch (e) {
    $('instances-table-container').innerHTML = `<div class="empty-state"><p>加载失败: ${e.message}</p></div>`;
  }
}

function renderInstancesTable() {
  const container = $('instances-table-container');
  if (state.instances.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="icon">◎</div>
        <h3>暂无实例</h3>
        <p>通过场景实例化或手动创建</p>
      </div>`;
    return;
  }
  const rows = state.instances.map(inst => `
    <tr>
      <td><strong>${inst.name}</strong></td>
      <td><span class="tag tag-blue">${inst.type_id}</span></td>
      <td>${inst.demographics?.role || '-'}, ${inst.demographics?.age || '-'}岁</td>
      <td>${inst.behavioral_traits?.skepticism_level?.toFixed(2) || '-'}</td>
      <td>${inst.behavioral_traits?.price_sensitivity?.toFixed(2) || '-'}</td>
      <td>${inst.memory?.trust_level?.toFixed(2) || '0.30'}</td>
      <td>${inst.message_history_count || 0}</td>
      <td>
        <button class="btn btn-secondary btn-sm" onclick="viewInstance('${inst.instance_id}')">详情</button>
        <button class="btn btn-danger btn-sm" onclick="deleteInstance('${inst.instance_id}')">删除</button>
      </td>
    </tr>
  `).join('');
  container.innerHTML = `<table><thead><tr>
    <th>名称</th><th>类型</th><th>身份</th><th>怀疑度</th><th>价格敏感</th><th>信任度</th><th>对话数</th><th>操作</th>
  </tr></thead><tbody>${rows}</tbody></table>`;
}

$('btn-new-instance').addEventListener('click', () => {
  const sel = $('instance-type-id');
  if (state.types.length === 0) {
    toast('请先创建人格类型（仪表盘 → 创建预设类型）', 'error');
    return;
  }
  sel.innerHTML = state.types.map(t => `<option value="${t.type_id}">${t.name}</option>`).join('');
  openModal('modal-instance');
});

$('btn-save-instance').addEventListener('click', async () => {
  const typeId = $('instance-type-id').value;
  if (!typeId) {
    toast('请选择人格类型', 'error');
    return;
  }
  const payload = {
    type_id: typeId,
    name: $('instance-name').value.trim() || null,
    variation_seed: $('instance-seed').value ? parseInt($('instance-seed').value) : null,
  };
  try {
    await apiPost('/instances', payload);
    closeModal('modal-instance');
    toast('实例创建成功', 'success');
    loadInstances();
  } catch (e) {
    toast('创建失败: ' + e.message, 'error');
  }
});

window.viewInstance = async function(instanceId) {
  try {
    const data = await apiGet('/instances/' + instanceId);
    const inst = data.instance;
    let html = `<div class="detail-section"><div class="detail-section-title">基本信息</div>
      <div class="detail-grid">
        <div class="detail-item"><div class="detail-item-label">ID</div><div class="detail-item-value">${inst.instance_id}</div></div>
        <div class="detail-item"><div class="detail-item-label">名称</div><div class="detail-item-value">${inst.name}</div></div>
        <div class="detail-item"><div class="detail-item-label">类型</div><div class="detail-item-value">${inst.type_id}</div></div>
        <div class="detail-item"><div class="detail-item-label">角色</div><div class="detail-item-value">${inst.demographics?.role || '-'}</div></div>
        <div class="detail-item"><div class="detail-item-label">年龄</div><div class="detail-item-value">${inst.demographics?.age || '-'}</div></div>
        <div class="detail-item"><div class="detail-item-label">经验</div><div class="detail-item-value">${inst.demographics?.years_experience || '-'}年</div></div>
      </div></div>`;
    html += `<div class="detail-section"><div class="detail-section-title">行为特征</div>
      <div class="detail-grid">
        <div class="detail-item"><div class="detail-item-label">怀疑度</div><div class="detail-item-value">${inst.behavioral_traits?.skepticism_level}</div></div>
        <div class="detail-item"><div class="detail-item-label">价格敏感度</div><div class="detail-item-value">${inst.behavioral_traits?.price_sensitivity}</div></div>
        <div class="detail-item"><div class="detail-item-label">沟通风格</div><div class="detail-item-value">${inst.behavioral_traits?.communication || '-'}</div></div>
        <div class="detail-item"><div class="detail-item-label">风险承受</div><div class="detail-item-value">${inst.behavioral_traits?.risk_tolerance || '-'}</div></div>
      </div></div>`;
    html += `<div class="detail-section"><div class="detail-section-title">记忆状态</div>
      <div class="detail-grid">
        <div class="detail-item"><div class="detail-item-label">信任度</div><div class="detail-item-value">${inst.memory?.trust_level}</div></div>
        <div class="detail-item"><div class="detail-item-label">情绪</div><div class="detail-item-value">${inst.memory?.emotional_state}</div></div>
        <div class="detail-item"><div class="detail-item-label">接触次数</div><div class="detail-item-value">${inst.memory?.exposure_count}</div></div>
      </div></div>`;
    if (inst.variation && Object.keys(inst.variation).length) {
      html += `<div class="detail-section"><div class="detail-section-title">差异化参数</div>
        <div class="detail-grid">`;
      for (const [k, v] of Object.entries(inst.variation)) {
        html += `<div class="detail-item"><div class="detail-item-label">${k}</div><div class="detail-item-value">${v}</div></div>`;
      }
      html += '</div></div>';
    }
    html += `<div class="detail-section"><div class="detail-section-title">System Prompt</div>
      <div class="prompt-preview">${escapeHtml(inst.system_prompt || '(空)')}</div></div>`;
    $('detail-title').textContent = '实例详情：' + inst.name;
    $('detail-body').innerHTML = html;
    openModal('modal-detail');
  } catch (e) {
    toast('加载失败: ' + e.message, 'error');
  }
};

window.deleteInstance = async function(instanceId) {
  if (!confirm('确定删除此实例吗？')) return;
  try {
    await apiDelete('/instances/' + instanceId);
    toast('已删除', 'success');
    loadInstances();
  } catch (e) {
    toast('删除失败: ' + e.message, 'error');
  }
};

// ═══ Chat ═══
async function loadChatInstances() {
  try {
    const data = await apiGet('/instances');
    state.instances = data.instances || [];
    renderChatInstanceList();
  } catch (e) {
    $('chat-instance-list').innerHTML = `<div class="empty-state"><p>加载失败</p></div>`;
  }
}

function renderChatInstanceList() {
  const container = $('chat-instance-list');
  if (state.instances.length === 0) {
    container.innerHTML = `<div class="empty-state" style="padding:30px 10px;"><p>暂无可用实例</p></div>`;
    return;
  }
  container.innerHTML = state.instances.map(inst => `
    <div class="chat-instance-item ${state.chatInstanceId === inst.instance_id ? 'active' : ''}"
         data-id="${inst.instance_id}"
         style="padding:10px 12px;border-radius:8px;cursor:pointer;transition:background .15s;margin-bottom:2px;">
      <div style="font-weight:600;font-size:13px;">${inst.name} <span class="tag tag-blue">${inst.type_id}</span></div>
      <div style="font-size:11px;color:var(--ink-tertiary);margin-top:2px;">
        ${inst.demographics?.role}, ${inst.demographics?.age}岁 · 信任度 ${inst.memory?.trust_level?.toFixed(2) || '0.30'}
      </div>
    </div>
  `).join('');
  container.querySelectorAll('.chat-instance-item').forEach(el => {
    el.addEventListener('click', () => selectChatInstance(el.dataset.id));
  });
}

async function selectChatInstance(id) {
  state.chatInstanceId = id;
  renderChatInstanceList();
  const inst = state.instances.find(i => i.instance_id === id);
  if (!inst) return;
  $('chat-instance-info').innerHTML = `当前对话：<strong>${inst.name}</strong> (${inst.type_id}) · ${inst.demographics?.role} · 信任度 ${inst.memory?.trust_level?.toFixed(2) || '0.30'}`;
  $('chat-input').disabled = false;
  $('chat-send').disabled = false;
  // Load history
  try {
    const data = await apiGet('/instances/' + id);
    const full = data.instance;
    state.chatHistory = full.message_history || [];
    renderChatMessages();
  } catch (e) {
    state.chatHistory = [];
    renderChatMessages();
  }
}

function renderChatMessages() {
  const container = $('chat-messages');
  if (state.chatHistory.length === 0) {
    container.innerHTML = `<div class="empty-state" style="padding-top:120px;">
      <div class="icon">✉</div>
      <h3>开始对话</h3>
      <p>输入消息与虚拟人交互</p>
    </div>`;
    return;
  }
  container.innerHTML = state.chatHistory.map(msg => {
    const isUser = msg.role === 'user';
    return `<div class="chat-bubble ${isUser ? 'user' : 'assistant'}">
      ${escapeHtml(msg.content)}
    </div>`;
  }).join('');
  container.scrollTop = container.scrollHeight;
}

$('chat-send').addEventListener('click', sendChat);
$('chat-input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); }
});

async function sendChat() {
  const input = $('chat-input');
  const msg = input.value.trim();
  if (!msg || !state.chatInstanceId) return;
  input.value = '';
  input.disabled = true;
  $('chat-send').disabled = true;

  // Optimistically add user message
  state.chatHistory.push({ role: 'user', content: msg });
  renderChatMessages();

  try {
    const data = await apiPost('/instances/' + state.chatInstanceId + '/interact', {
      message: msg,
      include_history: true,
      temperature: 0.7,
    });
    state.chatHistory.push({ role: 'assistant', content: data.response });
    renderChatMessages();
    // Update trust level display
    const inst = state.instances.find(i => i.instance_id === state.chatInstanceId);
    if (inst && data.memory) {
      inst.memory = data.memory;
      $('chat-instance-info').innerHTML = `当前对话：<strong>${inst.name}</strong> (${inst.type_id}) · ${inst.demographics?.role} · 信任度 ${inst.memory.trust_level.toFixed(2)}`;
    }
  } catch (e) {
    toast('对话失败: ' + e.message, 'error');
  } finally {
    input.disabled = false;
    $('chat-send').disabled = false;
    input.focus();
  }
}

// ═══ Modal helpers ═══
function openModal(id) { $(id).classList.add('active'); }
function closeModal(id) { $(id).classList.remove('active'); }
$$('[data-close]').forEach(el => {
  el.addEventListener('click', () => closeModal(el.dataset.close));
});
$$('.modal-overlay').forEach(el => {
  el.addEventListener('click', e => { if (e.target === el) el.classList.remove('active'); });
});

// ═══ Utils ═══
function escapeHtml(str) {
  if (!str) return '';
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// ═══ Init ═══
loadDashboard();
