# 🤖 虚拟用户生成 Agent —— LLM 增强版

> **目的**：利用 LLM 模拟真实用户，快速验证产品有效性。
>
> **理论基础**：基于「五层设计框架」——让 LLM **以特定身份和场景约束来运行**，而非简单"扮演"。
>
> **LLM 后端**：DeepSeek / Qwen API（自动切换）

---

## 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                         VMU 架构                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │ PersonaType │───▶│  Variation  │───▶│   Instance  │         │
│  │   人格模板   │    │   差异化生成  │    │   人格实例   │         │
│  └─────────────┘    └─────────────┘    └──────┬──────┘         │
│        ▲                                       │                │
│        │                                       │                │
│   CRUD 管理                              ┌─────▼─────┐          │
│   (Manager)                              │  Agent    │          │
│                                          │ LLM 交互  │          │
│  ┌─────────────┐    ┌─────────────┐     └─────┬─────┘          │
│  │    Scene    │◀───│ instantiate │◀──────────┘                │
│  │   场景管理   │    │   场景实例化  │                             │
│  └─────────────┘    └─────────────┘                             │
│                                                                 │
│  Storage: JSON 文件持久化 (data/types, instances, scenes)        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 核心概念

| 概念 | 说明 | 类比 |
|------|------|------|
| **PersonaType** | 人格类型模板，定义一类人的共同特征 | 类 (Class) |
| **PersonaInstance** | 基于模板的差异化实例，带对话状态和记忆 | 对象 (Object) |
| **Scene** | 场景，包含多种类型、多个实例的交互环境 | 容器 (Container) |
| **Agent** | 包装实例与 LLM 交互，管理 message history | 代理 (Agent) |

---

## 环境准备

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
export DEEPSEEK_API_KEY="your-api-key-here"
# 或
export DASHSCOPE_API_KEY="your-qwen-key-here"
```

---

## 快速开始

### 模式一：CLI 生成单个 Persona（原有功能）

```bash
python persona_generator.py
```

交互式生成单个虚拟用户 prompt。

### 模式二：Web API（原有功能）

```bash
python web_server.py
# 访问 http://localhost:8000
```

### 模式三：VMU 场景管理（新增）

```python
from vmu import PersonaManager, PersonaAgent
from vmu.models import SceneParticipant
from deepseek_client import chat_completion

# 1. 初始化管理器
manager = PersonaManager()

# 2. 创建预设人格类型（4种）
manager.create_preset_types()

# 3. 创建场景
types = ["anxious_buyer", "rational_analyst", "tech_skeptic", "impulsive_decider"]
scene = manager.create_scene(
    name="产品演示会",
    scenario="评估一款新的项目管理 SaaS 工具",
    participant_configs=[
        SceneParticipant(type_id="anxious_buyer", count=2),
        SceneParticipant(type_id="rational_analyst", count=1),
        SceneParticipant(type_id="tech_skeptic", count=1),
        SceneParticipant(type_id="impulsive_decider", count=1),
    ],
)

# 4. 实例化场景（自动生成差异化个体）
manager.instantiate_scene(scene.scene_id)

# 5. 获取场景中的虚拟人
participants = manager.get_scene_instances(scene.scene_id)

# 6. 与虚拟人交互
for p in participants:
    agent = PersonaAgent(instance=p, llm_client=chat_completion)
    result = agent.interact("你好，想给你介绍我们的产品...")
    print(f"{p.name}: {result.response}")
```

运行完整示例：

```bash
python examples/run_scene.py
```

---

## VMU API 详解

### PersonaType 管理

```python
# 创建类型
pt = manager.create_type(
    type_id="budget_controller",
    name="预算控制者",
    demographics=Demographics(age=45, role="财务总监"),
    behavioral_traits=BehavioralTraits(price_sensitivity=0.95),
    variation_config={
        "demographics.age": {"range": [38, 55]},
        "behavioral_traits.price_sensitivity": {"range": [0.8, 1.0]},
    },
)

# CRUD
manager.get_type("budget_controller")
manager.list_types()
manager.update_type("budget_controller", name="成本控制者")
manager.delete_type("budget_controller")
```

### 实例化（差异化生成）

```python
# 自动生成差异化参数
inst = manager.instantiate("budget_controller")
print(inst.variation)  # {'demographics.age': 42, 'behavioral_traits.price_sensitivity': 0.89}

# 手动指定变异
inst = manager.instantiate(
    "budget_controller",
    variation={"demographics.age": 50, "demographics.location": "上海"},
)

# 指定随机种子（可复现）
inst = manager.instantiate("budget_controller", variation_seed=42)
```

### 场景管理

```python
# 创建场景
scene = manager.create_scene(
    name="采购决策会议",
    scenario="评估 CRM 系统采购",
    participant_configs=[
        SceneParticipant(type_id="anxious_buyer", count=2),
        SceneParticipant(type_id="rational_analyst", count=1),
    ],
)

# 实例化场景中的所有参与者
manager.instantiate_scene(scene.scene_id)

# 获取场景参与者
participants = manager.get_scene_instances(scene.scene_id)
```

### LLM 交互

```python
# 基础交互
agent = PersonaAgent(instance=inst, llm_client=chat_completion)
result = agent.interact("这是我们的报价方案...")
print(result.response)
print(result.updated_memory.trust_level)

# 自动持久化
agent = PersonaAgent(
    instance=inst,
    llm_client=chat_completion,
    auto_persist=True,
    storage=manager.storage,
)

# 带历史的多轮对话
result2 = agent.interact("那如果按年付呢？", include_history=True)

# 导出对话记录
print(agent.export_conversation())

# 重置对话历史
agent.reset_history()
```

---

## 预设人格类型

| 类型 ID | 名称 | 特征 |
|---------|------|------|
| `anxious_buyer` | 焦虑型买家 | 怀疑度高(0.6-0.9)，价格敏感，需要安全感 |
| `rational_analyst` | 理性分析师 | 数据驱动，关注 ROI，要求具体案例 |
| `tech_skeptic` | 技术怀疑者 | 对新技术持疑，偏好成熟方案，challenge 架构 |
| `impulsive_decider` | 冲动决策者 | 直觉驱动，快节奏，不耐烦 |

每种类型可以通过 `variation_config` 定义差异范围，实例化时自动生成不同参数。

---

## 持久化

所有数据自动保存到 `data/` 目录：

```
data/
├── types/      # PersonaType JSON
├── instances/  # PersonaInstance JSON（含对话历史）
└── scenes/     # Scene JSON
```

JSON 格式，可直接查看和手动编辑。

---

## 五层设计框架

| 层级 | 名称 | 作用 | 对应模型 |
|------|------|------|----------|
| Layer 1 | **角色定义层** | demographics + psychographics + behavioral traits | `PersonaType` |
| Layer 2 | **大模型条件化** | System Prompt | `PersonaInstance.system_prompt` |
| Layer 3 | **场景注入层** | 当前场景、初始态度、时间压力 | `SceneContext` |
| Layer 4 | **记忆系统层** | trust_level、emotional_state、exposure_count | `MemoryState` |
| Layer 5 | **行为引擎层** | 注意力关键词、怀疑触发词、知识边界 | `BehaviorEngine` |

---

## 项目结构

```
.
├── vmu/                      # 虚拟人管理核心库（新增）
│   ├── __init__.py
│   ├── models.py             # Pydantic 数据模型
│   ├── storage.py            # JSON 持久化
│   ├── manager.py            # 核心管理器（CRUD + 实例化）
│   ├── agent.py              # LLM 交互 Agent
│   ├── prompts.py            # Prompt 渲染 + 变异生成
│   └── testing/              # 🆕 测试外部 Agent 模块
│       ├── tester.py         #   测试引擎
│       ├── adapters.py       #   Agent 适配器
│       ├── evaluator.py      #   对话评估器
│       └── report.py         #   测试报告
├── examples/
│   ├── run_scene.py          # 场景运行示例
│   └── test_external_agent.py  # 🆕 测试外部 Agent 示例
├── data/                     # 持久化数据（运行时生成）
├── persona_generator.py      # CLI 单人生成（原有）
├── deepseek_client.py        # LLM API 封装（原有）
├── web_server.py             # FastAPI Web 服务（原有 + Testing API）
├── requirements.txt
└── README.md
```

---

## 🆕 模式四：用虚拟人测试外部 Agent

pmf-dummy-user 不仅可以生成虚拟人，还可以作为「测试基础设施」，帮助其他 agent 完成测试。

### 核心思路

```
你的 Agent ──▶ pmf-dummy-user ──▶ 虚拟人回复 ──▶ 你的 Agent
    │                                    │
    └──────────── 评估报告 ◀─────────────┘
```

### 方式一：Python SDK（推荐）

```python
from vmu.testing import DummyUserTester, AgentAdapter

# 初始化测试器
tester = DummyUserTester()

# 测试一个函数型 agent
def my_agent(message: str) -> str:
    return "我是一个销售 agent，收到: " + message

session = tester.test_agent(
    agent=my_agent,
    persona_type="anxious_buyer",  # 用焦虑型买家测试
    opening="你好，我想了解一下你们的产品",
    rounds=5,
)

# 查看对话
for turn in session.turns:
    print(f"Agent: {turn.agent_message}")
    print(f"User:  {turn.user_response}")

# 生成评估报告
from vmu.testing import ConversationEvaluator, TestReport

evaluator = ConversationEvaluator()
report = TestReport.from_session(session)
print(report.to_markdown())
```

### 方式二：测试 HTTP API 型 Agent

```python
# 连接你的 agent 服务
adapter = AgentAdapter.http(
    url="http://localhost:3000/chat",
    message_field="message",
    response_field="response",
)

session = tester.test_agent(agent=adapter, persona_type="rational_analyst", rounds=3)
```

### 方式三：交互式测试（HTTP API）

启动服务：
```bash
python web_server.py
```

外部 agent 通过 API 逐步与虚拟人交互：

```bash
# 1. 创建测试会话
curl -X POST http://localhost:8000/api/v1/testing/sessions \
  -H "Content-Type: application/json" \
  -d '{"persona_type": "anxious_buyer"}'
# 返回: {"session_id": "session_xxx", ...}

# 2. 发送消息（agent → 虚拟人），返回虚拟人回复
curl -X POST http://localhost:8000/api/v1/testing/sessions/session_xxx/message \
  -H "Content-Type: application/json" \
  -d '{"message": "你好，我是销售小李..."}'
# 返回: {"user_response": "（虚拟人回复）", "trust_level": 0.3, ...}

# 3. 获取评估报告
curl http://localhost:8000/api/v1/testing/sessions/session_xxx/report

# 4. 关闭会话
curl -X DELETE http://localhost:8000/api/v1/testing/sessions/session_xxx
```

### Testing API 端点一览

| 方法 | 端点 | 说明 |
|------|------|------|
| POST | `/api/v1/testing/sessions` | 创建测试会话 |
| GET | `/api/v1/testing/sessions` | 列出所有会话 |
| GET | `/api/v1/testing/sessions/{id}` | 获取会话详情 |
| POST | `/api/v1/testing/sessions/{id}/message` | 发送消息给虚拟人 |
| POST | `/api/v1/testing/sessions/{id}/evaluate` | 评估对话质量 |
| GET | `/api/v1/testing/sessions/{id}/report` | 获取测试报告 |
| DELETE | `/api/v1/testing/sessions/{id}` | 关闭会话 |
| POST | `/api/v1/testing/run` | 快速运行完整测试 |
| POST | `/api/v1/testing/scene/{id}/run` | 场景测试（多虚拟人） |

### 完整示例

```bash
python examples/test_external_agent.py
```

---

## 一句话总结

> **虚拟用户 = 类型模板（PersonaType）× 差异化变异（Variation）+ 场景上下文（Scene）+ LLM Agent 交互。同类型可生成多个有差异的实例，统一管理、持久化、可交互。现在，这些虚拟人还可以作为「测试用户」，帮助外部 agent 完成自动化测试。**
