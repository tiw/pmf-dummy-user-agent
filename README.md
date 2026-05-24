# 🤖 虚拟用户生成 Agent —— LLM 增强版

> **目的**：利用 LLM 模拟真实用户，快速验证产品有效性。
>
> **理论基础**：基于「五层设计框架」——让 LLM **以特定身份和场景约束来运行**，而非简单"扮演"。
>
> **LLM 后端**：DeepSeek API（所有 LLM 场景统一使用）

---

## 核心升级：LLM 增强

相比模板版，LLM 增强版在三个关键环节引入 DeepSeek：

| 环节 | 模板版 | LLM 增强版 |
|------|--------|-----------|
| **信息收集** | 手动填写 29 个字段 | 用户说一句话，DeepSeek 自动扩展完整 persona |
| **Prompt 生成** | 字符串模板拼接 | DeepSeek 优化：增加人味、内心独白、强化约束 |
| **质量保障** | 无 | DeepSeek Self-Critique：打分 + 指出问题 + 给建议 |

---

## 环境准备

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 DeepSeek API Key

```bash
export DEEPSEEK_API_KEY="your-api-key-here"
```

建议添加到 `~/.zshrc` 或 `~/.bashrc`：

```bash
echo 'export DEEPSEEK_API_KEY="your-key"' >> ~/.zshrc
source ~/.zshrc
```

---

## 快速开始

### 模式一：LLM 快速扩展（推荐）

只需描述你想模拟的用户，DeepSeek 自动填充所有细节：

```bash
python persona_generator.py
# 选择「是」使用 LLM 快速扩展模式
# 输入一段描述，例如：
# "一个大厂5年经验的产品经理，对数据分析工具很挑剔，
#  之前用过 Tableau 觉得太贵，正在看国产替代方案。
#  他正在参加一个 B2B SaaS 的 demo 会议，时间很紧。"
```

Agent 会自动：
1. 🧠 调用 DeepSeek 扩展为完整五层框架
2. 📋 展示草案供你确认
3. ✨ 调用 DeepSeek 优化 System Prompt（增加人味）
4. 🔍 调用 DeepSeek Self-Critique（质量评分 + 问题诊断）
5. 💾 输出 YAML + Markdown 文件

### 模式二：手动逐字段填写

```bash
python persona_generator.py
# 选择「否」进入手动模式
# 逐字段填写（有默认值，可回车跳过）
```

手动模式也会经过 LLM 优化和 Self-Critique（如果配置了 API Key）。

### 模式三：仅查看信息清单

```bash
python persona_generator.py --guide
```

---

## 五层设计框架

| 层级 | 名称 | 作用 |
|------|------|------|
| Layer 1 | **角色定义层** | demographics + psychographics + behavioral traits |
| Layer 2 | **大模型条件化** | System Prompt（LLM 优化后的自然语言约束） |
| Layer 3 | **场景注入层** | 当前场景、初始态度、时间压力、参与动机 |
| Layer 4 | **记忆系统层** | trust_level、emotional_state、exposure_count |
| Layer 5 | **行为引擎层** | 注意力关键词、怀疑触发词、知识边界 |

---

## 输出文件

生成在 `generated/` 目录：

```
generated/
├── {role}_{age}.yaml   # 完整的五层框架 + Self-Critique 结果
└── {role}_{age}.md     # 可直接复制的 System Prompt
```

**使用方式**：将 `.md` 文件中的 System Prompt 粘贴到 ChatGPT / Claude / DeepSeek 等 LLM 的 system prompt 中，开始对话测试你的产品。

---

## LLM 质量保障机制

### 1. 自动扩展的丰富性

DeepSeek 会根据你的一句话描述，自动推断：
- 具体的技术栈（不是"常用工具"而是"Figma, Notion, Mixpanel"）
- 真实的痛点（不是"效率低"而是"需求验证要跑 3 轮评审，每次都改方向"）
- 合理的量化指标（怀疑度 0.7 对应"对 vendor 非常敏感，会立刻 challenge"）
- 竞品经历（具体产品名，如"用过 Tableau，被价格劝退"）

### 2. Prompt 优化：从"简历"到"真人"

DeepSeek 会把模板化的描述改写得更自然：

| 模板版 | LLM 优化版 |
|--------|-----------|
| "怀疑程度高（0.7）" | "你已经被 4 个 vendor 坑过，现在听到'一键部署'就会冷笑" |
| "时间压力：高" | "你只有 25 分钟，之后要赶一个 production incident，所以如果 10 分钟内看不到价值，你会找借口结束会议" |
| "沟通风格：直接" | "你会打断对方说'说重点'，如果对方还在讲公司愿景，你会直接问'多少钱，多久能上线'" |

### 3. Self-Critique 质检

生成后 DeepSeek 会对 Prompt 做质量评估，输出：

```yaml
llm_critique:
  score: 8
  strengths:
    - "角色细节丰富，有具体的竞品经历"
    - "约束明确，不容易角色漂移"
  issues:
    - "怀疑度 0.7 但没有体现'会追问同行案例'"
    - "时间压力的表达可以更紧迫"
  suggestions:
    - "增加一条：'如果对方给不出同行业案例，你会直接结束对话'"
```

---

## 为什么用 LLM 模拟用户？

在找真实用户之前，先用虚拟用户做**预演测试**：

1. **5 分钟生成 persona** → 立即测试产品 pitch
2. **低成本发现问题**：虚拟用户会尖锐 challenge，提前暴露问题
3. **对比不同用户类型**：怀疑型、开放型、价格敏感型分别测试
4. **打磨话术**：测试哪些卖点能打动用户，哪些词汇触发反感

---

## 进阶：直接用 LLM 运行模拟对话

如果你想让 Agent 直接以生成的 persona 和你对话（而不只是输出 Prompt），可以扩展：

```python
from deepseek_client import chat_completion

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "你好，我是 TestApp 的销售，想给你演示一下我们的产品..."}
]

response = chat_completion(messages)
print(response)  # 虚拟用户的回复
```

（后续可基于此扩展为内置对话模拟器）

---

## 项目结构

```
.
├── persona_generator.py    # 主 Agent 脚本（交互式 CLI）
├── deepseek_client.py      # DeepSeek API 封装（所有 LLM 调用统一入口）
├── requirements.txt        # Python 依赖
├── README.md               # 本文件
└── generated/              # 生成的虚拟用户文件
```

**所有 LLM 调用都通过 `deepseek_client.py`**，便于统一配置模型、温度、重试等参数。

---

## 一句话总结

> **虚拟用户 = LLM 自动扩展（丰富细节）+ 结构化身份 + LLM 优化 Prompt（人味化）+ LLM Self-Critique（质量保障）。不是让 LLM "扮演"人，而是给 LLM 装上"身份操作系统"，让它以特定身份运行。**
