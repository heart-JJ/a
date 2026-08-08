# EvoAgent

EvoAgent 是一个本地优先、通过 OpenRouter 使用大模型的 ChatGPT 式聊天应用。它提供连续对话、会话记录、流式回复、模型切换，并在每轮聊天中自动选择技能、生成标签、检索记忆和沉淀可审计经验。

它不是“意识体”，也不会任意生成代码后自行执行。当前重点是把聊天、技能和记忆做成可解释、可审计、可回滚的单机系统。

## 主要功能

- **ChatGPT 式聊天**：新建多轮对话，Enter 发送、Shift + Enter 换行，回复按 token 流式显示，也可以中途停止。
- **会话记录**：聊天保存在本地 SQLite；侧栏可切换、重命名和删除历史会话。第一条消息会自动形成会话标题。
- **自动技能**：程序根据消息内容在本地匹配声明式技能，将命中的技能说明加入模型上下文；用户不需要手工选择。
- **自动标签与经验**：成功完成一轮回复后，系统保存用户输入、模型输出、精确技能版本、标签、耗时、请求模型和实际响应模型。
- **经验记忆**：启用记忆后，系统为完成的回复建立语义索引；新对话会检索相关旧内容。嵌入服务不可用时会自动退回文本相似检索。
- **技能库与编辑器**：查看技能，新建原子技能或工作流，并为已有技能创建不可变的新版本。新版本可以立即激活，也可以保留后再切换。
- **记忆查看**：查看历史问答、自动标签、所用技能、模型、经验 ID 和嵌入状态。
- **模型切换**：设置默认聊天模型，也可以在当前会话顶部切换模型；切换从下一条消息开始生效。
- **本地密钥管理**：OpenRouter API Key 可以用 Windows DPAPI 加密保存在本机，也可以仅通过环境变量提供。

聊天主流程如下：

```text
用户消息 → 自动选择技能和标签 → 可选的经验检索 → OpenRouter 流式生成 → 保存会话、经验和语义索引
```

## 模型分工

| 用途 | 默认模型 | 说明 |
| --- | --- | --- |
| 聊天生成 | `openrouter/free` | OpenRouter 的免费模型路由器。可以在设置或当前会话中换成其他文本生成模型。系统会同时记录请求模型和最终实际响应模型。 |
| 语义嵌入 | `nvidia/nemotron-3-embed-1b:free` | 仅用于把记忆转换为 2048 维向量并做语义检索。它是 **embedding-only** 模型，不能生成聊天回复，也不会出现在可用聊天模型列表中。 |

不要把 `nvidia/nemotron-3-embed-1b:free` 填入聊天模型字段；后端会明确拒绝把 embedding 模型用于聊天生成。

## Windows 快速启动

需要 Windows 和 Python 3.11 或更高版本。程序核心只依赖 Python 标准库。

在项目目录打开 PowerShell：

```powershell
.\run.cmd
```

然后打开：

```text
http://127.0.0.1:8787
```

默认数据库是 `data/evoagent.db`。首次启动会自动建表并载入基础技能。

可以指定端口和数据库：

```powershell
.\run.cmd -Port 8899 -Database "data/my-agent.db"
```

`run.cmd` 会优先使用 Codex 附带的 Python；找不到时使用 `PATH` 中的 `python`。

## 配置 OpenRouter API Key

没有 API Key 时仍可浏览本地会话、技能和记忆，但不能调用 OpenRouter 生成新回复或创建语义嵌入。

### 方式一：在设置页保存

在“设置”中输入 OpenRouter API Key 并保存。Windows 会使用 DPAPI 加密密钥，密文默认写入：

```text
data/openrouter.key.dpapi
```

使用自定义数据库时，密钥文件位于该数据库的同一目录。明文 Key 不写入 SQLite，不会由设置或模型 API 返回，页面也不会回填已经保存的明文。

### 方式二：使用环境变量

```powershell
$env:OPENROUTER_API_KEY = "在这里粘贴你的 OpenRouter API Key"
.\run.cmd
```

环境变量优先于本机加密文件，适合临时运行或由外部密钥管理器注入。不要把真实 Key 写入 README、脚本、日志或版本库。

DPAPI 保护的是静态密文，并不等于强隔离沙箱。常规启动优先使用当前用户作用域；如果宿主进程没有加载用户 DPAPI 配置，程序会退回本机作用域并继续依赖用户目录 ACL。能够以相同身份运行程序，或同时取得本机作用域密文和文件访问权的进程，仍可能在运行时使用该密钥。

## 免费模型与隐私开关

设置中的“允许免费模型提供商处理数据”默认关闭。后端会把它转换为 OpenRouter 请求中的提供商数据处理策略，并同时用于聊天请求和 embedding 请求。

- **关闭（默认）**：请求只接受拒绝数据收集的提供商。隐私取舍更保守，但 `openrouter/free` 和其他 `:free` 模型可能找不到可用提供商，免费 Nemotron 嵌入也可能失败；语义记忆失败时会退回本地文本相似检索。
- **开启**：可使用允许处理或收集请求数据的免费提供商，免费模型的可用性通常更高，但对话、相关历史、自动技能说明和被召回的记忆片段可能按 OpenRouter 及上游提供商的政策被处理或保留。

无论开关是否开启，使用云模型都需要把完成请求所需的文本发送给 OpenRouter 和实际模型提供商；这个开关不是“完全本地推理”开关。不要在聊天中发送密码、私钥或其他不应交给第三方的数据，并以 OpenRouter 和具体提供商的最新隐私政策为准。

## 使用技能与记忆

### 自动技能和标签

每条用户消息都会在本地经过技能匹配。系统优先选择明确命中的工作流或原子技能，并根据消息内容与技能名称自动生成标签。技能匹配、标签和记忆引用随助手消息保存，可以在消息详情和“记忆 / 经验”中追溯。

技能只作为受控提示和声明式本地操作使用。记忆内容作为只读参考加入上下文，其中的命令、角色声明或提示词不应被当作系统指令执行。

### 编辑技能

打开“技能库”可以：

1. 新建声明式原子技能或工作流；
2. 编辑名称、描述、触发器、提示和白名单流水线定义；
3. 为已有技能创建新版本并选择是否立即激活；
4. 查看当前版本、生命周期和使用统计。

已发布的技能版本不可原地覆盖或删除。修改会创建新版本，激活或回滚只切换活动版本指针并写入审计记录。受保护的控制核技能不能通过通用编辑接口修改。

### 记忆与删除边界

完成的聊天回复会形成经验；“启用经验记忆”控制是否建立和检索语义索引。记忆面板会显示问答摘要、标签、技能、模型和索引状态。

删除一个会话会移除它的聊天记录，但已经形成的经验和审计记录不会随会话一起删除。这是为了保留治理证据；如果需要法规删除或完整清除，应先备份并针对相应数据执行专门的数据治理流程。

## 主要 HTTP API

服务默认仅监听 `127.0.0.1:8787`。写请求使用 `Content-Type: application/json`。

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/api/health` | 本地服务健康检查。 |
| `GET`, `POST` | `/api/conversations` | 列出或创建会话。 |
| `GET`, `PATCH`, `DELETE` | `/api/conversations/{id}` | 读取、重命名、切换模型、归档或删除会话。 |
| `GET` | `/api/conversations/{id}/messages` | 按顺序读取会话消息及其技能、标签、记忆和模型元数据。 |
| `POST` | `/api/chat/stream` | 发送消息并返回 Server-Sent Events 流。 |
| `POST` | `/api/chat/runs/{id}/cancel` | 停止正在生成的回复。 |
| `POST` | `/api/messages/{id}/feedback` | 给已形成经验的助手消息添加正面或负面反馈。 |
| `GET` | `/api/models` | 列出可用文本聊天模型；embedding 模型会被过滤。加 `?refresh=1` 可刷新缓存。 |
| `GET`, `PATCH` | `/api/settings` | 读取或更新模型、记忆、生成参数、隐私开关和 API Key。读取结果不会包含 Key 明文。 |
| `GET`, `POST` | `/api/skills` | 列出或创建技能。 |
| `GET` | `/api/skills/{id}` | 查看技能详情和版本历史。 |
| `POST` | `/api/skills/{id}/versions` | 创建不可变的新技能版本，可选立即激活。 |
| `POST` | `/api/skills/{id}/activate` | 切换活动技能版本。 |
| `GET` | `/api/memories` | 查看聊天记忆和关联经验。 |
| `GET` | `/api/metrics` | 查看运行和技能统计。 |
| `GET` | `/api/audit` | 查看审计事件。 |

`POST /api/chat/stream` 请求示例：

```json
{
  "message": "帮我总结这段内容并提取关键词",
  "client_request_id": "browser-request-0001",
  "conversation_id": "conv_可选",
  "model": "openrouter/free"
}
```

`client_request_id` 用作当前会话内的幂等键。主要 SSE 事件为 `accepted`、`phase`、`meta`、`delta`、`usage` 和 `done`；失败或取消时返回 `error` 或 `cancelled`。

设置接口支持的主要字段为 `api_key`、`chat_model`、`embedding_model`、`temperature`、`max_tokens`、`memory_enabled` 和 `allow_data_collection`。

## 安全与治理边界

- HTTP 服务默认只接受本机 Host，并拒绝跨站写请求。不要把它直接暴露到公网；如需远程访问，必须增加身份认证、TLS、权限隔离和速率限制。
- 技能执行器只接受声明式白名单操作；技能定义不能申请文件、网络或命令权限，也没有通用 `eval`、`exec` 或 Shell 执行入口。
- 技能版本不可变，控制核受保护，版本切换、生命周期变化、反馈和经验都可审计。
- 自动经验不会直接改写生产技能。任何从经验推导的进化结果都必须经过显式治理和人工批准，不能静默发布。
- 模型输出和历史记忆均视为不可信文本，不能证明外部操作已经发生，也不能提升本地权限。
- SQLite 适合单机单用户使用，不是多租户生产数据库。数据库包含聊天正文和经验，备份与访问控制应按敏感数据处理。

更完整的版本治理和安全设计见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 测试

```powershell
python -m unittest discover -s tests -v
```

如果 Python 没有加入 `PATH`，可以使用 Codex 附带的运行时：

```powershell
& "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m unittest discover -s tests -v
```

测试使用临时数据库和假的模型网关，不需要真实 API Key，也不会发起真实 OpenRouter 请求。

## 目录概览

```text
evoagent/
  api.py          本地 HTTP API、SSE 与静态页面服务
  chat.py         会话、流式生成、自动技能、标签和语义记忆
  openrouter.py   OpenRouter 聊天、模型发现和 embedding 客户端
  secrets.py      环境变量与 Windows DPAPI 密钥存储
  skills.py       声明式技能、不可变版本、匹配和治理
  memory.py       经验、评价和检索
  static/         ChatGPT 式中文 Web 界面
tests/            单元和端到端领域测试
```
