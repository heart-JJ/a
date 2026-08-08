const state = {
  conversations: [],
  activeConversationId: null,
  messagesByConversation: new Map(),
  draftsByConversation: new Map(),
  models: [],
  settings: {},
  selectedModel: "",
  activeRun: null,
  messageLoadController: null,
  messageLoadToken: 0,
  renameConversationId: null,
  deleteConversationId: null,
  sidebarOpen: false,
  messageNodes: new Map(),
  pendingPaintIds: new Set(),
  paintFrame: 0,
};

const dom = {
  sidebar: document.querySelector("#sidebar"),
  sidebarBackdrop: document.querySelector("#sidebarBackdrop"),
  sidebarCloseButton: document.querySelector("#sidebarCloseButton"),
  mobileMenuButton: document.querySelector("#mobileMenuButton"),
  newChatButton: document.querySelector("#newChatButton"),
  conversationList: document.querySelector("#conversationList"),
  conversationTitle: document.querySelector("#conversationTitle"),
  connectionStatus: document.querySelector("#connectionStatus"),
  modelSelect: document.querySelector("#modelSelect"),
  messageScroller: document.querySelector("#messageScroller"),
  messageList: document.querySelector("#messageList"),
  jumpLatestButton: document.querySelector("#jumpLatestButton"),
  composerForm: document.querySelector("#composerForm"),
  messageInput: document.querySelector("#messageInput"),
  sendButton: document.querySelector("#sendButton"),
  stopButton: document.querySelector("#stopButton"),
  composerHelp: document.querySelector("#composerHelp"),
  toast: document.querySelector("#toast"),
  streamAnnouncement: document.querySelector("#streamAnnouncement"),
  managementDialog: document.querySelector("#managementDialog"),
  managementTitle: document.querySelector("#managementTitle"),
  managementCloseButton: document.querySelector("#managementCloseButton"),
  skillsView: document.querySelector("#skillsView"),
  skillsList: document.querySelector("#skillsList"),
  createSkillButton: document.querySelector("#createSkillButton"),
  skillEditorForm: document.querySelector("#skillEditorForm"),
  skillEditorMode: document.querySelector("#skillEditorMode"),
  skillEditorId: document.querySelector("#skillEditorId"),
  skillEditorKicker: document.querySelector("#skillEditorKicker"),
  skillEditorTitle: document.querySelector("#skillEditorTitle"),
  skillEditorCloseButton: document.querySelector("#skillEditorCloseButton"),
  skillName: document.querySelector("#skillName"),
  skillDescription: document.querySelector("#skillDescription"),
  skillPrompt: document.querySelector("#skillPrompt"),
  skillTriggers: document.querySelector("#skillTriggers"),
  skillKind: document.querySelector("#skillKind"),
  skillActivate: document.querySelector("#skillActivate"),
  skillChangelog: document.querySelector("#skillChangelog"),
  skillSpec: document.querySelector("#skillSpec"),
  skillSaveButton: document.querySelector("#skillSaveButton"),
  memoriesView: document.querySelector("#memoriesView"),
  memoriesList: document.querySelector("#memoriesList"),
  refreshMemoriesButton: document.querySelector("#refreshMemoriesButton"),
  settingsView: document.querySelector("#settingsView"),
  settingsForm: document.querySelector("#settingsForm"),
  apiKey: document.querySelector("#apiKey"),
  apiKeyStatus: document.querySelector("#apiKeyStatus"),
  chatModelSelect: document.querySelector("#chatModelSelect"),
  embeddingModel: document.querySelector("#embeddingModel"),
  temperature: document.querySelector("#temperature"),
  memoryEnabled: document.querySelector("#memoryEnabled"),
  allowDataCollection: document.querySelector("#allowDataCollection"),
  settingsSaveButton: document.querySelector("#settingsSaveButton"),
  runtimeStatusDot: document.querySelector("#runtimeStatusDot"),
  runtimeStatusText: document.querySelector("#runtimeStatusText"),
  renameDialog: document.querySelector("#renameDialog"),
  renameForm: document.querySelector("#renameForm"),
  renameInput: document.querySelector("#renameInput"),
  deleteDialog: document.querySelector("#deleteDialog"),
  confirmDeleteButton: document.querySelector("#confirmDeleteButton"),
};

const DEFAULT_SKILL_SPEC = {
  schema_version: 1,
  executable: true,
  prompt: "请根据用户输入完成此技能。",
  triggers: { include: [], examples: [], exclude: [] },
  executor: { type: "pipeline", steps: [{ op: "normalize_text" }] },
  permissions: { filesystem: [], network: [], commands: [] },
};

function createElement(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function setButtonBusy(button, busy, busyText, idleText) {
  if (!button) return;
  button.disabled = busy;
  button.textContent = busy ? busyText : idleText;
}

function showToast(message, isError = false) {
  dom.toast.textContent = String(message || (isError ? "操作失败" : "操作完成"));
  dom.toast.className = `toast show${isError ? " error" : ""}`;
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => { dom.toast.className = "toast"; }, 3200);
}

function setConnectionStatus(text, offline = false) {
  dom.connectionStatus.textContent = text;
  dom.connectionStatus.classList.toggle("offline", offline);
  dom.runtimeStatusText.textContent = text;
  dom.runtimeStatusDot.classList.toggle("online", !offline);
  dom.runtimeStatusDot.classList.toggle("offline", offline);
}

async function readResponsePayload(response) {
  if (response.status === 204) return null;
  const text = await response.text();
  if (!text) return null;
  try { return JSON.parse(text); }
  catch (_) { return text; }
}

function errorMessage(payload, fallback) {
  if (typeof payload === "string" && payload.trim()) return payload;
  return payload?.detail || payload?.message || payload?.error?.message || payload?.error || fallback;
}

async function api(path, options = {}) {
  const request = { ...options };
  const headers = { Accept: "application/json", ...(request.headers || {}) };
  if (request.body !== undefined) {
    headers["Content-Type"] = "application/json";
    if (typeof request.body !== "string") request.body = JSON.stringify(request.body);
  }
  request.headers = headers;
  const response = await fetch(path, request);
  const payload = await readResponsePayload(response);
  if (!response.ok) throw new Error(errorMessage(payload, `HTTP ${response.status}`));
  return payload;
}

function payloadItems(payload, keys = []) {
  if (Array.isArray(payload)) return payload;
  for (const key of ["items", ...keys]) {
    if (Array.isArray(payload?.[key])) return payload[key];
  }
  if (Array.isArray(payload?.data)) return payload.data;
  return [];
}

function payloadObject(payload, keys = []) {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) return {};
  for (const key of keys) {
    if (payload[key] && typeof payload[key] === "object") return payload[key];
  }
  return payload;
}

function makeId(prefix) {
  if (globalThis.crypto?.randomUUID) return `${prefix}_${crypto.randomUUID()}`;
  return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2)}`;
}

function encodeId(id) { return encodeURIComponent(String(id)); }

function normalizeConversation(item) {
  const source = item?.conversation && typeof item.conversation === "object" ? item.conversation : (item || {});
  const id = source.id || source.conversation_id;
  if (!id) return null;
  return {
    ...source,
    id: String(id),
    title: String(source.title || source.name || "新对话"),
    model: String(source.model || source.model_id || ""),
    createdAt: source.created_at || source.createdAt || null,
    updatedAt: source.last_message_at || source.updated_at || source.updatedAt || source.created_at || null,
  };
}

function contentText(value) {
  if (typeof value === "string") return value;
  if (Array.isArray(value)) return value.map(contentText).filter(Boolean).join("");
  if (!value || typeof value !== "object") return "";
  if (typeof value.text === "string") return value.text;
  if (typeof value.content === "string") return value.content;
  if (value.content !== undefined) return contentText(value.content);
  if (value.output !== undefined) return contentText(value.output);
  if (typeof value.message === "string") return value.message;
  return "";
}

function normalizeMessage(item, conversationId) {
  const source = item?.message && typeof item.message === "object" ? item.message : (item || {});
  const role = source.role === "user" ? "user" : "assistant";
  return {
    ...source,
    id: String(source.id || source.message_id || makeId("msg")),
    conversationId: String(source.conversation_id || conversationId || ""),
    role,
    content: contentText(source.content ?? source.text ?? source.output),
    status: String(source.status || "done"),
    model: String(source.model || source.model_id || ""),
    usage: source.usage || null,
    metadata: source.metadata || source.meta || source.trace || null,
    error: source.error || null,
    experienceId: source.experience_id || null,
    createdAt: source.created_at || null,
  };
}

function normalizeModel(item) {
  const id = item?.id || item?.model_id || item?.name;
  if (!id) return null;
  return {
    ...item,
    id: String(id),
    label: String(item.label || item.display_name || item.name || id),
    available: item.available !== false && item.status !== "unavailable",
  };
}

function findConversation(id) { return state.conversations.find(item => item.id === id) || null; }

function upsertConversation(value, { placeFirst = false } = {}) {
  const conversation = normalizeConversation(value);
  if (!conversation) return null;
  const index = state.conversations.findIndex(item => item.id === conversation.id);
  if (index >= 0) {
    state.conversations[index] = { ...state.conversations[index], ...conversation };
    if (placeFirst && index > 0) {
      const [moved] = state.conversations.splice(index, 1);
      state.conversations.unshift(moved);
    }
  } else if (placeFirst) {
    state.conversations.unshift(conversation);
  } else {
    state.conversations.push(conversation);
  }
  return conversation;
}

function conversationGroupLabel(value) {
  if (!value) return "更早";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "更早";
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const target = new Date(date.getFullYear(), date.getMonth(), date.getDate());
  const days = Math.round((today - target) / 86400000);
  if (days <= 0) return "今天";
  if (days === 1) return "昨天";
  if (days < 7) return "最近 7 天";
  if (days < 30) return "最近 30 天";
  return "更早";
}

function renderConversationList() {
  dom.conversationList.replaceChildren();
  if (!state.conversations.length) {
    dom.conversationList.append(createElement("div", "conversation-empty", "还没有对话。点击“新建对话”开始。"));
    return;
  }

  const groups = new Map();
  for (const conversation of state.conversations) {
    const label = conversationGroupLabel(conversation.updatedAt);
    if (!groups.has(label)) groups.set(label, []);
    groups.get(label).push(conversation);
  }

  for (const [label, conversations] of groups) {
    dom.conversationList.append(createElement("h2", "conversation-group-title", label));
    for (const conversation of conversations) {
      const row = createElement("div", "conversation-row");
      row.classList.toggle("active", conversation.id === state.activeConversationId);

      const select = createElement("button", "conversation-select");
      select.type = "button";
      select.dataset.conversationId = conversation.id;
      select.setAttribute("aria-current", conversation.id === state.activeConversationId ? "page" : "false");
      const title = createElement("span", "conversation-title", conversation.title);
      select.title = conversation.title;
      select.append(title);
      if (state.activeRun?.conversationId === conversation.id) {
        const dot = createElement("span", "stream-dot");
        dot.setAttribute("aria-label", "正在生成");
        select.append(dot);
      }

      const actions = createElement("div", "conversation-actions");
      const rename = createElement("button", "conversation-action", "✎");
      rename.type = "button";
      rename.dataset.renameConversation = conversation.id;
      rename.setAttribute("aria-label", `重命名“${conversation.title}”`);
      const remove = createElement("button", "conversation-action danger", "×");
      remove.type = "button";
      remove.dataset.deleteConversation = conversation.id;
      remove.setAttribute("aria-label", `删除“${conversation.title}”`);
      actions.append(rename, remove);
      row.append(select, actions);
      dom.conversationList.append(row);
    }
  }
}

function messageStatusLabel(message) {
  const labels = {
    sending: "正在发送",
    accepted: "已接受",
    streaming: "正在生成",
    done: "已完成",
    error: "生成失败",
    cancelled: "已停止",
  };
  return labels[message.status] || message.status || "";
}

function usageLabel(usage) {
  if (!usage || typeof usage !== "object") return "";
  const input = usage.input_tokens ?? usage.prompt_tokens;
  const output = usage.output_tokens ?? usage.completion_tokens;
  const total = usage.total_tokens ?? ((Number(input) || 0) + (Number(output) || 0));
  const parts = [];
  if (input != null) parts.push(`输入 ${input}`);
  if (output != null) parts.push(`输出 ${output}`);
  if (!parts.length && total) parts.push(`共 ${total} tokens`);
  return parts.join(" · ");
}

function messageMetaLabel(message) {
  const parts = [];
  if (message.model) parts.push(message.model);
  const status = messageStatusLabel(message);
  if (status) parts.push(status);
  const usage = usageLabel(message.usage);
  if (usage) parts.push(usage);
  return parts.join(" · ");
}

function renderWelcome() {
  const root = createElement("div", "welcome");
  root.append(
    createElement("div", "welcome-mark", "E"),
    createElement("h2", "", "今天想做什么？"),
    createElement("p", "", "直接描述目标或粘贴需要处理的文本。EvoAgent 会选择已授权技能，并把执行过程记录为可追溯经验。"),
  );
  dom.messageList.append(root);
}

function createTraceBlock(metadata) {
  const details = createElement("details", "trace-block");
  details.append(createElement("summary", "", "查看技能与执行信息"));
  const pre = createElement("pre");
  try { pre.textContent = JSON.stringify(metadata, null, 2); }
  catch (_) { pre.textContent = String(metadata); }
  details.append(pre);
  return details;
}

function createMessageNode(message) {
  const article = createElement("article", `message ${message.role}`);
  article.dataset.messageId = message.id;
  article.setAttribute("aria-label", message.role === "user" ? "你的消息" : "EvoAgent 的回复");
  if (message.role === "assistant") article.append(createElement("div", "message-avatar", "E"));

  const body = createElement("div", "message-body");
  const content = createElement("p", "message-content", message.content);
  const meta = createElement("div", "message-meta", messageMetaLabel(message));
  const error = createElement("div", "message-error");
  const traceHost = createElement("div", "message-trace-host");
  body.append(content, meta, error, traceHost);
  article.append(body);
  state.messageNodes.set(message.id, { article, content, meta, error, traceHost, traceSignature: "" });
  updateMessageNode(message);
  return article;
}

function updateMessageNode(message) {
  const refs = state.messageNodes.get(message.id);
  if (!refs) return;
  refs.article.classList.toggle("streaming", ["sending", "accepted", "streaming"].includes(message.status));
  refs.article.classList.toggle("error", message.status === "error");
  refs.content.textContent = message.content || "";
  refs.meta.textContent = messageMetaLabel(message);

  const errorText = typeof message.error === "string" ? message.error : message.error?.message;
  refs.error.textContent = errorText || "";
  refs.error.classList.toggle("hidden", !errorText);

  let signature = "";
  if (message.metadata) {
    try { signature = JSON.stringify(message.metadata); }
    catch (_) { signature = String(message.metadata); }
  }
  if (signature !== refs.traceSignature) {
    refs.traceHost.replaceChildren();
    if (message.metadata) refs.traceHost.append(createTraceBlock(message.metadata));
    refs.traceSignature = signature;
  }
}

function renderMessages({ scrollToEnd = false } = {}) {
  state.messageNodes.clear();
  dom.messageList.replaceChildren();
  const messages = state.messagesByConversation.get(state.activeConversationId) || [];
  if (!messages.length) renderWelcome();
  else messages.forEach(message => dom.messageList.append(createMessageNode(message)));
  if (scrollToEnd) requestAnimationFrame(() => scrollToBottom(true));
}

function isNearBottom() {
  const remaining = dom.messageScroller.scrollHeight - dom.messageScroller.scrollTop - dom.messageScroller.clientHeight;
  return remaining < 110;
}

function scrollToBottom(force = false) {
  if (!force && !isNearBottom()) return;
  dom.messageScroller.scrollTop = dom.messageScroller.scrollHeight;
  dom.jumpLatestButton.classList.add("hidden");
}

function queueMessagePaint(message) {
  state.pendingPaintIds.add(message.id);
  if (state.paintFrame) return;
  const shouldStick = isNearBottom();
  state.paintFrame = requestAnimationFrame(() => {
    for (const id of state.pendingPaintIds) {
      const currentMessages = state.messagesByConversation.get(state.activeConversationId) || [];
      const current = currentMessages.find(item => item.id === id);
      if (current) updateMessageNode(current);
    }
    state.pendingPaintIds.clear();
    state.paintFrame = 0;
    if (shouldStick) scrollToBottom(true);
    else dom.jumpLatestButton.classList.remove("hidden");
  });
}

function currentMessages(conversationId) {
  if (!state.messagesByConversation.has(conversationId)) state.messagesByConversation.set(conversationId, []);
  return state.messagesByConversation.get(conversationId);
}

function replaceMessageId(conversationId, oldId, newId) {
  if (!newId || oldId === newId) return oldId;
  const messages = currentMessages(conversationId);
  const message = messages.find(item => item.id === oldId);
  if (!message) return oldId;
  const duplicate = messages.find(item => item.id === newId);
  if (duplicate && duplicate !== message) {
    Object.assign(duplicate, message, { id: newId });
    const index = messages.indexOf(message);
    if (index >= 0) messages.splice(index, 1);
  } else {
    message.id = String(newId);
  }
  if (state.activeConversationId === conversationId) renderMessages();
  return String(newId);
}

function updateHeader() {
  const conversation = findConversation(state.activeConversationId);
  dom.conversationTitle.textContent = conversation?.title || "新对话";
  if (conversation?.model) state.selectedModel = conversation.model;
  dom.modelSelect.value = state.selectedModel || "";
  syncComposerState();
}

function saveCurrentDraft() {
  const key = state.activeConversationId || "__new__";
  state.draftsByConversation.set(key, dom.messageInput.value);
}

function restoreDraft() {
  const key = state.activeConversationId || "__new__";
  dom.messageInput.value = state.draftsByConversation.get(key) || "";
  resizeComposer();
}

function updateRoute(conversationId, replace = false) {
  const path = conversationId ? `/c/${encodeId(conversationId)}` : "/";
  if (location.pathname === path) return;
  history[replace ? "replaceState" : "pushState"]({ conversationId }, "", path);
}

function routeConversationId() {
  const match = location.pathname.match(/^\/c\/([^/]+)\/?$/);
  if (!match) return null;
  try { return decodeURIComponent(match[1]); }
  catch (_) { return match[1]; }
}

async function loadConversations() {
  const payload = await api("/api/conversations");
  state.conversations = payloadItems(payload, ["conversations"])
    .map(normalizeConversation)
    .filter(Boolean);
  renderConversationList();
}

async function loadConversationRecord(conversationId) {
  const payload = await api(`/api/conversations/${encodeId(conversationId)}`);
  return upsertConversation(payloadObject(payload, ["conversation"]));
}

function mergeMessages(serverMessages, localMessages, conversationId) {
  const result = [];
  const byId = new Map();
  for (const message of serverMessages) {
    byId.set(message.id, message);
    result.push(message);
  }
  for (const message of localMessages) {
    const server = byId.get(message.id);
    if (server) Object.assign(server, ["sending", "accepted", "streaming"].includes(message.status) ? message : server);
    else if (state.activeRun?.conversationId === conversationId || ["sending", "accepted", "streaming"].includes(message.status)) result.push(message);
  }
  return result;
}

async function loadMessages(conversationId) {
  state.messageLoadController?.abort();
  const controller = new AbortController();
  state.messageLoadController = controller;
  const token = ++state.messageLoadToken;
  dom.messageList.setAttribute("aria-busy", "true");
  try {
    const payload = await api(`/api/conversations/${encodeId(conversationId)}/messages`, { signal: controller.signal });
    if (token !== state.messageLoadToken || state.activeConversationId !== conversationId) return;
    const serverMessages = payloadItems(payload, ["messages"])
      .map(item => normalizeMessage(item, conversationId));
    const local = state.messagesByConversation.get(conversationId) || [];
    state.messagesByConversation.set(conversationId, mergeMessages(serverMessages, local, conversationId));
    renderMessages({ scrollToEnd: true });
  } catch (error) {
    if (error.name !== "AbortError") {
      showToast(error.message, true);
      if (!(state.messagesByConversation.get(conversationId) || []).length) renderMessages();
    }
  } finally {
    if (token === state.messageLoadToken) dom.messageList.setAttribute("aria-busy", "false");
  }
}

async function selectConversation(conversationId, { push = true, focusComposer = false } = {}) {
  if (!conversationId) return;
  saveCurrentDraft();
  let conversation = findConversation(conversationId);
  if (!conversation) {
    try { conversation = await loadConversationRecord(conversationId); }
    catch (error) { showToast(error.message, true); return; }
  }
  state.activeConversationId = conversationId;
  if (conversation?.model) state.selectedModel = conversation.model;
  renderConversationList();
  updateHeader();
  restoreDraft();
  renderMessages();
  if (push) updateRoute(conversationId);
  closeSidebar({ restoreFocus: mobileMedia.matches && !focusComposer });
  if (focusComposer) dom.messageInput.focus();
  await loadMessages(conversationId);
  if (focusComposer) dom.messageInput.focus();
}

async function createConversation({ focusComposer = true } = {}) {
  if (state.activeConversationId && !(state.messagesByConversation.get(state.activeConversationId) || []).length) {
    if (focusComposer) dom.messageInput.focus();
    return findConversation(state.activeConversationId);
  }
  dom.newChatButton.disabled = true;
  dom.newChatButton.setAttribute("aria-busy", "true");
  try {
    const body = state.selectedModel ? { model: state.selectedModel } : {};
    const payload = await api("/api/conversations", { method: "POST", body });
    const conversation = upsertConversation(payloadObject(payload, ["conversation"]), { placeFirst: true });
    if (!conversation) throw new Error("服务未返回会话 ID");
    state.messagesByConversation.set(conversation.id, []);
    await selectConversation(conversation.id, { focusComposer });
    return conversation;
  } catch (error) {
    showToast(error.message, true);
    return null;
  } finally {
    dom.newChatButton.disabled = false;
    dom.newChatButton.removeAttribute("aria-busy");
  }
}

function openRenameDialog(conversationId) {
  const conversation = findConversation(conversationId);
  if (!conversation) return;
  state.renameConversationId = conversationId;
  dom.renameInput.value = conversation.title;
  dom.renameDialog.showModal();
  requestAnimationFrame(() => { dom.renameInput.focus(); dom.renameInput.select(); });
}

async function renameConversation(event) {
  event.preventDefault();
  const id = state.renameConversationId;
  const title = dom.renameInput.value.trim();
  if (!id || !title) return;
  const submit = dom.renameForm.querySelector('[type="submit"]');
  setButtonBusy(submit, true, "保存中…", "保存");
  try {
    const payload = await api(`/api/conversations/${encodeId(id)}`, { method: "PATCH", body: { title } });
    const returned = normalizeConversation(payloadObject(payload, ["conversation"]));
    const conversation = findConversation(id);
    if (conversation) Object.assign(conversation, returned || {}, { title: returned?.title || title });
    renderConversationList();
    updateHeader();
    dom.renameDialog.close();
  } catch (error) { showToast(error.message, true); }
  finally { setButtonBusy(submit, false, "保存中…", "保存"); }
}

function openDeleteDialog(conversationId) {
  if (state.activeRun?.conversationId === conversationId) {
    showToast("请先停止该对话的生成任务", true);
    return;
  }
  state.deleteConversationId = conversationId;
  dom.deleteDialog.showModal();
}

async function deleteConversation() {
  const id = state.deleteConversationId;
  if (!id) return;
  setButtonBusy(dom.confirmDeleteButton, true, "删除中…", "删除");
  try {
    await api(`/api/conversations/${encodeId(id)}`, { method: "DELETE" });
    const index = state.conversations.findIndex(item => item.id === id);
    if (index >= 0) state.conversations.splice(index, 1);
    state.messagesByConversation.delete(id);
    state.draftsByConversation.delete(id);
    dom.deleteDialog.close();
    if (state.activeConversationId === id) {
      state.activeConversationId = null;
      const next = state.conversations[Math.min(index, state.conversations.length - 1)] || state.conversations[0];
      if (next) await selectConversation(next.id, { push: false });
      else {
        updateRoute(null);
        updateHeader();
        restoreDraft();
        renderMessages();
      }
    }
    renderConversationList();
  } catch (error) { showToast(error.message, true); }
  finally { setButtonBusy(dom.confirmDeleteButton, false, "删除中…", "删除"); }
}

function resizeComposer() {
  dom.messageInput.style.height = "auto";
  dom.messageInput.style.height = `${Math.min(dom.messageInput.scrollHeight, 190)}px`;
  syncComposerState();
}

function syncComposerState() {
  const running = Boolean(state.activeRun);
  dom.sendButton.classList.toggle("hidden", running);
  dom.stopButton.classList.toggle("hidden", !running);
  dom.stopButton.disabled = Boolean(state.activeRun?.cancelSent);
  dom.modelSelect.disabled = running;
  dom.sendButton.disabled = !dom.messageInput.value.trim();
  dom.composerHelp.textContent = running
    ? (state.activeRun?.cancelRequested ? "正在停止生成…" : "当前有一条回复正在生成，可随时停止。")
    : "Enter 发送，Shift + Enter 换行。系统只执行已授权的本地技能。";
}

function parseSseData(raw) {
  if (!raw) return {};
  try { return JSON.parse(raw); }
  catch (_) { return raw; }
}

async function consumeEventStream(response, onEvent) {
  if (!response.body?.getReader) throw new Error("当前浏览器不支持流式响应");
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let eventName = "";
  let eventId = "";
  let dataLines = [];

  const dispatch = async () => {
    if (!eventName && !dataLines.length && !eventId) return;
    const event = {
      type: eventName || "message",
      id: eventId,
      data: parseSseData(dataLines.join("\n")),
    };
    eventName = "";
    eventId = "";
    dataLines = [];
    await onEvent(event);
  };

  const processLine = async rawLine => {
    const line = rawLine.endsWith("\r") ? rawLine.slice(0, -1) : rawLine;
    if (line === "") { await dispatch(); return; }
    if (line.startsWith(":")) return;
    const separator = line.indexOf(":");
    const field = separator < 0 ? line : line.slice(0, separator);
    let value = separator < 0 ? "" : line.slice(separator + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") eventName = value;
    else if (field === "data") dataLines.push(value);
    else if (field === "id" && !value.includes("\0")) eventId = value;
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let newline;
    while ((newline = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, newline);
      buffer = buffer.slice(newline + 1);
      await processLine(line);
    }
  }

  buffer += decoder.decode();
  if (buffer) await processLine(buffer);
  await dispatch();
}

function streamEventType(event) {
  if (event.type !== "message") return event.type;
  if (event.data && typeof event.data === "object" && event.data.type) return String(event.data.type);
  return "message";
}

function eventText(data, keys = ["delta", "text", "content"]) {
  if (typeof data === "string") return data;
  for (const key of keys) {
    if (typeof data?.[key] === "string") return data[key];
  }
  return "";
}

function runAssistantMessage(run) {
  return currentMessages(run.conversationId).find(item => item.id === run.assistantMessageId) || null;
}

function moveRunConversation(run, nextConversationId) {
  if (!nextConversationId || nextConversationId === run.conversationId) return;
  const previousId = run.conversationId;
  const messages = state.messagesByConversation.get(previousId) || [];
  const target = currentMessages(nextConversationId);
  for (const message of messages) {
    message.conversationId = nextConversationId;
    if (!target.some(item => item.id === message.id)) target.push(message);
  }
  state.messagesByConversation.delete(previousId);
  run.conversationId = nextConversationId;
  if (state.activeConversationId === previousId) {
    state.activeConversationId = nextConversationId;
    updateRoute(nextConversationId, true);
  }
}

function finishRun(run, status, announcement) {
  window.clearTimeout(run.cancelFallbackTimer);
  const assistant = runAssistantMessage(run);
  const user = currentMessages(run.conversationId).find(item => item.id === run.userMessageId);
  if (user && ["sending", "accepted"].includes(user.status)) user.status = "done";
  if (assistant && ["sending", "accepted", "streaming"].includes(assistant.status)) assistant.status = status;
  if (assistant) queueMessagePaint(assistant);
  if (user) queueMessagePaint(user);
  if (state.activeRun === run) state.activeRun = null;
  dom.messageList.setAttribute("aria-busy", "false");
  setConnectionStatus("本地服务正常", false);
  dom.streamAnnouncement.textContent = announcement;
  syncComposerState();
  renderConversationList();
}

async function cancelRunOnServer(run) {
  if (run.cancelSent || !run.runId) return;
  run.cancelSent = true;
  syncComposerState();
  try {
    await api(`/api/chat/runs/${encodeId(run.runId)}/cancel`, {
      method: "POST",
      body: { client_request_id: run.requestId },
    });
    run.controller.abort();
    finishRun(run, "cancelled", "回复已停止");
  } catch (error) {
    run.cancelSent = false;
    run.cancelRequested = false;
    syncComposerState();
    showToast(`停止失败：${error.message}`, true);
  }
}

async function handleStreamEvent(run, event) {
  const type = streamEventType(event);
  const data = event.data && typeof event.data === "object" ? event.data : event.data;
  const eventConversationId = data?.conversation_id || data?.conversation?.id;
  if (eventConversationId) moveRunConversation(run, String(eventConversationId));

  if (type === "accepted") {
    run.runId = String(data?.run_id || data?.id || run.runId || "");
    if (data?.conversation) upsertConversation(data.conversation, { placeFirst: true });
    else if (eventConversationId && !findConversation(String(eventConversationId))) {
      upsertConversation({ id: eventConversationId, title: data?.title || "新对话" }, { placeFirst: true });
    }
    if (data?.user_message_id) run.userMessageId = replaceMessageId(run.conversationId, run.userMessageId, String(data.user_message_id));
    if (data?.assistant_message_id || data?.message_id) {
      run.assistantMessageId = replaceMessageId(
        run.conversationId,
        run.assistantMessageId,
        String(data.assistant_message_id || data.message_id),
      );
    }
    const assistant = runAssistantMessage(run);
    const user = currentMessages(run.conversationId).find(item => item.id === run.userMessageId);
    if (user) { user.status = "done"; queueMessagePaint(user); }
    if (assistant) { assistant.status = "accepted"; queueMessagePaint(assistant); }
    renderConversationList();
    updateHeader();
    if (run.cancelRequested) await cancelRunOnServer(run);
    return;
  }

  const eventRunId = data?.run_id;
  if (eventRunId && run.runId && String(eventRunId) !== run.runId) return;
  if (data?.conversation_id && String(data.conversation_id) !== run.conversationId) return;
  const assistant = runAssistantMessage(run);
  if (!assistant) return;

  if (type === "meta") {
    assistant.status = "streaming";
    assistant.model = String(data?.model || data?.model_id || assistant.model || "");
    assistant.metadata = data?.metadata || data?.meta || data?.selection || data;
    if (data?.title) {
      const conversation = findConversation(run.conversationId);
      if (conversation) conversation.title = String(data.title);
      renderConversationList();
      updateHeader();
    }
    queueMessagePaint(assistant);
    return;
  }

  if (type === "delta" || type === "message") {
    const delta = eventText(data);
    if (delta) {
      assistant.status = "streaming";
      assistant.content += delta;
      queueMessagePaint(assistant);
    }
    return;
  }

  if (type === "usage") {
    assistant.usage = data?.usage || data;
    queueMessagePaint(assistant);
    return;
  }

  if (type === "done") {
    const finalText = eventText(data, ["content", "text", "output"]);
    if (finalText && (!assistant.content || data?.authoritative === true)) assistant.content = finalText;
    assistant.model = String(data?.model || data?.model_id || assistant.model || "");
    assistant.usage = data?.usage || assistant.usage;
    assistant.experienceId = data?.experience_id || assistant.experienceId;
    assistant.status = "done";
    if (data?.conversation) upsertConversation(data.conversation, { placeFirst: true });
    else {
      const conversation = findConversation(run.conversationId);
      if (conversation && data?.title) conversation.title = String(data.title);
      if (conversation) conversation.updatedAt = new Date().toISOString();
    }
    queueMessagePaint(assistant);
    finishRun(run, "done", "回复已完成");
    void refreshConversationsAfterRun(run.conversationId);
    return;
  }

  if (type === "error") {
    assistant.status = "error";
    assistant.error = errorMessage(data, "生成失败");
    queueMessagePaint(assistant);
    finishRun(run, "error", "回复生成失败");
    return;
  }

  if (type === "cancelled") {
    assistant.status = "cancelled";
    queueMessagePaint(assistant);
    finishRun(run, "cancelled", "回复已停止");
  }
}

async function refreshConversationsAfterRun(conversationId) {
  try {
    const payload = await api("/api/conversations");
    const fresh = payloadItems(payload, ["conversations"]).map(normalizeConversation).filter(Boolean);
    if (fresh.length) state.conversations = fresh;
    renderConversationList();
    if (state.activeConversationId === conversationId) updateHeader();
  } catch (_) {
    renderConversationList();
  }
}

async function sendMessage() {
  if (state.activeRun) { showToast("请先停止或等待当前回复完成", true); return; }
  const text = dom.messageInput.value.trim();
  if (!text) return;

  let conversation = findConversation(state.activeConversationId);
  if (!conversation) conversation = await createConversation({ focusComposer: false });
  if (!conversation) return;

  const requestId = makeId("request");
  const conversationId = conversation.id;
  const userMessage = {
    id: makeId("local_user"), conversationId, role: "user", content: text, status: "sending", model: "",
  };
  const assistantMessage = {
    id: makeId("local_assistant"), conversationId, role: "assistant", content: "", status: "sending",
    model: state.selectedModel,
  };
  currentMessages(conversationId).push(userMessage, assistantMessage);
  const controller = new AbortController();
  const run = {
    requestId,
    runId: null,
    conversationId,
    userMessageId: userMessage.id,
    assistantMessageId: assistantMessage.id,
    controller,
    cancelRequested: false,
    cancelSent: false,
    cancelFallbackTimer: null,
  };
  state.activeRun = run;
  state.draftsByConversation.set(conversationId, "");
  dom.messageInput.value = "";
  resizeComposer();
  renderMessages({ scrollToEnd: true });
  renderConversationList();
  dom.messageList.setAttribute("aria-busy", "true");
  setConnectionStatus("正在生成回复", false);
  syncComposerState();

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({
        conversation_id: conversationId,
        message: text,
        client_request_id: requestId,
        ...(state.selectedModel ? { model: state.selectedModel } : {}),
      }),
      signal: controller.signal,
    });
    if (!response.ok) {
      const payload = await readResponsePayload(response);
      throw new Error(errorMessage(payload, `HTTP ${response.status}`));
    }
    const contentType = response.headers.get("Content-Type") || "";
    if (!contentType.toLowerCase().includes("text/event-stream")) {
      throw new Error("聊天接口未返回 text/event-stream");
    }
    await consumeEventStream(response, event => handleStreamEvent(run, event));
    if (state.activeRun === run) {
      const assistant = runAssistantMessage(run);
      if (assistant?.content) finishRun(run, "done", "回复已完成");
      else throw new Error("流式连接提前结束，未收到完成事件");
    }
  } catch (error) {
    if (error.name === "AbortError" && run.cancelRequested) {
      if (state.activeRun === run) finishRun(run, "cancelled", "回复已停止");
      return;
    }
    const assistant = runAssistantMessage(run);
    if (assistant) {
      assistant.status = "error";
      assistant.error = error.message;
      queueMessagePaint(assistant);
    }
    if (state.activeRun === run) finishRun(run, "error", "回复生成失败");
    showToast(error.message, true);
  }
}

function requestStop() {
  const run = state.activeRun;
  if (!run || run.cancelRequested) return;
  run.cancelRequested = true;
  syncComposerState();
  if (run.runId) void cancelRunOnServer(run);
  else {
    showToast("已请求停止，正在等待服务确认运行编号");
    run.cancelFallbackTimer = window.setTimeout(() => {
      if (state.activeRun !== run || run.runId) return;
      run.controller.abort();
      finishRun(run, "cancelled", "回复已停止");
    }, 5000);
  }
}

async function loadModels() {
  try {
    const payload = await api("/api/models");
    state.models = payloadItems(payload, ["models"]).map(normalizeModel).filter(Boolean);
  } catch (error) {
    state.models = [];
    showToast(`模型列表加载失败：${error.message}`, true);
  }
  populateModelSelects();
}

function populateModelSelect(select, value, { allowEmpty = true } = {}) {
  select.replaceChildren();
  if (allowEmpty) {
    const automatic = createElement("option", "", "自动选择");
    automatic.value = "";
    select.append(automatic);
  }
  let hasSelectedValue = false;
  for (const model of state.models) {
    const freeProvider = model.is_free === true || model.id.includes(":free");
    const option = createElement("option", "", `${model.label}${freeProvider ? " · 需数据处理许可" : ""}`);
    option.value = model.id;
    option.disabled = !model.available;
    select.append(option);
    if (model.id === value) hasSelectedValue = true;
  }
  if (value && !hasSelectedValue) {
    const current = createElement("option", "", value);
    current.value = value;
    select.append(current);
  }
  select.value = value || (allowEmpty ? "" : "openrouter/free");
}

function populateModelSelects() {
  populateModelSelect(dom.modelSelect, state.selectedModel || state.settings.chat_model || "openrouter/free", { allowEmpty: false });
  populateModelSelect(dom.chatModelSelect, state.settings.chat_model || "openrouter/free", { allowEmpty: false });
}

async function changeConversationModel() {
  const previous = state.selectedModel || state.settings.chat_model || "openrouter/free";
  const next = dom.modelSelect.value || state.settings.chat_model || "openrouter/free";
  if (next.includes(":free") && state.settings.allow_data_collection !== true) {
    showToast("该免费模型可能需要提供商处理对话数据；当前隐私许可为关闭，可在设置中明确开启。", true);
  }
  state.selectedModel = next;
  const conversation = findConversation(state.activeConversationId);
  if (!conversation) return;
  conversation.model = next;
  try {
    const payload = await api(`/api/conversations/${encodeId(conversation.id)}`, {
      method: "PATCH",
      body: { model: next || null },
    });
    const returned = normalizeConversation(payloadObject(payload, ["conversation"]));
    if (returned) Object.assign(conversation, returned);
  } catch (error) {
    state.selectedModel = previous;
    conversation.model = previous;
    dom.modelSelect.value = previous;
    showToast(error.message, true);
  }
}

async function loadHealth() {
  try {
    await api("/api/health");
    setConnectionStatus("本地服务正常", false);
  } catch (_) { setConnectionStatus("本地服务未连接", true); }
}

function managementTitle(view) {
  return { skills: "技能库", memories: "记忆 / 经验", settings: "设置" }[view] || "管理";
}

async function openManagement(view) {
  dom.managementTitle.textContent = managementTitle(view);
  for (const section of document.querySelectorAll("[data-drawer-view]")) {
    section.classList.toggle("hidden", section.dataset.drawerView !== view);
  }
  closeSkillEditor();
  if (!dom.managementDialog.open) dom.managementDialog.showModal();
  closeSidebar();
  if (view === "skills") await loadSkills();
  if (view === "memories") await loadMemories();
  if (view === "settings") await loadSettings();
}

function emptyList(container, text) {
  container.replaceChildren(createElement("div", "empty-card", text));
}

async function loadSkills() {
  emptyList(dom.skillsList, "正在加载技能…");
  try {
    const payload = await api("/api/skills");
    renderSkills(payloadItems(payload, ["skills"]));
  } catch (error) {
    emptyList(dom.skillsList, "技能加载失败");
    showToast(error.message, true);
  }
}

function renderSkills(skills) {
  dom.skillsList.replaceChildren();
  if (!skills.length) { emptyList(dom.skillsList, "还没有技能"); return; }
  for (const skill of skills) {
    const card = createElement("article", "list-card");
    const head = createElement("div", "list-card-head");
    const titleWrap = createElement("div");
    titleWrap.append(
      createElement("h4", "", skill.name || skill.slug || skill.id || "未命名技能"),
      createElement("p", "", skill.description || "暂无描述"),
    );
    const lifecycle = createElement("span", "connection-status", skill.lifecycle || skill.status || "");
    head.append(titleWrap, lifecycle);
    const meta = createElement("div", "list-meta");
    meta.append(
      createElement("span", "", `类型：${skill.kind || "—"}`),
      createElement("span", "", `当前版本：v${skill.active_version ?? skill.version ?? "—"}`),
      createElement("span", "", `最新版本：v${skill.latest_version ?? skill.version ?? "—"}`),
    );
    if (skill.protected) meta.append(createElement("span", "", "受保护"));
    const actions = createElement("div", "list-card-actions");
    const versionButton = createElement("button", "secondary-button", "新建版本");
    versionButton.type = "button";
    if (skill.protected) {
      versionButton.disabled = true;
      versionButton.title = "系统控制核受保护，不能人工创建版本";
    } else {
      versionButton.addEventListener("click", () => void loadSkillVersionEditor(skill, versionButton));
    }
    actions.append(versionButton);
    card.append(head, meta, actions);
    dom.skillsList.append(card);
  }
}

function skillSpecFromRecord(skill) {
  const direct = skill?.spec || skill?.active_spec || skill?.definition;
  if (direct && typeof direct === "object") return direct;
  const versions = Array.isArray(skill?.versions) ? skill.versions : [];
  const activeVersion = versions.find(item => String(item.version) === String(skill?.active_version));
  const candidate = activeVersion || versions[0];
  return candidate?.spec && typeof candidate.spec === "object" ? candidate.spec : null;
}

async function loadSkillVersionEditor(skill, button) {
  const id = skill?.id || skill?.skill_id;
  if (!id) { showToast("缺少技能 ID", true); return; }
  setButtonBusy(button, true, "加载中…", "新建版本");
  try {
    const payload = await api(`/api/skills/${encodeId(id)}`);
    const detail = payloadObject(payload, ["skill"]);
    const spec = skillSpecFromRecord(detail);
    if (!spec) throw new Error("未找到当前活跃版本的技能定义");
    openSkillEditor("version", { ...skill, ...detail, editorSpec: spec });
  } catch (error) {
    showToast(`技能详情加载失败：${error.message}`, true);
  } finally {
    setButtonBusy(button, false, "加载中…", "新建版本");
  }
}

function parseTriggerInput(value) {
  return [...new Set(String(value || "").split(/[,，\n]/).map(item => item.trim()).filter(Boolean))];
}

function buildSkillSpecFromForm() {
  let spec;
  try { spec = JSON.parse(dom.skillSpec.value); }
  catch (error) { throw new Error(`技能定义不是有效 JSON：${error.message}`); }
  if (!spec || typeof spec !== "object" || Array.isArray(spec)) throw new Error("技能定义必须是 JSON 对象");
  const prompt = dom.skillPrompt.value.trim();
  if (!prompt) throw new Error("请填写技能提示词");
  const existingTriggers = spec.triggers && typeof spec.triggers === "object" ? spec.triggers : {};
  spec.schema_version = 1;
  spec.executable = true;
  spec.prompt = prompt;
  spec.triggers = {
    ...existingTriggers,
    include: parseTriggerInput(dom.skillTriggers.value),
    examples: Array.isArray(existingTriggers.examples) ? existingTriggers.examples : [],
    exclude: Array.isArray(existingTriggers.exclude) ? existingTriggers.exclude : [],
  };
  if (!spec.executor || spec.executor.type !== "pipeline" || !Array.isArray(spec.executor.steps) || !spec.executor.steps.length) {
    spec.executor = { type: "pipeline", steps: [{ op: "normalize_text" }] };
  }
  spec.permissions = { filesystem: [], network: [], commands: [] };
  dom.skillSpec.value = JSON.stringify(spec, null, 2);
  return spec;
}

function openSkillEditor(mode, skill = null) {
  dom.skillEditorForm.reset();
  dom.skillEditorMode.value = mode;
  dom.skillEditorId.value = skill?.id || skill?.skill_id || "";
  dom.skillEditorForm.classList.remove("hidden");
  if (mode === "version") {
    const spec = skill?.editorSpec || skillSpecFromRecord(skill);
    if (!spec) { closeSkillEditor(); showToast("未找到当前活跃版本的技能定义", true); return; }
    dom.skillEditorKicker.textContent = "NEW VERSION";
    dom.skillEditorTitle.textContent = `为“${skill?.name || skill?.id || "技能"}”创建新版本`;
    dom.skillName.value = skill?.name || "";
    dom.skillName.disabled = true;
    dom.skillDescription.value = skill?.description || "";
    dom.skillDescription.disabled = true;
    dom.skillKind.value = skill?.kind || "atomic";
    dom.skillKind.disabled = true;
    dom.skillPrompt.value = String(spec.prompt || skill?.description || "");
    dom.skillTriggers.value = Array.isArray(spec.triggers?.include) ? spec.triggers.include.join("，") : "";
    dom.skillActivate.checked = true;
    dom.skillChangelog.value = "通过聊天控制台创建新版本";
    dom.skillSpec.value = JSON.stringify(spec, null, 2);
    dom.skillSaveButton.textContent = "保存新版本";
  } else {
    dom.skillEditorKicker.textContent = "NEW SKILL";
    dom.skillEditorTitle.textContent = "新建技能";
    dom.skillName.disabled = false;
    dom.skillDescription.disabled = false;
    dom.skillKind.disabled = false;
    dom.skillKind.value = "atomic";
    dom.skillPrompt.value = DEFAULT_SKILL_SPEC.prompt;
    dom.skillTriggers.value = "";
    dom.skillActivate.checked = true;
    dom.skillChangelog.value = "通过聊天控制台创建";
    dom.skillSpec.value = JSON.stringify(DEFAULT_SKILL_SPEC, null, 2);
    dom.skillSaveButton.textContent = "保存技能";
  }
  dom.skillEditorForm.scrollIntoView({ behavior: "smooth", block: "start" });
  requestAnimationFrame(() => (mode === "version" ? dom.skillPrompt : dom.skillName).focus());
}

function closeSkillEditor() {
  dom.skillEditorForm.classList.add("hidden");
  dom.skillName.disabled = false;
  dom.skillDescription.disabled = false;
  dom.skillKind.disabled = false;
}

async function saveSkill(event) {
  event.preventDefault();
  let spec;
  try { spec = buildSkillSpecFromForm(); }
  catch (error) { showToast(error.message, true); return; }
  const mode = dom.skillEditorMode.value;
  const id = dom.skillEditorId.value;
  setButtonBusy(dom.skillSaveButton, true, "保存中…", mode === "version" ? "保存新版本" : "保存技能");
  try {
    if (mode === "version") {
      if (!id) throw new Error("缺少技能 ID");
      await api(`/api/skills/${encodeId(id)}/versions`, {
        method: "POST",
        body: { spec, changelog: dom.skillChangelog.value.trim(), activate: dom.skillActivate.checked },
      });
      showToast("技能新版本已创建");
    } else {
      await api("/api/skills", {
        method: "POST",
        body: {
          name: dom.skillName.value.trim(),
          description: dom.skillDescription.value.trim(),
          kind: dom.skillKind.value,
          spec,
          changelog: dom.skillChangelog.value.trim(),
          activate: dom.skillActivate.checked,
        },
      });
      showToast("技能已创建");
    }
    closeSkillEditor();
    await loadSkills();
  } catch (error) { showToast(error.message, true); }
  finally { setButtonBusy(dom.skillSaveButton, false, "保存中…", mode === "version" ? "保存新版本" : "保存技能"); }
}

async function loadMemories() {
  emptyList(dom.memoriesList, "正在加载记忆…");
  try {
    const payload = await api("/api/memories");
    const memories = Array.isArray(payload) ? payload : (Array.isArray(payload?.items) ? payload.items : []);
    const experiences = Array.isArray(payload?.experiences) ? payload.experiences : [];
    renderMemories(memories, experiences);
  } catch (error) {
    emptyList(dom.memoriesList, "记忆加载失败");
    showToast(error.message, true);
  }
}

function appendMemoryCards(records, kind) {
  for (const memory of records) {
    const card = createElement("article", "list-card");
    const title = kind === "memory"
      ? (memory.title || "对话记忆")
      : (memory.task || memory.title || memory.name || "经验记录");
    let content;
    if (kind === "memory") {
      const userText = contentText(memory.user);
      const assistantText = contentText(memory.assistant ?? memory.content);
      content = [userText && `用户：${userText}`, assistantText && `助手：${assistantText}`].filter(Boolean).join("\n\n");
    } else {
      content = contentText(memory.summary ?? memory.content ?? memory.text ?? memory.input_text ?? memory.output);
    }
    card.append(createElement("h4", "", title), createElement("p", "", content || "暂无摘要"));
    const meta = createElement("div", "list-meta");
    const created = memory.created_at ? new Date(memory.created_at).toLocaleString("zh-CN") : "";
    if (kind === "memory" && memory.model) meta.append(createElement("span", "", `模型：${memory.model}`));
    if (kind === "memory" && memory.embedding_status) meta.append(createElement("span", "", `向量状态：${memory.embedding_status}`));
    if (kind === "experience" && typeof memory.technical_success === "boolean") {
      meta.append(createElement("span", "", `执行：${memory.technical_success ? "成功" : "未成功"}`));
    }
    if (Array.isArray(memory.tags) && memory.tags.length) meta.append(createElement("span", "", `标签：${memory.tags.join("、")}`));
    if (created) meta.append(createElement("span", "", created));
    card.append(meta);
    dom.memoriesList.append(card);
  }
}

function renderMemories(memories, experiences) {
  dom.memoriesList.replaceChildren();
  if (!memories.length && !experiences.length) { emptyList(dom.memoriesList, "还没有形成记忆或经验"); return; }
  dom.memoriesList.append(createElement("h4", "list-section-title", `对话记忆 · ${memories.length}`));
  if (memories.length) appendMemoryCards(memories, "memory");
  else dom.memoriesList.append(createElement("div", "empty-card", "还没有对话记忆"));
  dom.memoriesList.append(createElement("h4", "list-section-title", `经验记录 · ${experiences.length}`));
  if (experiences.length) appendMemoryCards(experiences, "experience");
  else dom.memoriesList.append(createElement("div", "empty-card", "还没有经验记录"));
}

function updateApiKeyStatus(configured) {
  dom.apiKeyStatus.textContent = configured
    ? "已安全配置。为保护密钥，页面不会回填明文；输入新 Key 可替换。"
    : "尚未配置。保存后密钥由本地服务安全存储，页面不会再次显示。";
  dom.apiKeyStatus.classList.toggle("configured", configured);
}

async function loadSettings() {
  try {
    const payload = await api("/api/settings");
    state.settings = payloadObject(payload, ["settings"]);
    const chatModel = state.settings.chat_model || "openrouter/free";
    if (!state.selectedModel && chatModel) state.selectedModel = String(chatModel);
    dom.apiKey.value = "";
    updateApiKeyStatus(state.settings.api_key_configured === true);
    dom.embeddingModel.value = state.settings.embedding_model || "nvidia/nemotron-3-embed-1b:free";
    dom.temperature.value = Number.isFinite(Number(state.settings.temperature)) ? String(state.settings.temperature) : "";
    dom.memoryEnabled.checked = state.settings.memory_enabled !== false;
    dom.allowDataCollection.checked = state.settings.allow_data_collection === true;
    populateModelSelects();
  } catch (error) { showToast(error.message, true); }
  await loadHealth();
}

async function saveSettings(event) {
  event.preventDefault();
  setButtonBusy(dom.settingsSaveButton, true, "保存中…", "保存设置");
  const body = {
    chat_model: dom.chatModelSelect.value || state.settings.chat_model || "openrouter/free",
    embedding_model: dom.embeddingModel.value.trim() || "nvidia/nemotron-3-embed-1b:free",
    memory_enabled: dom.memoryEnabled.checked,
    allow_data_collection: dom.allowDataCollection.checked,
  };
  const temperatureText = dom.temperature.value.trim();
  if (temperatureText) {
    const temperature = Number(temperatureText);
    if (!Number.isFinite(temperature) || temperature < 0 || temperature > 2) {
      showToast("生成温度必须在 0 到 2 之间", true);
      setButtonBusy(dom.settingsSaveButton, false, "保存中…", "保存设置");
      return;
    }
    body.temperature = temperature;
  }
  const apiKey = dom.apiKey.value.trim();
  if (apiKey) body.api_key = apiKey;
  try {
    const payload = await api("/api/settings", { method: "PATCH", body });
    const { api_key: _discardedApiKey, ...nonSecretSettings } = body;
    state.settings = { ...state.settings, ...nonSecretSettings, ...payloadObject(payload, ["settings"]) };
    dom.apiKey.value = "";
    updateApiKeyStatus(state.settings.api_key_configured === true || Boolean(apiKey));
    if (!state.selectedModel && body.chat_model) {
      state.selectedModel = body.chat_model;
      dom.modelSelect.value = body.chat_model;
    }
    showToast("设置已保存");
  } catch (error) { showToast(error.message, true); }
  finally { setButtonBusy(dom.settingsSaveButton, false, "保存中…", "保存设置"); }
}

const mobileMedia = window.matchMedia("(max-width: 760px)");

function syncSidebarMode() {
  if (mobileMedia.matches) {
    dom.sidebar.classList.toggle("open", state.sidebarOpen);
    dom.sidebarBackdrop.classList.toggle("show", state.sidebarOpen);
    dom.mobileMenuButton.setAttribute("aria-expanded", String(state.sidebarOpen));
    dom.sidebar.setAttribute("aria-hidden", String(!state.sidebarOpen));
    dom.sidebar.inert = !state.sidebarOpen;
  } else {
    state.sidebarOpen = false;
    dom.sidebar.classList.remove("open");
    dom.sidebarBackdrop.classList.remove("show");
    dom.mobileMenuButton.setAttribute("aria-expanded", "false");
    dom.sidebar.removeAttribute("aria-hidden");
    dom.sidebar.inert = false;
  }
}

function openSidebar() {
  if (!mobileMedia.matches) return;
  state.sidebarOpen = true;
  syncSidebarMode();
  requestAnimationFrame(() => dom.newChatButton.focus());
}

function closeSidebar({ restoreFocus = false } = {}) {
  if (!mobileMedia.matches) return;
  state.sidebarOpen = false;
  syncSidebarMode();
  if (restoreFocus) dom.mobileMenuButton.focus();
}

function bindEvents() {
  dom.mobileMenuButton.addEventListener("click", openSidebar);
  dom.sidebarCloseButton.addEventListener("click", () => closeSidebar({ restoreFocus: true }));
  dom.sidebarBackdrop.addEventListener("click", () => closeSidebar({ restoreFocus: true }));
  dom.newChatButton.addEventListener("click", () => void createConversation());
  document.querySelectorAll("[data-new-chat]").forEach(node => node.addEventListener("click", event => {
    event.preventDefault();
    void createConversation();
  }));

  dom.conversationList.addEventListener("click", event => {
    const select = event.target.closest("[data-conversation-id]");
    const rename = event.target.closest("[data-rename-conversation]");
    const remove = event.target.closest("[data-delete-conversation]");
    if (select) void selectConversation(select.dataset.conversationId, { focusComposer: true });
    if (rename) openRenameDialog(rename.dataset.renameConversation);
    if (remove) openDeleteDialog(remove.dataset.deleteConversation);
  });

  document.querySelectorAll("[data-management]").forEach(button => {
    button.addEventListener("click", () => void openManagement(button.dataset.management));
  });
  dom.managementCloseButton.addEventListener("click", () => dom.managementDialog.close());
  dom.managementDialog.addEventListener("click", event => {
    if (event.target === dom.managementDialog) dom.managementDialog.close();
  });

  dom.composerForm.addEventListener("submit", event => { event.preventDefault(); void sendMessage(); });
  dom.messageInput.addEventListener("input", () => { saveCurrentDraft(); resizeComposer(); });
  dom.messageInput.addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      dom.composerForm.requestSubmit();
    }
  });
  dom.stopButton.addEventListener("click", requestStop);
  dom.jumpLatestButton.addEventListener("click", () => scrollToBottom(true));
  dom.messageScroller.addEventListener("scroll", () => {
    if (isNearBottom()) dom.jumpLatestButton.classList.add("hidden");
  }, { passive: true });

  dom.modelSelect.addEventListener("change", () => void changeConversationModel());
  dom.renameForm.addEventListener("submit", renameConversation);
  dom.confirmDeleteButton.addEventListener("click", () => void deleteConversation());
  document.querySelectorAll("[data-dialog-cancel]").forEach(button => {
    button.addEventListener("click", () => document.querySelector(`#${button.dataset.dialogCancel}`)?.close());
  });

  dom.createSkillButton.addEventListener("click", () => openSkillEditor("create"));
  dom.skillEditorCloseButton.addEventListener("click", closeSkillEditor);
  dom.skillEditorForm.addEventListener("submit", saveSkill);
  dom.refreshMemoriesButton.addEventListener("click", () => void loadMemories());
  dom.settingsForm.addEventListener("submit", saveSettings);

  window.addEventListener("popstate", () => {
    const id = routeConversationId();
    if (id) void selectConversation(id, { push: false });
    else {
      saveCurrentDraft();
      state.activeConversationId = null;
      updateHeader();
      restoreDraft();
      renderConversationList();
      renderMessages();
    }
  });
  mobileMedia.addEventListener?.("change", syncSidebarMode);
}

async function initialize() {
  bindEvents();
  syncSidebarMode();
  resizeComposer();
  renderMessages();

  await Promise.allSettled([loadHealth(), loadModels(), loadSettings()]);
  try { await loadConversations(); }
  catch (error) { showToast(`会话列表加载失败：${error.message}`, true); }

  const routeId = routeConversationId();
  if (routeId) await selectConversation(routeId, { push: false });
  else if (state.conversations.length) {
    await selectConversation(state.conversations[0].id, { push: false });
    updateRoute(state.conversations[0].id, true);
  } else {
    updateHeader();
    renderMessages();
    dom.messageInput.focus();
  }
}

void initialize();
