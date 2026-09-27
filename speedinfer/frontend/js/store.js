/**
 * SpeedInfer Centralized Reactive State Store
 */

class SpeedInferStore {
  constructor() {
    this.state = {
      user: null,
      balance: 0.0,
      keys: [],
      selectedKey: null,
      models: [],
      selectedModel: 'Qwen/Qwen2.5-7B-Instruct',
      health: { status: 'loading', workers: [], models: [] },
      chatHistory: [],
      currentTab: 'dashboard',
    };
    this.listeners = new Set();
  }

  subscribe(listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  notify(changedKey = null) {
    for (const listener of this.listeners) {
      listener(this.state, changedKey);
    }
  }

  setState(partial) {
    this.state = { ...this.state, ...partial };
    this.notify();
  }

  setUser(user) {
    const balance = user?.credit_balance ?? user?.balance ?? 0.0;
    this.setState({ user, balance });
  }

  setKeys(keys) {
    let selectedKey = this.state.selectedKey;
    const activeKeys = keys.filter(k => k.status === 'active' || k.is_active);
    
    // Auto-select first active key if none selected or current is invalid
    if (!selectedKey || !activeKeys.some(k => k.prefix === selectedKey.prefix || k.id === selectedKey.id)) {
      selectedKey = activeKeys.length > 0 ? activeKeys[0] : null;
    }

    this.setState({ keys, selectedKey });
  }

  setSelectedKey(key) {
    this.setState({ selectedKey: key });
  }

  setModels(models) {
    const modelNames = models.map(m => m.id || m.name);
    let selectedModel = this.state.selectedModel;
    if (modelNames.length > 0 && !modelNames.includes(selectedModel)) {
      selectedModel = modelNames[0];
    }
    this.setState({ models, selectedModel });
  }

  setSelectedModel(modelName) {
    this.setState({ selectedModel: modelName });
  }

  setHealth(health) {
    this.setState({ health });
  }

  setCurrentTab(tab) {
    this.setState({ currentTab: tab });
  }

  addChatMessage(message) {
    const chatHistory = [...this.state.chatHistory, message];
    this.setState({ chatHistory });
  }

  updateLastAssistantMessage(content, metrics = null) {
    const history = [...this.state.chatHistory];
    const lastMsg = history[history.length - 1];
    if (lastMsg && lastMsg.role === 'assistant') {
      lastMsg.content = content;
      if (metrics) lastMsg.metrics = metrics;
      this.setState({ chatHistory: history });
    }
  }

  clearChat() {
    this.setState({ chatHistory: [] });
  }
}

export const store = new SpeedInferStore();
