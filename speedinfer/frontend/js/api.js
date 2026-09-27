/**
 * SpeedInfer API Client
 * Production client communicating directly with SpeedInfer FastAPI gateway endpoints.
 */

const API_BASE = window.location.origin;

class SpeedInferApiClient {
  constructor() {
    this.tokenKey = 'speedinfer_access_token';
    this.userKey = 'speedinfer_user_profile';
  }

  // Session Token Management
  getToken() {
    return localStorage.getItem(this.tokenKey);
  }

  setToken(token) {
    if (token) {
      localStorage.setItem(this.tokenKey, token);
    } else {
      localStorage.removeItem(this.tokenKey);
    }
  }

  getUser() {
    try {
      const data = localStorage.getItem(this.userKey);
      return data ? JSON.parse(data) : null;
    } catch {
      return null;
    }
  }

  setUser(user) {
    if (user) {
      localStorage.setItem(this.userKey, JSON.stringify(user));
    } else {
      localStorage.removeItem(this.userKey);
    }
  }

  clearSession() {
    localStorage.removeItem(this.tokenKey);
    localStorage.removeItem(this.userKey);
  }

  isAuthenticated() {
    return !!this.getToken();
  }

  /**
   * Internal HTTP request helper with unified authorization and error envelopes
   */
  async request(endpoint, options = {}) {
    const url = `${API_BASE}${endpoint}`;
    const headers = {
      'Content-Type': 'application/json',
      Accept: 'application/json',
      ...options.headers,
    };

    // Attach JWT Bearer if available and not explicitly provided
    const token = this.getToken();
    if (token && !headers.Authorization && !headers.authorization) {
      headers.Authorization = `Bearer ${token}`;
    }

    try {
      const response = await fetch(url, {
        ...options,
        headers,
      });

      // Handle 401 Unauthorized session expiration
      if (response.status === 401 && !endpoint.includes('/auth/login') && !endpoint.includes('/auth/register')) {
        // If an API key wasn't the caller, handle session expiration
        if (!headers.Authorization?.startsWith('Bearer sk-speedinfer-')) {
          this.clearSession();
          window.dispatchEvent(new CustomEvent('speedinfer:session_expired'));
        }
      }

      // Parse JSON response or handle errors
      let data = null;
      const text = await response.text();
      if (text) {
        try {
          data = JSON.parse(text);
        } catch {
          data = { error: { message: text } };
        }
      }

      if (!response.ok) {
        const errorMsg = data?.error?.message || data?.detail?.error?.message || data?.detail || `Request failed with status ${response.status}`;
        const err = new Error(errorMsg);
        err.status = response.status;
        err.data = data;
        throw err;
      }

      return data;
    } catch (err) {
      throw err;
    }
  }

  // =========================================================================
  // Authentication & User Profile
  // =========================================================================
  async login(email, password, turnstile_token = null) {
    const payload = { email, password };
    if (turnstile_token) payload.turnstile_token = turnstile_token;
    const data = await this.request('/v1/auth/login', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    if (data.access_token) {
      this.setToken(data.access_token);
      this.setUser(data.user);
    }
    return data;
  }

  async register({
    email,
    password,
    name = '',
    initial_balance = 0.0,
    create_api_key = false,
    accepted_policy_version = null,
    referral_code = null,
    device_fingerprint = null,
    turnstile_token = null,
  }) {
    const payload = {
      email,
      password,
      name: name || undefined,
      initial_balance: parseFloat(initial_balance),
      create_api_key: Boolean(create_api_key),
      accepted_policy_version,
    };
    if (referral_code) payload.referral_code = referral_code;
    if (device_fingerprint) payload.device_fingerprint = device_fingerprint;
    if (turnstile_token) payload.turnstile_token = turnstile_token;

    const data = await this.request('/v1/auth/register', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    if (data.access_token) {
      this.setToken(data.access_token);
      this.setUser(data.user);
    }
    return data;
  }

  async verifyEmail({ email, code }) {
    return await this.request('/v1/auth/verify-email', {
      method: 'POST',
      body: JSON.stringify({ email, code }),
    });
  }

  async resendCode({ email, purpose = 'registration' }) {
    return await this.request('/v1/auth/resend-code', {
      method: 'POST',
      body: JSON.stringify({ email, purpose }),
    });
  }

  async forgotPassword({ email, turnstile_token = null }) {
    const payload = { email };
    if (turnstile_token) payload.turnstile_token = turnstile_token;
    return await this.request('/v1/auth/forgot-password', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  async resetPassword({ email, code, new_password }) {
    return await this.request('/v1/auth/reset-password', {
      method: 'POST',
      body: JSON.stringify({ email, code, new_password }),
    });
  }

  async getMe() {
    const data = await this.request('/v1/auth/me');
    this.setUser(data);
    return data;
  }

  async getProfile() {
    const data = await this.request('/v1/auth/profile');
    this.setUser(data);
    return data;
  }

  async updateProfile({ name = null, avatar_url = null, location = null, organization = null }) {
    const payload = {};
    if (name !== null) payload.name = name;
    if (avatar_url !== null) payload.avatar_url = avatar_url;
    if (location !== null) payload.location = location;
    if (organization !== null) payload.organization = organization;

    const data = await this.request('/v1/auth/profile', {
      method: 'PATCH',
      body: JSON.stringify(payload),
    });
    this.setUser(data);
    return data;
  }

  // =========================================================================
  // Referral Program
  // =========================================================================
  async getReferrals() {
    return await this.request('/v1/referrals/me');
  }

  // =========================================================================
  // API Keys Management
  // =========================================================================
  async listKeys(limit = 100, offset = 0) {
    return await this.request(`/v1/keys?limit=${limit}&offset=${offset}`);
  }

  async createKey({ name, rpm_limit = 60, tpm_limit = 60000, expires_in_days = null, permissions = null }) {
    const payload = {
      name,
      rpm_limit: parseInt(rpm_limit, 10),
      tpm_limit: parseInt(tpm_limit, 10),
    };
    if (expires_in_days) payload.expires_in_days = parseInt(expires_in_days, 10);
    if (permissions) payload.permissions = permissions;

    return await this.request('/v1/keys', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  async revokeKey(keyId, permanent = false) {
    return await this.request(`/v1/keys/${keyId}?permanent=${permanent}`, {
      method: 'DELETE',
    });
  }

  // =========================================================================
  // Models Catalog & System Health
  // =========================================================================
  async listModels(apiKey = null) {
    const headers = {};
    if (apiKey) {
      headers.Authorization = `Bearer ${apiKey}`;
    }
    return await this.request('/v1/models', { headers });
  }

  async checkHealth() {
    return await this.request('/health');
  }

  // =========================================================================
  // Usage & Quota
  // =========================================================================
  async getUsage(apiKey) {
    return await this.request('/v1/usage', {
      headers: { Authorization: `Bearer ${apiKey}` },
    });
  }

  // =========================================================================
  // Billing
  // =========================================================================
  async getCreditPackages() {
    return await this.request('/v1/billing/packages');
  }

  async createCheckout(amount_usd, api_key_id = null) {
    return await this.request('/v1/billing/checkout', {
      method: 'POST',
      body: JSON.stringify({ amount_usd: Number(amount_usd), api_key_id }),
    });
  }

  // =========================================================================
  // Inference & Streaming Chat Completions
  // =========================================================================
  async streamChatCompletion({
    model,
    messages,
    temperature = 0.7,
    max_tokens = 1024,
    apiKey,
    onChunk,
    onDone,
    onError,
  }) {
    const url = `${API_BASE}/v1/chat/completions`;
    const payload = {
      model,
      messages,
      temperature: parseFloat(temperature),
      max_tokens: parseInt(max_tokens, 10),
      stream: true,
    };

    const startTime = performance.now();
    let ttft = null;
    let fullText = '';
    let promptTokens = 0;
    let completionTokens = 0;

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const message = errorData?.error?.message || `Inference error: ${response.status}`;
        throw new Error(message);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); // keep partial line

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || trimmed.startsWith(':')) continue;

          if (trimmed.startsWith('data: ')) {
            const dataStr = trimmed.substring(6).trim();
            if (dataStr === '[DONE]') {
              continue;
            }

            try {
              const parsed = JSON.parse(dataStr);
              const deltaContent = parsed.choices?.[0]?.delta?.content || '';
              if (deltaContent) {
                if (ttft === null) {
                  ttft = Math.round(performance.now() - startTime);
                }
                fullText += deltaContent;
                onChunk({ delta: deltaContent, fullText, ttft });
              }
            } catch {
              // Ignore non-json or malformed SSE line
            }
          }
        }
      }

      const totalTimeMs = Math.round(performance.now() - startTime);
      // Rough estimation if backend doesn't send final usage chunk in streaming
      promptTokens = Math.max(1, Math.round(messages.reduce((acc, m) => acc + (m.content?.length || 0), 0) / 4));
      completionTokens = Math.max(1, Math.round(fullText.length / 4));

      onDone({
        fullText,
        ttft: ttft || totalTimeMs,
        totalTimeMs,
        promptTokens,
        completionTokens,
        totalTokens: promptTokens + completionTokens,
      });
    } catch (err) {
      onError(err);
    }
  }
}

export const api = new SpeedInferApiClient();
