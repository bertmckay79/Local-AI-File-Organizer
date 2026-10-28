"""
Unified AI client for file classification and geocoding.
Supports two backends through a single interface:
  - Local: Ollama server (default)
  - Cloud:  Any OpenAI-compatible API (OpenAI, Groq, OpenRouter, ...)
The backend is chosen at construction time via the `ai_provider` flag
or at runtime by calling switch_to_cloud() / switch_to_local().
"""

import json
import logging
import requests
from typing import Optional


class OllamaClient:
    """Client for AI inference — local Ollama or cloud (OpenAI-compatible)."""

    def __init__(self, server_url: str = "http://localhost:11434",
                 model: str = "qwen2.5:3b", timeout: int = 60,
                 temperature: float = 0.1, max_tokens: int = 100,
                 # Cloud parameters
                 ai_provider: str = "local",
                 cloud_api_key: str = "",
                 cloud_base_url: str = "https://api.openai.com/v1",
                 cloud_model: str = "gpt-4o-mini",
                 cloud_timeout: int = 60,
                 cloud_temperature: float = 0.1,
                 cloud_max_tokens: int = 500):
        # Local settings
        self.server_url = server_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Cloud settings
        self.cloud_api_key = cloud_api_key
        self.cloud_base_url = cloud_base_url.rstrip("/") if cloud_base_url else ""
        self.cloud_model = cloud_model
        self.cloud_timeout = cloud_timeout
        self.cloud_temperature = cloud_temperature
        self.cloud_max_tokens = cloud_max_tokens

        # Active provider: "local" or "cloud"
        self.ai_provider = ai_provider

        self.logger = logging.getLogger(__name__)

    # ── provider switching ──────────────────────────────────────────

    @property
    def is_cloud(self) -> bool:
        return self.ai_provider == "cloud" and bool(self.cloud_api_key)

    def switch_to_local(self) -> None:
        self.ai_provider = "local"

    def switch_to_cloud(self, api_key: str = None, base_url: str = None,
                        model: str = None) -> None:
        if api_key:
            self.cloud_api_key = api_key
        if base_url:
            self.cloud_base_url = base_url.rstrip("/")
        if model:
            self.cloud_model = model
        self.ai_provider = "cloud"

    def update_settings(self, server_url: str = None, model: str = None,
                        timeout: int = None, temperature: float = None,
                        max_tokens: int = None,
                        # Cloud overloads
                        api_key: str = None, base_url: str = None,
                        ai_provider: str = None,
                        cloud_timeout: int = None,
                        cloud_temperature: float = None,
                        cloud_max_tokens: int = None) -> None:
        """Update settings — works for both local and cloud."""
        # Local
        if server_url is not None:
            self.server_url = server_url.rstrip("/")
        if model is not None:
            self.model = model
        if timeout is not None:
            self.timeout = timeout
        if temperature is not None:
            self.temperature = temperature
        if max_tokens is not None:
            self.max_tokens = max_tokens
        # Cloud
        if api_key is not None:
            self.cloud_api_key = api_key
        if base_url is not None:
            self.cloud_base_url = base_url.rstrip("/") if base_url else ""
        if ai_provider is not None:
            self.ai_provider = ai_provider
        if cloud_timeout is not None:
            self.cloud_timeout = cloud_timeout
        if cloud_temperature is not None:
            self.cloud_temperature = cloud_temperature
        if cloud_max_tokens is not None:
            self.cloud_max_tokens = cloud_max_tokens

    # ── availability & models ───────────────────────────────────────

    def is_available(self) -> bool:
        """Check if the active AI backend is reachable."""
        if self.is_cloud:
            if not self.cloud_api_key:
                return False
            try:
                resp = requests.get(
                    f"{self.cloud_base_url}/models",
                    headers={"Authorization": f"Bearer {self.cloud_api_key}"},
                    timeout=10,
                )
                return resp.status_code == 200
            except requests.RequestException:
                return False
        else:
            try:
                resp = requests.get(f"{self.server_url}/api/tags", timeout=5)
                return resp.status_code == 200
            except requests.RequestException:
                return False

    def list_models(self) -> list[str]:
        """List available models on the active backend.
        For local Ollama, /api/tags returns both local and cloud models
        (cloud models appear with no size). Falls back to /api/ps for
        currently running models.
        """
        if self.is_cloud:
            if not self.cloud_api_key:
                return []
            try:
                resp = requests.get(
                    f"{self.cloud_base_url}/models",
                    headers={"Authorization": f"Bearer {self.cloud_api_key}"},
                    timeout=10,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    models = data.get("data", [])
                    return [m.get("id", "") for m in models if m.get("id")]
            except requests.RequestException:
                pass
            return []
        else:
            models = []
            try:
                resp = requests.get(f"{self.server_url}/api/tags", timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    # Include all models — local ones have a size, cloud ones don't
                    models = [m["name"] for m in data.get("models", []) if m.get("name")]
            except requests.RequestException:
                pass
            # If nothing from /api/tags, also try /api/ps (running models)
            if not models:
                try:
                    resp = requests.get(f"{self.server_url}/api/ps", timeout=5)
                    if resp.status_code == 200:
                        data = resp.json()
                        models = [m["name"] for m in data.get("models", []) if m.get("name")]
                except requests.RequestException:
                    pass
            return models

    # ── inference ───────────────────────────────────────────────────

    def generate(self, prompt: str, system: str = None,
                 stream: bool = False) -> Optional[str]:
        """Generate text (local Ollama only — cloud uses chat)."""
        if self.is_cloud:
            # Route through chat for cloud
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            return self.chat(messages)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        if system:
            payload["system"] = system
        try:
            resp = requests.post(
                f"{self.server_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("response", "")
        except requests.RequestException as e:
            raise ConnectionError(f"Ollama request failed: {e}")
        return None

    def chat(self, messages: list[dict], stream: bool = False,
             use_json_format: bool = True, num_predict: int = None,
             think: Optional[bool] = None) -> Optional[str]:
        """Send a chat request — routes to local Ollama or cloud API.

        Args:
            messages: list of {role, content} dicts.
            stream: if True, stream response (not used here).
            use_json_format: if True, force JSON output mode. Set False
                for tasks where the model might not know the answer.
            num_predict: override max_tokens for this request.
        """
        if self.is_cloud:
            return self._chat_cloud(messages, use_json_format, num_predict)
        return self._chat_local(messages, stream, use_json_format, num_predict, think)

    def test_chat(self, message: str) -> Optional[str]:
        """Test normal conversation, never JSON-mode classification.

        Qwen3 reasoning can consume a small token budget without returning
        visible content. Disable thinking for local Qwen3 variants and allow
        a sufficient response budget for a short human-readable answer.
        """
        model_name = self.cloud_model if self.is_cloud else self.model
        no_thinking = not self.is_cloud and model_name.lower().startswith("qwen3")
        return self.chat(
            [{"role": "user", "content": message}],
            use_json_format=False,
            num_predict=max(768, self.cloud_max_tokens if self.is_cloud else self.max_tokens),
            think=False if no_thinking else None,
        )

    def _chat_local(self, messages: list[dict], stream: bool,
                    use_json_format: bool, num_predict: int,
                    think: Optional[bool] = None) -> Optional[str]:
        """Chat via local Ollama API."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_predict": num_predict if num_predict else self.max_tokens,
            },
        }
        if use_json_format:
            payload["format"] = "json"
        if think is not None:
            payload["think"] = think
        try:
            resp = requests.post(
                f"{self.server_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                message = data.get("message", {})
                content = message.get("content", "")
                if not content:
                    self.logger.warning(
                        "Ollama model %s returned no answer (done_reason=%s, "
                        "eval_count=%s, thinking_characters=%s). "
                        "Check model selection and thinking/token settings.",
                        self.model, data.get("done_reason"), data.get("eval_count"),
                        len(message.get("thinking") or "")
                    )
                return content
            else:
                self.logger.warning(
                    f"Ollama HTTP {resp.status_code}: {resp.text[:500]}"
                )
                # If 401/403, model likely requires ollama signin
                if resp.status_code in (401, 403):
                    self.logger.warning(
                        "Ollama returned auth error — if using a cloud model, "
                        "run 'ollama signin' in terminal first."
                    )
        except requests.RequestException as e:
            raise ConnectionError(f"Ollama chat request failed: {e}")
        return None

    def _chat_cloud(self, messages: list[dict],
                    use_json_format: bool, num_predict: int) -> Optional[str]:
        """Chat via OpenAI-compatible cloud API."""
        if not self.cloud_api_key:
            self.logger.warning("Cloud AI: no API key configured")
            return None

        payload = {
            "model": self.cloud_model,
            "messages": messages,
            "temperature": self.cloud_temperature,
            "max_tokens": num_predict if num_predict else self.cloud_max_tokens,
        }
        if use_json_format:
            payload["response_format"] = {"type": "json_object"}

        try:
            resp = requests.post(
                f"{self.cloud_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.cloud_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.cloud_timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    if not content:
                        self.logger.warning(
                            f"Cloud AI returned empty content. "
                            f"Response: {str(data)[:300]}"
                        )
                    return content
                else:
                    self.logger.warning(
                        f"Cloud AI: no choices in response: {str(data)[:300]}"
                    )
            else:
                self.logger.warning(
                    f"Cloud AI HTTP {resp.status_code}: {resp.text[:300]}"
                )
        except requests.RequestException as e:
            self.logger.warning(f"Cloud AI request failed: {e}")

        return None

    # ── classification ──────────────────────────────────────────────

    def classify_file(self, file_info: dict, categories: list[str],
                      content_summary: str = None,
                      category_details: list[dict] = None) -> Optional[str]:
        """Classify a file into one of the given categories."""
        categories_str = ", ".join(categories)
        file_name = file_info.get("file_name", "unknown")
        extension = file_info.get("extension", "")
        metadata = file_info.get("metadata", {})

        prompt_parts = [
            f"File name: {file_name}",
            f"Extension: {extension}",
        ]
        if metadata:
            prompt_parts.append(f"Metadata: {json.dumps(metadata, indent=2, default=str)}")
        if content_summary:
            prompt_parts.append(f"Document evidence: {content_summary[:7000]}")

        if category_details:
            descriptions = []
            for item in category_details:
                name = item.get("name", "")
                if name in categories:
                    keywords = ", ".join(str(k) for k in item.get("keywords", [])[:12])
                    description = item.get("description", "")
                    descriptions.append(
                        f"- {name}: {description} Keywords: {keywords}".strip()
                    )
            prompt_parts.append("Category definitions:\n" + "\n".join(descriptions))
        else:
            prompt_parts.append(f"\nAvailable categories: {categories_str}")
        prompt_parts.append(
            "\nYou are a professional file organizer. Based on the filename, extension, "
            "metadata, and available content, return only the best category from the "
            "configured category list. If uncertain, return Miscellaneous."
        )
        prompt_parts.append('\nReturn JSON only: {"category": "<category_name>"}')

        prompt_parts.append(
            "Classify documents primarily by their subject matter, "
            "not their extension. Prefer specific RPG, software, "
            "game development or fiction categories over generic "
            "Documents and eBooks categories. "
            "Only use generic categories when no specific "
            "category is appropriate."
        )

        prompt = "\n".join(prompt_parts)
        system_msg = (
            "You are a professional file organizer. Analyze file information and classify "
            "it into the most appropriate category. Return only valid JSON with a 'category' key. "
            "Be precise and consistent with classifications."
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]

        # Older Ollama versions may ignore format=json for Qwen3.5 with
        # thinking disabled, so request a plain answer and parse its JSON.
        qwen35_local = not self.is_cloud and self.model.lower().startswith("qwen3.5")
        response = self.chat(
            messages, use_json_format=not qwen35_local,
            num_predict=max(768, self.max_tokens),
            think=False if qwen35_local else None,
        )
        if not response:
            self.logger.warning("Classification returned no answer for %s using model %s",
                                file_name, self.cloud_model if self.is_cloud else self.model)
            return None

        # Accept structured JSON surrounded by markdown fences or a preamble,
        # but never classify merely because the text contains a category name.
        parsed_category = None
        decoder = json.JSONDecoder()
        for index, char in enumerate(response):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(response[index:])
                if isinstance(parsed, dict) and "category" in parsed:
                    parsed_category = str(parsed.get("category") or "").strip()
                    break
            except json.JSONDecodeError:
                continue
        if parsed_category is None:
            self.logger.warning("Invalid classification JSON for %s (model=%s, preview=%r)",
                                file_name, self.model, response[:160])
            return None
        for candidate in categories:
            if candidate.casefold() == parsed_category.casefold():
                return candidate
        self.logger.warning("Unrecognised category %r from model for %s",
                            parsed_category, file_name)
        return None

    def batch_classify(self, files: list[dict], categories: list[str],
                       content_summaries: dict = None,
                       progress_callback: callable = None) -> dict[str, str]:
        """Classify multiple files. Returns {file_path: category}."""
        results = {}
        total = len(files)
        content_summaries = content_summaries or {}
        for i, file_info in enumerate(files):
            file_path = file_info.get("file_path", file_info.get("file_name", str(i)))
            summary = content_summaries.get(file_path)
            category = self.classify_file(file_info, categories, summary)
            if category:
                results[file_path] = category
            if progress_callback:
                progress_callback(i + 1, total)
        return results

    def pull_model(self, model_name: str) -> bool:
        """Pull a model — only relevant for local Ollama."""
        if self.is_cloud:
            return True  # Cloud models are already available
        try:
            resp = requests.post(
                f"{self.server_url}/api/pull",
                json={"name": model_name, "stream": False},
                timeout=300,
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False
