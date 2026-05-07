from __future__ import annotations
import os
import time
import warnings
from typing import Optional
import httpx
warnings.filterwarnings('ignore', message='.*Unverified HTTPS.*')
_DEFAULT_YT_URLS = ['https://api.eliza.yandex.net/internal/qwen3-5-397b-a17b-fp8/v1', 'https://api.eliza.yandex.net/internal/alice-ai-llm-235b-latest/generative/v1', 'https://api.eliza.yandex.net/internal/glm-latest/v1', 'https://api.eliza.yandex.net/internal/minimax-latest/v1', 'https://api.eliza.yandex.net/internal/deepseek-latest/v1', 'https://api.eliza.yandex.net/internal/gpt-oss-120b/v1']

class ElizaClient:

    def __init__(self, token: str, base_url: str=_DEFAULT_YT_URLS[0], model_name: str='', timeout: float=120.0) -> None:
        self.token = token
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.timeout = timeout
        self._http = httpx.Client(verify=False, timeout=timeout)

    @classmethod
    def from_env(cls, timeout: float=90.0) -> 'ElizaClient':
        token = os.environ.get('SOY_TOKEN', '')
        if not token:
            raise EnvironmentError('SOY_TOKEN env var is not set')
        return cls(token=token, timeout=timeout)

    def _auth_headers(self) -> dict[str, str]:
        return {'Authorization': f'OAuth {self.token}'}

    def list_models(self, base_url: str | None=None) -> list[str]:
        url = (base_url or self.base_url).rstrip('/')
        resp = self._http.get(f'{url}/models', headers=self._auth_headers(), timeout=15)
        resp.raise_for_status()
        body = resp.json()
        inner = body.get('response', body)
        return [m['id'] for m in inner.get('data', [])]

    def probe(self, candidate_urls: list[str] | None=None) -> bool:
        urls = candidate_urls or _DEFAULT_YT_URLS
        probe_msgs = [{'role': 'user', 'content': 'Ответь одним словом: OK'}]

        def _endpoint_label(u: str) -> str:
            u = u.rstrip('/')
            if '/internal/' in u:
                return u.split('/internal/', 1)[1].removesuffix('/v1')
            return u
        for url in urls:
            mark = _endpoint_label(url)
            try:
                models = self.list_models(url)
                if not models:
                    print(f'  ? {mark}  /v1/models returned empty list')
                    continue
                print(f'  {mark}  vLLM models: {models}')
                model = models[0]
            except Exception as e:
                body = getattr(getattr(e, 'response', None), 'text', str(e))[:300]
                print(f'  ✗ {mark}  /v1/models: {body}')
                continue
            try:
                self.chat(probe_msgs, model=model, base_url=url)
                self.model_name = model
                self.base_url = url.rstrip('/')
                print(f'✓ Working pair: model={model!r}\n  base_url={self.base_url}')
                return True
            except httpx.HTTPStatusError as e:
                body = e.response.text[:300]
                print(f'  ✗ {mark}  chat/completions HTTP {e.response.status_code}: {body}')
            except Exception as e:
                print(f'  ✗ {mark}  chat/completions: {e}')
        return False

    def chat(self, messages: list[dict], *, model: Optional[str]=None, base_url: Optional[str]=None, max_tokens: int=512, temperature: float=0.1) -> str:
        _model = model or self.model_name
        _base = (base_url or self.base_url).rstrip('/')
        resp = self._http.post(f'{_base}/chat/completions', headers={**self._auth_headers(), 'Content-Type': 'application/json'}, json={'model': _model, 'messages': messages, 'max_tokens': max_tokens, 'temperature': temperature}, timeout=self.timeout)
        resp.raise_for_status()
        body = resp.json()
        data = body.get('response', body)
        return data['choices'][0]['message']['content']

    def chat_with_retry(self, messages: list[dict], attempts: int=3, delay: float=5.0, **kwargs) -> tuple[str | None, str | None]:
        last_error: str | None = None
        for attempt in range(attempts):
            try:
                return (self.chat(messages, **kwargs), None)
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                body = e.response.text[:400]
                last_error = f'HTTP {status}: {body}'
                if status == 429 and '0 >= 0' in body:
                    break
            except Exception as e:
                last_error = f'{type(e).__name__}: {e}'
            if attempt < attempts - 1:
                time.sleep(delay * (attempt + 1))
        return (None, last_error)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> 'ElizaClient':
        return self

    def __exit__(self, *_) -> None:
        self.close()
