from __future__ import annotations
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
import pandas as pd
from tqdm.auto import tqdm
from antifraud_components.labeling.eliza_client import ElizaClient
_EXCLUDE_FROM_PROMPT: frozenset[str] = frozenset(['resolution_fraud', 'resolution_lvl_1', 'resolution_lvl_2', 'resolution_negative', 'resolution_no_fraud', 'resolution_fraud', 'ead7_2'])
_SCHEMA_KEYS = ('resolution_fraud', 'resolution_negative', 'resolution_no_fraud', 'resolution_lvl_1', 'resolution_lvl_2', 'confidence', 'reasoning', 'escalation')

def _sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj

def _normalize_flag(v) -> int:
    if v is None or (isinstance(v, str) and (not v.strip())):
        return 0
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return 1 if float(v) != 0 else 0
    s = str(v).strip().lower()
    if s in {'1', 'true', 'yes', 'y'}:
        return 1
    if s in {'0', 'false', 'no', 'n', 'null', 'none', 'nan'}:
        return 0
    raise ValueError(f'Cannot coerce to 0/1: {v!r}')

def _normalize_nullable_str(v) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return None if s.lower() in {'', 'none', 'null', 'nan'} else s

def _parse_llm_response(text: str) -> dict:
    text = text.strip()
    text = re.sub('^```(?:json)?\\s*', '', text)
    text = re.sub('\\s*```$', '', text)
    match = re.search('\\{.*\\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f'No JSON found in response: {text[:200]}')

def _validate_response(resp: dict) -> dict:
    out = {'resolution_fraud': _normalize_flag(resp.get('resolution_fraud', 0)), 'resolution_negative': _normalize_flag(resp.get('resolution_negative', 0)), 'resolution_no_fraud': _normalize_flag(resp.get('resolution_no_fraud', 0)), 'resolution_lvl_1': _normalize_nullable_str(resp.get('resolution_lvl_1')), 'resolution_lvl_2': _normalize_nullable_str(resp.get('resolution_lvl_2')), 'confidence': _normalize_nullable_str(resp.get('confidence')), 'reasoning': _normalize_nullable_str(resp.get('reasoning')), 'escalation': _normalize_nullable_str(resp.get('escalation'))}
    active = out['resolution_fraud'] + out['resolution_negative'] + out['resolution_no_fraud']
    if active > 1:
        raise ValueError(f'More than one branch active (sum={active})')
    if out['resolution_lvl_2'] is not None and out['resolution_lvl_1'] is None:
        raise ValueError('resolution_lvl_2 is set but resolution_lvl_1 is None')
    return out

class LLMLabeler:

    def __init__(self, client: ElizaClient, system_prompt: str, max_workers: int=2, retry_attempts: int=3, retry_delay: float=5.0) -> None:
        self.client = client
        self.system_prompt = system_prompt
        self.max_workers = max_workers
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

    @classmethod
    def from_env(cls, system_prompt_path: str | Path='docs/llm_alert_resolution_labeling_prompt.md', max_workers: int=2, timeout: float=90.0) -> 'LLMLabeler':
        client = ElizaClient.from_env(timeout=timeout)
        prompt_path = Path(system_prompt_path)
        system_prompt = cls._load_system_prompt(prompt_path)
        return cls(client=client, system_prompt=system_prompt, max_workers=max_workers)

    @staticmethod
    def _load_system_prompt(path: Path) -> str:
        text = path.read_text(encoding='utf-8')
        marker = '## Системный промпт'
        start = text.find(marker)
        if start == -1:
            return text
        tail = text[start:]
        code_start = tail.find('```')
        if code_start == -1:
            return tail
        tail = tail[code_start + 3:]
        code_end = tail.find('```')
        return tail[:code_end].strip() if code_end != -1 else tail.strip()

    def _row_to_prompt(self, row: dict) -> str:
        payload = {k: v for k, v in row.items() if k not in _EXCLUDE_FROM_PROMPT}
        payload = _sanitize_for_json(payload)
        return 'Разметь резолюции для следующей строки таблицы (обезличенно). Верни только JSON по схеме из системного промпта.\n\n' + json.dumps(payload, ensure_ascii=False, default=str, allow_nan=False)

    def label_row(self, row: dict, idx: int=0) -> dict:
        messages = [{'role': 'system', 'content': self.system_prompt}, {'role': 'user', 'content': self._row_to_prompt(row)}]
        text, error = self.client.chat_with_retry(messages, attempts=self.retry_attempts, delay=self.retry_delay)
        if error:
            return {k: 0 if 'flag' in k or k.startswith('resolution_fraud') or k.startswith('resolution_neg') or k.startswith('resolution_no') else None for k in _SCHEMA_KEYS} | {'_idx': idx, '_raw': None, '_error': error}
        try:
            validated = _validate_response(_parse_llm_response(text))
            return {**validated, '_idx': idx, '_raw': text, '_error': None}
        except Exception as e:
            return {k: 0 if k.startswith('resolution_fraud') or k.startswith('resolution_neg') or k.startswith('resolution_no') else None for k in _SCHEMA_KEYS} | {'_idx': idx, '_raw': text, '_error': f'parse error: {e}'}

    def label_dataframe(self, df: pd.DataFrame, max_rows: Optional[int]=None) -> pd.DataFrame:
        subset = df.head(max_rows) if max_rows else df
        rows = subset.to_dict(orient='records')
        results: list[Optional[dict]] = [None] * len(rows)
        print(f'Labeling {len(rows)} orders with model {self.client.model_name}...')
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self.label_row, row, idx): idx for idx, row in enumerate(rows)}
            for future in tqdm(as_completed(futures), total=len(futures), desc='LLM labeling'):
                res = future.result()
                results[res['_idx']] = res
        errors = sum((1 for r in results if r and r['_error']))
        print(f'Done. Errors: {errors}/{len(rows)}')
        result_df = subset.copy()
        for key in _SCHEMA_KEYS:
            llm_col = f'llm_{key}' if not key.startswith('resolution_') else f'llm_{key}'
            result_df[llm_col] = [r[key] for r in results]
        result_df['llm_error'] = [r['_error'] for r in results]
        result_df['llm_raw'] = [r['_raw'] for r in results]
        return result_df

def label_batch(df: pd.DataFrame, max_rows: Optional[int]=200, system_prompt_path: str | Path='docs/llm_alert_resolution_labeling_prompt.md', max_workers: int=2) -> pd.DataFrame:
    labeler = LLMLabeler.from_env(system_prompt_path=system_prompt_path, max_workers=max_workers)
    if not labeler.client.probe():
        raise RuntimeError('No working Eliza endpoint found. Check SOY_TOKEN and https://wiki.yandex-team.ru/eliza/communal/')
    return labeler.label_dataframe(df, max_rows=max_rows)
