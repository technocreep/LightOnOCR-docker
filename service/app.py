import sys
import os
import io
import asyncio
import base64
import logging
import time
import re
import math
import json
import pprint
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

import aiohttp
import cv2
import numpy as np
import secrets
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Header, Request
from fastapi.responses import JSONResponse
from pdf2image import convert_from_bytes
from pypdf import PdfReader
from PIL import Image

import pymorphy2
import markdown_table_repair as mtr
from ru_text_cleaner import SimpleCleaner
from lingua import LanguageDetectorBuilder
from symspellpy import SymSpell, Verbosity


Image.MAX_IMAGE_PIXELS = None

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("ocr-app")

app = FastAPI()

VLLM_URL = os.getenv("VLLM_URL", "http://vllm-server:8001/v1/chat/completions")
MODEL_NAME = "lightonai/LightOnOCR-1B-1025"
OUTPUT_DIR = Path("./output_texts")

PDF_DPI = 200
SAVE_QUALITY = 90
MAX_CONCURRENT_REQUESTS = 8

executor = ThreadPoolExecutor(max_workers=8)

text_cleaner = SimpleCleaner(
    spaces=False,        # убирает многократные пробелы в тексте
    punctuation=False,   # убирает знаки пунктуации в строке
    html=True,           # убирает HTML-теги
    emoji=True,          # убирает эмодзи
    lower=False,         # переводит текст в нижний регистр
    stop_words=False,    # убирает стоп-слова (союзы, предлоги и так далее)
    morpheme=False,      # преобразует слова в их начальные формы (автоматически переводит текст в нижний регистр)
)


# --- Mistral OCR API compatibility (OpenWebUI mistral_ocr loader) ---
_STORE_DIR = Path(os.getenv("OCR_STORE_DIR", "/tmp/local-ocr-store"))
_STORE_DIR.mkdir(parents=True, exist_ok=True)

# file_id -> metadata
_FILE_INDEX: Dict[str, Dict[str, Any]] = {}

# Spell checking
_SYMSPELL: Optional[SymSpell] = None
_DICT_PATH: Optional[Path] = None
_MORPH: Optional[pymorphy2.MorphAnalyzer] = None
_CFG_CACHE: Dict[str, Any] = {}
_SEEDED_KEY: Optional[str] = None

_RU_WORD_RE = re.compile(r"[А-Яа-яЁё]+")
_ALNUM_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]")

_INFLECT_GRAMS = {
    "nomn","gent","datv","accs","ablt","loct","voct","gen2","acc2","loc2",
    "sing","plur","masc","femn","neut","anim","inan","past","pres","futr",
    "1per","2per","3per","indc","impr","perf","impf","actv","pssv","tran","intr",
    "brev","plen","comp","supr",
}


def _now() -> float:
    return time.time()


def _file_path(file_id: str) -> Path:
    return _STORE_DIR / f"{file_id}.bin"


def _delete_file_local(file_id: str) -> None:
    _FILE_INDEX.pop(file_id, None)
    try:
        _file_path(file_id).unlink(missing_ok=True)
    except Exception:
        pass


def _prune_store() -> None:
    ttl_s = int(os.getenv("OCR_STORE_TTL_SECONDS", "3600"))
    max_files = int(os.getenv("OCR_STORE_MAX_FILES", "200"))

    cutoff = _now() - ttl_s
    expired = [fid for fid, meta in list(_FILE_INDEX.items()) if float(meta.get("created_at", 0)) < cutoff]
    for fid in expired:
        _delete_file_local(fid)

    if len(_FILE_INDEX) > max_files:
        ordered = sorted(_FILE_INDEX.items(), key=lambda kv: float(kv[1].get("created_at", 0)))
        for fid, _ in ordered[: max(0, len(_FILE_INDEX) - max_files)]:
            _delete_file_local(fid)


def _is_pdf(filename: Optional[str], content_type: Optional[str]) -> bool:
    if content_type and "pdf" in content_type.lower():
        return True
    if filename and filename.lower().endswith(".pdf"):
        return True
    return False


def build_report(
    text: str,
    previous_report: Optional[Dict[str, Any]] = None,
    *,
    config_path: str = "./spell_config.json",
    cache_dir: str = "./.symspell_cache",
    dict_filename: str = "ru-100k.txt",
    dict_url: str = "https://raw.githubusercontent.com/wolfgarbe/SymSpell/master/SymSpell.FrequencyDictionary/ru-100k.txt",
    html_tag_ratio_ok: float = 0.01,
    empty_line_ratio_warn: float = 0.15,
    short_line_ratio_warn: float = 0.20,
    short_line_max_chars: int = 30,
    single_char_token_fail: float = 0.10,
    max_anomaly_samples: int = 10,
    max_edit_distance_dictionary: int = 2,
    prefix_length: int = 7,
    max_edit_distance_lookup: int = 2,
    min_candidate_count: int = 200,
    spell_min_word_len: int = 5,
    spell_max_occurrences: int = 20000,
    auto_whitelist_min_count: int = 8,
    min_repeat_whitelist: int = 2,
    skip_titlecase: bool = True,
    require_same_pos: bool = True,
    max_len_delta: int = 2,
    require_prefix_chars: int = 2,
) -> Dict[str, Any]:
    global _SYMSPELL, _DICT_PATH, _MORPH, _CFG_CACHE, _SEEDED_KEY

    s = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    lines = s.split("\n")
    total_chars = len(s)
    total_lines = max(1, len(lines))

    empty_lines = sum(1 for ln in lines if not ln.strip())
    short_lines = sum(1 for ln in lines if 0 < len(ln.strip()) <= short_line_max_chars)

    toks = re.findall(r"\S+", s)
    total_tokens = len(toks)

    def is_single_letter(t: str) -> bool:
        return len(t) == 1 and re.fullmatch(r"[A-Za-zА-Яа-яЁё]", t) is not None

    avg_token_length = (sum(len(t) for t in toks) / total_tokens) if total_tokens else 0.0
    single_char_token_ratio = (sum(1 for t in toks if is_single_letter(t)) / total_tokens) if total_tokens else 0.0

    html_tag_ratio = len(re.findall(r"<[^>]+>", s)) / max(1, total_chars)
    empty_line_ratio = empty_lines / total_lines
    short_line_ratio = short_lines / total_lines

    cyr = len(re.findall(r"[А-Яа-яЁё]", s))
    lat = len(re.findall(r"[A-Za-z]", s))
    cyrillic_ratio = cyr / max(1, (cyr + lat))
    latin_russian_ratio = sum(
        1 for t in toks
        if re.search(r"[A-Za-z]", t) and not re.search(r"[А-Яа-яЁё]", t) and len(t) >= 4
    ) / max(1, total_tokens)

    detector = LanguageDetectorBuilder.from_all_languages().build()
    paras = [p for p in re.split(r"\n{2,}", s) if p.strip()]
    unknown_language_blocks = sum(1 for p in paras if len(p) >= 200 and detector.detect_language_of(p) is None)

    cfg: Dict[str, Any] = {}
    cp = Path(config_path)
    if cp.exists():
        key = f"{cp.as_posix()}::{cp.stat().st_mtime_ns}"
        cached = _CFG_CACHE.get(key)
        if cached is None:
            try:
                cached = json.loads(cp.read_text(encoding="utf-8")) or {}
            except Exception:
                cached = {}
            _CFG_CACHE = {key: cached}
        cfg = cached if isinstance(cached, dict) else {}

    params = cfg.get("params") or {}
    if isinstance(params, dict):
        max_edit_distance_lookup = int(params.get("max_edit_distance_lookup", max_edit_distance_lookup))
        min_candidate_count = int(params.get("min_candidate_count", min_candidate_count))
        require_prefix_chars = int(params.get("require_prefix_chars", require_prefix_chars))
        max_len_delta = int(params.get("max_len_delta", max_len_delta))
        require_same_pos = bool(params.get("require_same_pos", require_same_pos))
        skip_titlecase = bool(params.get("skip_titlecase", skip_titlecase))
        spell_min_word_len = int(params.get("spell_min_word_len", spell_min_word_len))

    wl = set()
    wl_list = cfg.get("whitelist") or []
    if isinstance(wl_list, list):
        wl |= {str(x).lower() for x in wl_list if isinstance(x, (str, int))}
    wl_extra = cfg.get("whitelist_extra") or []
    if isinstance(wl_extra, list):
        wl |= {str(x).lower() for x in wl_extra if isinstance(x, (str, int))}

    if _SYMSPELL is None:
        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        dict_path = cache / dict_filename
        if not dict_path.exists():
            urllib.request.urlretrieve(dict_url, dict_path.as_posix())
        sym = SymSpell(max_dictionary_edit_distance=max_edit_distance_dictionary, prefix_length=prefix_length)
        if not sym.load_dictionary(dict_path.as_posix(), term_index=0, count_index=1, separator=" "):
            raise RuntimeError(f"SymSpell dictionary load failed: {dict_path.as_posix()}")
        _SYMSPELL, _DICT_PATH = sym, dict_path

    if _MORPH is None:
        _MORPH = pymorphy2.MorphAnalyzer()

    seed = cfg.get("seed_dictionary_entries") or {}
    seed_key = json.dumps(seed, ensure_ascii=False, sort_keys=True) if isinstance(seed, dict) else "{}"
    if seed_key != _SEEDED_KEY:
        if isinstance(seed, dict):
            for term, cnt in seed.items():
                try:
                    _SYMSPELL.create_dictionary_entry(str(term).lower(), int(cnt))
                except Exception:
                    pass
        _SEEDED_KEY = seed_key

    prefix_no_fix = set()
    pref_cfg = cfg.get("no_fix_prefixes") or []
    if isinstance(pref_cfg, list):
        prefix_no_fix = {str(x).lower() for x in pref_cfg if x}

    block_contains = cfg.get("block_if_contains") or []
    if isinstance(block_contains, list):
        block_contains = [str(x).lower() for x in block_contains if isinstance(x, str) and x]
    else:
        block_contains = []

    def compile_rules(key: str) -> List[Dict[str, Any]]:
        rs = cfg.get(key) or []
        out: List[Dict[str, Any]] = []
        if not isinstance(rs, list):
            return out
        for r in rs:
            if not isinstance(r, dict):
                continue
            rr = dict(r)
            if "src_regex" in rr:
                try:
                    rr["_src_re"] = re.compile(str(rr["src_regex"]), re.IGNORECASE)
                except Exception:
                    rr["_src_re"] = None
            if "candidate_regex" in rr:
                try:
                    rr["_cand_re"] = re.compile(str(rr["candidate_regex"]), re.IGNORECASE)
                except Exception:
                    rr["_cand_re"] = None
            out.append(rr)
        return out

    rule_block = compile_rules("block_rules")
    rule_allow = compile_rules("allow_rules")

    def is_sentence_start(pos: int) -> bool:
        i = pos - 1
        while i >= 0 and s[i].isspace():
            i -= 1
        if i < 0:
            return True
        if s[i] in ".!?…":
            return True
        if s[i] == "\n":
            j = i - 1
            while j >= 0 and s[j].isspace() and s[j] != "\n":
                j -= 1
            if j < 0:
                return True
            return s[j] in ".!?…"
        return False

    def pref_len(a: str, b: str) -> int:
        m = min(len(a), len(b))
        i = 0
        while i < m and a[i] == b[i]:
            i += 1
        return i

    def restore_form(orig_parse, fixed_lemma: str) -> Optional[str]:
        grams = set(orig_parse.tag.grammemes) & _INFLECT_GRAMS
        orig_pos = orig_parse.tag.POS
        parses = _MORPH.parse(fixed_lemma)
        if not parses:
            return None
        base = next((pp for pp in parses if orig_pos is None or pp.tag.POS == orig_pos), parses[0])
        if grams:
            inf = base.inflect(grams)
            if inf and inf.word:
                return inf.word
        return base.word

    def entropy_alnum(x: str) -> float:
        chars = _ALNUM_RE.findall(x or "")
        c = Counter(chars)
        n = sum(c.values())
        return sum((-(v / n) * math.log2(v / n)) for v in c.values()) if n else 0.0

    def match_rule(rule: Dict[str, Any], *, src: str, src_lemma: str, src_pos: Optional[str], cand_lemma: str, cand_pos: Optional[str]) -> bool:
        r1 = rule.get("_src_re")
        r2 = rule.get("_cand_re")
        if r1 and not r1.search(src):
            return False
        if r2 and not r2.search(cand_lemma):
            return False
        if "src_lemma_prefix" in rule and not src_lemma.startswith(str(rule["src_lemma_prefix"]).lower()):
            return False
        if "candidate_lemma_prefix" in rule and not cand_lemma.startswith(str(rule["candidate_lemma_prefix"]).lower()):
            return False
        if "src_pos" in rule:
            sp = rule["src_pos"]
            if isinstance(sp, list):
                if src_pos not in sp:
                    return False
            else:
                if src_pos != sp:
                    return False
        if "candidate_pos" in rule:
            cp0 = rule["candidate_pos"]
            if isinstance(cp0, list):
                if cand_pos not in cp0:
                    return False
            else:
                if cand_pos != cp0:
                    return False
        if "require_src_chars" in rule:
            req = set(str(rule["require_src_chars"]).lower())
            if not req.issubset(set(src_lemma)):
                return False
        if "require_candidate_chars" in rule:
            req = set(str(rule["require_candidate_chars"]).lower())
            if not req.issubset(set(cand_lemma)):
                return False
        if "max_len_delta" in rule and abs(len(cand_lemma) - len(src_lemma)) > int(rule["max_len_delta"]):
            return False
        if "min_common_prefix" in rule and pref_len(cand_lemma, src_lemma) < int(rule["min_common_prefix"]):
            return False
        return True

    def is_allowed(src: str, src_lemma: str, src_pos: Optional[str], cand_lemma: str, cand_pos: Optional[str]) -> bool:
        return any(match_rule(r, src=src, src_lemma=src_lemma, src_pos=src_pos, cand_lemma=cand_lemma, cand_pos=cand_pos) for r in rule_allow)

    def is_blocked(src: str, src_lemma: str, src_pos: Optional[str], cand_lemma: str, cand_pos: Optional[str]) -> bool:
        return any(match_rule(r, src=src, src_lemma=src_lemma, src_pos=src_pos, cand_lemma=cand_lemma, cand_pos=cand_pos) for r in rule_block)

    token_freq = Counter()
    candidates: List[Tuple[int, int, str, str]] = []
    for m in _RU_WORD_RE.finditer(s):
        if len(candidates) >= spell_max_occurrences:
            break
        st, en = m.start(), m.end()
        w = m.group(0)
        lw0 = w.lower()
        if len(w) < spell_min_word_len or (w.isupper() and len(w) >= 2):
            continue
        if skip_titlecase and w[:1].isupper():
            continue
        if prefix_no_fix and any(lw0.startswith(px) for px in prefix_no_fix):
            continue
        if block_contains and any(sub in lw0 for sub in block_contains) and len(lw0) >= 10:
            continue
        token_freq[lw0] += 1
        candidates.append((st, en, w, lw0))

    auto_wl = {w for w, c in token_freq.items() if c >= auto_whitelist_min_count}
    auto_wl |= {w for w, c in token_freq.items() if c >= min_repeat_whitelist}

    cache_lemma_sugg: Dict[str, Optional[Tuple[str, int, int]]] = {}
    unknown_freq = Counter()
    suggested_freq = Counter()
    unknown_unique = set()
    suggested_unique = set()
    miss_positions: List[Dict[str, Any]] = []

    for st, en, w, lw0 in candidates:
        if lw0 in wl or lw0 in auto_wl:
            continue
        if w[:1].isupper() and not is_sentence_start(st):
            continue

        parses = _MORPH.parse(lw0)
        if not parses:
            continue
        p0 = parses[0]
        if p0.is_known:
            continue

        src_pos = p0.tag.POS
        lemma = (p0.normal_form or lw0).lower()

        if lemma in wl or lemma in auto_wl:
            continue
        if prefix_no_fix and any(lemma.startswith(px) for px in prefix_no_fix):
            continue
        if block_contains and any(sub in lemma for sub in block_contains) and len(lemma) >= 10:
            continue

        unknown_unique.add(lw0)
        unknown_freq[lw0] += 1

        if lemma not in cache_lemma_sugg:
            sugg = _SYMSPELL.lookup(lemma, Verbosity.TOP, max_edit_distance=max_edit_distance_lookup, include_unknown=True)
            cache_lemma_sugg[lemma] = None
            if sugg and str(sugg[0].term).lower() != lemma:
                best = sugg[0]
                cache_lemma_sugg[lemma] = (str(best.term).lower(), int(best.distance), int(best.count))

        x = cache_lemma_sugg[lemma]
        if not x:
            continue
        fixed_lemma, dist, cnt = x

        if fixed_lemma[:1] != lemma[:1]:
            continue
        if abs(len(fixed_lemma) - len(lemma)) > max_len_delta:
            continue
        if pref_len(fixed_lemma, lemma) < require_prefix_chars:
            continue
        if int(cnt) < min_candidate_count:
            continue

        pf = _MORPH.parse(fixed_lemma)
        if not pf:
            continue
        cand_pos = pf[0].tag.POS

        allow = is_allowed(w, lemma, src_pos, fixed_lemma, cand_pos)
        if (not allow) and is_blocked(w, lemma, src_pos, fixed_lemma, cand_pos):
            continue
        if require_same_pos and (not allow) and src_pos and cand_pos and src_pos != cand_pos:
            continue

        restored = restore_form(p0, fixed_lemma)
        if not restored:
            continue

        if w.isupper():
            restored = restored.upper()
        elif len(w) > 1 and w[0].isupper() and w[1:].islower():
            restored = restored[:1].upper() + restored[1:]

        if restored == w:
            continue

        rl = restored.lower()
        if rl in wl or rl in auto_wl:
            continue
        if prefix_no_fix and any(rl.startswith(px) for px in prefix_no_fix):
            continue

        miss_positions.append(
            {"word": w, "lower": lw0, "start": st, "end": en, "suggestion": restored,
             "lemma": lemma, "fixed_lemma": fixed_lemma, "distance": dist, "count": cnt}
        )
        suggested_unique.add(lw0)
        suggested_freq[lw0] += 1

    checked_unique = len({lw0 for _, _, _, lw0 in candidates if lw0 not in wl and lw0 not in auto_wl})
    unknown_cnt = len(unknown_unique)
    suggested_cnt = len(suggested_unique)

    misspelled_ratio = unknown_cnt / max(1, checked_unique)
    spell_coverage = (checked_unique - unknown_cnt) / max(1, checked_unique)

    metric_status = {
        "html_tag_ratio": "OK" if html_tag_ratio < html_tag_ratio_ok else "WARN",
        "empty_line_ratio": "OK" if empty_line_ratio < empty_line_ratio_warn else "WARN",
        "short_line_ratio": "OK" if short_line_ratio < short_line_ratio_warn else "WARN",
        "single_char_token_ratio": "OK" if single_char_token_ratio < single_char_token_fail else "FAIL",
        "misspelled_ratio": "OK" if misspelled_ratio < 0.25 else "WARN",
        "spell_coverage": "OK" if spell_coverage > 0.80 else "WARN",
    }

    penalty = sum(0.25 if v == "FAIL" else (0.10 if v == "WARN" else 0.0) for v in metric_status.values())
    quality_score = max(0.0, min(1.0, 1.0 - penalty))
    status = "FAIL" if any(v == "FAIL" for v in metric_status.values()) else ("PASS_WITH_WARNINGS" if any(v == "WARN" for v in metric_status.values()) else "PASS")

    regression_flags: List[Dict[str, Any]] = []
    if previous_report:
        def gp(a: str, b: str) -> Optional[float]:
            try:
                v = (previous_report.get(a, {}) or {}).get(b, None)
                return float(v) if v is not None else None
            except Exception:
                return None
        for metric, (sec, key) in {
            "misspelled_ratio": ("text_quality_metrics","misspelled_ratio"),
            "spell_coverage": ("text_quality_metrics","spell_coverage"),
            "misspelled_unique": ("text_quality_metrics","misspelled_unique"),
        }.items():
            prev = gp(sec, key)
            if prev is None:
                continue
            cur = {"misspelled_ratio": misspelled_ratio, "spell_coverage": spell_coverage, "misspelled_unique": unknown_cnt}[metric]
            regression_flags.append({"metric": metric, "previous": prev, "current": float(cur), "delta": float(cur) - float(prev)})

    def entropy_alnum(x: str) -> float:
        chars = _ALNUM_RE.findall(x or "")
        c = Counter(chars)
        n = sum(c.values())
        return sum((-(v / n) * math.log2(v / n)) for v in c.values()) if n else 0.0

    ent = float(entropy_alnum(s))

    return {
        "timestamp": ts,
        "global_metrics": {
            "total_chars": total_chars,
            "total_tokens": total_tokens,
            "html_tag_ratio": round(html_tag_ratio, 6),
            "empty_line_ratio": round(empty_line_ratio, 6),
            "short_line_ratio": round(short_line_ratio, 6),
        },
        "text_quality_metrics": {
            "single_char_token_ratio": round(single_char_token_ratio, 6),
            "avg_token_length": round(avg_token_length, 4),
            "entropy": round(ent, 4),
            "spell_coverage": round(spell_coverage, 6),
            "misspelled_ratio": round(misspelled_ratio, 6),
            "misspelled_unique": int(unknown_cnt),
            "suggested_unique": int(suggested_cnt),
        },
        "language_metrics": {
            "cyrillic_ratio": round(cyrillic_ratio, 6),
            "latin_russian_ratio": round(latin_russian_ratio, 6),
            "unknown_language_blocks": int(unknown_language_blocks),
        },
        "anomaly_samples": {
            "unknown_words": list(unknown_freq.keys())[:max_anomaly_samples],
            "suggested_words": list(suggested_freq.keys())[:max_anomaly_samples],
            "misspelled_words_with_pos": [
                {k: x[k] for k in ("word","start","end","suggestion","lemma","fixed_lemma","distance")}
                for x in miss_positions[:max_anomaly_samples]
            ],
        },
        "summary": {
            "quality_score": round(quality_score, 4),
            "status": status,
            "metric_status": metric_status,
        },
        "spellcheck": {
            "misspelled_positions": miss_positions,
            "unknown_word_counts": dict(unknown_freq),
            "suggested_word_counts": dict(suggested_freq),
            "whitelist_size": int(len(wl)),
            "applied_replacements": [],
            "applied_count": 0,
            "config_path": str(config_path),
        },
        "symspell": {
            "dictionary_path": _DICT_PATH.as_posix() if _DICT_PATH else None,
            "max_edit_distance_lookup": int(max_edit_distance_lookup),
            "min_candidate_count": int(min_candidate_count),
        },
        "regression_flags": regression_flags,
    }


def spell_fix(
    text: str,
    report: Dict[str, Any],
    *,
    repair_markdown_tables: bool = True,
    max_fixes: Optional[int] = None,
) -> str:
    s = (text or "").replace("\r\n", "\n").replace("\r", "\n")

    if repair_markdown_tables:
        lines = s.split("\n")
        out: List[str] = []
        buf: List[str] = []
        in_table = False

        def is_table_line(ln: str) -> bool:
            t = ln.strip()
            if "|" not in t:
                return False
            if re.fullmatch(r"\|?\s*:?[-]{2,}:?\s*(\|\s*:?[-]{2,}:?\s*)+\|?\s*", t):
                return True
            return t.count("|") >= 2

        def flush_buf():
            nonlocal buf, out
            if not buf:
                return
            block = "\n".join(buf)
            try:
                out.append(str(mtr.repair(block)))
            except Exception:
                out.extend(buf)
            buf = []

        for ln in lines:
            tl = ln.strip()
            start = is_table_line(ln)
            if start:
                if not in_table:
                    in_table = True
                buf.append(ln)
            else:
                if in_table:
                    flush_buf()
                    in_table = False
                out.append(ln)

            if in_table and (tl.startswith("#") or tl.startswith("```")):
                flush_buf()
                in_table = False

        if in_table:
            flush_buf()

        s = "\n".join(out)

    items = (((report or {}).get("spellcheck") or {}).get("misspelled_positions") or [])
    patches: List[Tuple[int, int, str, str]] = []
    applied_replacements: List[Dict[str, str]] = []
    applied = 0

    for it in items:
        if max_fixes is not None and applied >= max_fixes:
            break
        st, en = int(it.get("start", -1)), int(it.get("end", -1))
        old, new = str(it.get("word", "")), str(it.get("suggestion", ""))
        if st < 0 or en <= st or en > len(s):
            continue
        if not new or new == old:
            continue
        if s[st:en] != old:
            continue
        patches.append((st, en, new, old))
        applied_replacements.append({old: new})
        applied += 1

    patches.sort(key=lambda x: x[0], reverse=True)
    for st, en, new, _old in patches:
        s = s[:st] + new + s[en:]

    if isinstance(report, dict):
        sc = report.setdefault("spellcheck", {})
        sc["applied_replacements"] = applied_replacements
        sc["applied_count"] = int(applied)

    return s


def summarize_reports(
    report: Dict[str, Any],
    *,
    out_json_path: Optional[str] = None,
) -> Dict[str, Any]:
    st = getattr(summarize_reports, "_state", None)
    if st is None:
        st = {
            "replacements": [],
            "status_cnt": Counter(),
            "n": 0,
            "total_chars": 0,
            "total_tokens": 0,
            "sum_html": 0.0,
            "sum_empty": 0.0,
            "sum_short": 0.0,
            "sum_single_char": 0.0,
            "sum_avg_tok_len": 0.0,
            "sum_entropy": 0.0,
            "sum_spell_cov": 0.0,
            "sum_miss_ratio": 0.0,
            "sum_miss_unique": 0.0,
            "sum_suggested_unique": 0.0,
            "sum_cyr": 0.0,
            "sum_lat_ru": 0.0,
            "sum_unknown_blocks": 0.0,
            "sum_quality": 0.0,
        }
        setattr(summarize_reports, "_state", st)

    st["n"] += 1

    gm = report.get("global_metrics") or {}
    tq = report.get("text_quality_metrics") or {}
    lm = report.get("language_metrics") or {}
    sm = report.get("summary") or {}
    sc = report.get("spellcheck") or {}

    st["total_chars"] += int(gm.get("total_chars") or 0)
    st["total_tokens"] += int(gm.get("total_tokens") or 0)

    reps = sc.get("applied_replacements") or []
    if isinstance(reps, list) and reps:
        st["replacements"].extend(reps)

    st["sum_html"] += float(gm.get("html_tag_ratio") or 0.0)
    st["sum_empty"] += float(gm.get("empty_line_ratio") or 0.0)
    st["sum_short"] += float(gm.get("short_line_ratio") or 0.0)

    st["sum_single_char"] += float(tq.get("single_char_token_ratio") or 0.0)
    st["sum_avg_tok_len"] += float(tq.get("avg_token_length") or 0.0)
    st["sum_entropy"] += float(tq.get("entropy") or 0.0)
    st["sum_spell_cov"] += float(tq.get("spell_coverage") or 0.0)
    st["sum_miss_ratio"] += float(tq.get("misspelled_ratio") or 0.0)
    st["sum_miss_unique"] += float(tq.get("misspelled_unique") or 0.0)
    st["sum_suggested_unique"] += float(tq.get("suggested_unique") or 0.0)

    st["sum_cyr"] += float(lm.get("cyrillic_ratio") or 0.0)
    st["sum_lat_ru"] += float(lm.get("latin_russian_ratio") or 0.0)
    st["sum_unknown_blocks"] += float(lm.get("unknown_language_blocks") or 0.0)

    st["sum_quality"] += float(sm.get("quality_score") or 0.0)
    st["status_cnt"][str(sm.get("status") or "UNKNOWN")] += 1

    denom = max(1, int(st["n"]))

    summary = {
        "chunks": int(st["n"]),
        "global_metrics": {
            "total_chars": int(st["total_chars"]),
            "total_tokens": int(st["total_tokens"]),
            "avg_html_tag_ratio": round(st["sum_html"] / denom, 6),
            "avg_empty_line_ratio": round(st["sum_empty"] / denom, 6),
            "avg_short_line_ratio": round(st["sum_short"] / denom, 6),
        },
        "text_quality_metrics": {
            "avg_single_char_token_ratio": round(st["sum_single_char"] / denom, 6),
            "avg_avg_token_length": round(st["sum_avg_tok_len"] / denom, 4),
            "avg_entropy": round(st["sum_entropy"] / denom, 4),
            "avg_spell_coverage": round(st["sum_spell_cov"] / denom, 6),
            "avg_misspelled_ratio": round(st["sum_miss_ratio"] / denom, 6),
            "avg_misspelled_unique": round(st["sum_miss_unique"] / denom, 3),
            "avg_suggested_unique": round(st["sum_suggested_unique"] / denom, 3),
        },
        "language_metrics": {
            "avg_cyrillic_ratio": round(st["sum_cyr"] / denom, 6),
            "avg_latin_russian_ratio": round(st["sum_lat_ru"] / denom, 6),
            "avg_unknown_language_blocks": round(st["sum_unknown_blocks"] / denom, 3),
        },
        "summary": {
            "avg_quality_score": round(st["sum_quality"] / denom, 4),
            "status_counts": dict(st["status_cnt"]),
        },
        "spellcheck": {
            "total_applied_replacements": int(len(st["replacements"])),
        },
    }

    if out_json_path:
        out_path = Path(out_json_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = list(st["replacements"])
        payload.append({"_summary": summary})
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"summary": summary, "total_replacements": len(st["replacements"])}


async def _ocr_pdf_bytes_to_pages(content: bytes, doc_name: str, total_pages: int) -> List[Dict[str, Any]]:
    target_dir = OUTPUT_DIR / "mistral" / doc_name / "images"
    target_dir.mkdir(parents=True, exist_ok=True)

    timeout = aiohttp.ClientTimeout(total=3600)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            convert_and_ocr_page(session, semaphore, content, page_num, target_dir, total_pages)
            for page_num in range(1, total_pages + 1)
        ]
        results = await asyncio.gather(*tasks)

    results.sort(key=lambda x: x[0])
    return [{"index": page_num - 1, "markdown": text} for page_num, text in results]


async def _ocr_image_bytes_to_pages(content: bytes) -> List[Dict[str, Any]]:
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    b64_image = preprocess_image_for_ocr(image)
    image.close()

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "OCR this page. Output only text."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                ],
            }
        ],
        "max_tokens": 2048,
    }

    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(VLLM_URL, json=payload) as resp:
            if resp.status != 200:
                err_text = await resp.text()
                raise HTTPException(status_code=502, detail=f"vLLM OCR failed: {err_text}")
            result = await resp.json()

    text = (result.get("choices") or [{}])[0].get("message", {}).get("content")
    return [{"index": 0, "markdown": text or ""}]


@app.post("/v1/files")
async def v1_files_create(
    authorization: Optional[str] = Header(default=None),
    purpose: str = Form(default="ocr"),
    file: UploadFile = File(...),
):
    """Minimal subset of Mistral's file upload API used by OpenWebUI."""
    # Auth intentionally not enforced here.
    if purpose and purpose != "ocr":
        raise HTTPException(status_code=400, detail="Unsupported purpose")

    _prune_store()
    file_id = secrets.token_urlsafe(18)
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    _file_path(file_id).write_bytes(data)
    _FILE_INDEX[file_id] = {
        "id": file_id,
        "filename": file.filename,
        "content_type": file.content_type or "application/octet-stream",
        "size": len(data),
        "created_at": _now(),
    }

    return {"id": file_id, "object": "file", "filename": file.filename}


@app.get("/v1/files/{file_id}/url")
async def v1_files_url(
    file_id: str,
    authorization: Optional[str] = Header(default=None),
    expiry: Optional[int] = None,
):
    if file_id not in _FILE_INDEX:
        raise HTTPException(status_code=404, detail="File not found")
    return {"url": f"local://{file_id}"}


@app.delete("/v1/files/{file_id}")
async def v1_files_delete(
    file_id: str,
    authorization: Optional[str] = Header(default=None),
):
    existed = file_id in _FILE_INDEX
    _delete_file_local(file_id)
    return {"id": file_id, "deleted": existed}


@app.post("/v1/ocr")
async def v1_ocr(
    request: Request,
    authorization: Optional[str] = Header(default=None),
):
    payload = await request.json()
    document = payload.get("document") or {}
    doc_url = document.get("document_url")
    if not doc_url or not isinstance(doc_url, str):
        raise HTTPException(status_code=400, detail="Missing document.document_url")

    if not doc_url.startswith("local://"):
        raise HTTPException(status_code=400, detail="Only local:// URLs are supported")

    file_id = doc_url[len("local://") :]
    meta = _FILE_INDEX.get(file_id)
    if not meta:
        raise HTTPException(status_code=404, detail="File not found")

    content = _file_path(file_id).read_bytes()
    filename = meta.get("filename") or "uploaded"
    content_type = meta.get("content_type") or "application/octet-stream"

    t0 = time.time()
    if _is_pdf(filename, content_type):
        try:
            reader = PdfReader(io.BytesIO(content))
            total_pages = len(reader.pages)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid PDF: {e}")

        pages = await _ocr_pdf_bytes_to_pages(content, doc_name=file_id, total_pages=total_pages)
    else:
        pages = await _ocr_image_bytes_to_pages(content)

    return JSONResponse(
        content={
            "pages": pages,
            "model": payload.get("model") or "local-ocr",
            "usage": {},
        },
        headers={"X-OCR-Ms": str(int((time.time() - t0) * 1000))},
    )

def preprocess_image_for_ocr(pil_image):
    img = np.array(pil_image)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), SAVE_QUALITY])
    return base64.b64encode(buffer).decode('utf-8')

def convert_page_sync(content, page_num, target_dir):
    """Sync page conversion"""
    start = time.time()
    try:
        page_images = convert_from_bytes(
            content, dpi=PDF_DPI, fmt="jpeg",
            first_page=page_num, last_page=page_num
        )
        
        if page_images:
            img = page_images[0]
            img_path = target_dir / f'image_{page_num-1:04d}.jpeg'
            img.save(str(img_path), "JPEG", quality=SAVE_QUALITY)
            img.close()
            elapsed = time.time() - start
            print(f'[Page {page_num}] Conversion took {elapsed:.2f}s', flush=True)
            return img_path
        return None
    except Exception as e:
        logger.error(f"Error converting page {page_num}: {e}")
        return None

async def process_single_page(session, semaphore, img_path, page_num):
    """Send single page to vLLM"""
    async with semaphore:
        start = time.time()
        try:
            prep_start = time.time()
            with Image.open(img_path) as img:
                b64_image = preprocess_image_for_ocr(img)
            print(f'[Page {page_num}] Image preprocessing took {time.time()-prep_start:.2f}s', flush=True)
            
            payload = {
                "model": MODEL_NAME,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "OCR this page. Output only text."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                    ]
                }],
                "max_tokens": 2048
            }

            req_start = time.time()
            async with session.post(VLLM_URL, json=payload) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    elapsed = time.time() - start
                    req_time = time.time() - req_start
                    text = result['choices'][0]['message']['content']
                    print(f'[Page {page_num}] ✓ OCR request took {req_time:.2f}s, total OCR: {elapsed:.2f}s', flush=True)
                    report = build_report(text)
                    clean_text = text_cleaner.clean_text(text)
                    corrected_text = spell_fix(clean_text, report)
                    print("📄 OCR Quality Report")
                    pprint.pprint(build_report(corrected_text, report))
                    summary = summarize_reports(report, out_json_path="./ocr_stats.json")
                    return (page_num, corrected_text)
                else:
                    err_text = await resp.text()
                    return (page_num, f"[Error page {page_num}: {err_text}]")
        except Exception as e:
            elapsed = time.time() - start
            print(f'[Page {page_num}] ✗ Exception after {elapsed:.2f}s: {str(e)}', flush=True)
            return (page_num, f"[Exception page {page_num}: {str(e)}]")

async def convert_and_ocr_page(session, semaphore, content, page_num, target_dir, total_pages):
    """Convert page then sand to OCR"""
    page_start = time.time()
    loop = asyncio.get_event_loop()
    
    print(f'[Page {page_num}/{total_pages}] ⏳ Starting conversion...', flush=True)
    conv_start = time.time()
    img_path = await loop.run_in_executor(
        executor, 
        convert_page_sync, 
        content, 
        page_num, 
        target_dir
    )
    conv_time = time.time() - conv_start
    
    if img_path is None:
        return (page_num, f"[Error: Failed to convert page {page_num}]")
    
    # 2. OCR
    print(f'[Page {page_num}/{total_pages}] 🚀 Converted in {conv_time:.2f}s, sending to OCR...', flush=True)
    ocr_start = time.time()
    result = await process_single_page(session, semaphore, img_path, page_num)
    ocr_time = time.time() - ocr_start
    
    total_time = time.time() - page_start
    print(f'[Page {page_num}/{total_pages}] ✅ DONE in {total_time:.2f}s (conv: {conv_time:.2f}s, ocr: {ocr_time:.2f}s)', flush=True)
    return result

@app.post("/ocr")
async def ocr_pdf(file: UploadFile = File(...)):
    overall_start = time.time()
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    content = await file.read()
    filename_stem = Path(file.filename).stem
    target_dir = OUTPUT_DIR / filename_stem / "images"
    target_dir.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(io.BytesIO(content))
    total_pages = len(reader.pages)
    
    print(f'\n{"="*80}')
    print(f'📄 PDF: {file.filename} | Pages: {total_pages}')
    print(f'🔧 Max concurrent OCR requests: {MAX_CONCURRENT_REQUESTS}')
    print(f'{"="*80}\n', flush=True)

    timeout = aiohttp.ClientTimeout(total=3600)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    tasks_start = time.time()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            convert_and_ocr_page(session, semaphore, content, page_num, target_dir, total_pages)
            for page_num in range(1, total_pages + 1)
        ]
        
        print(f'⚡ All {len(tasks)} tasks created in {time.time()-tasks_start:.2f}s, starting parallel execution...\n', flush=True)
        
        gather_start = time.time()
        results = await asyncio.gather(*tasks)
        gather_time = time.time() - gather_start

    results.sort(key=lambda x: x[0])
    full_text = "\n\n--- Page Break ---\n\n".join([text for _, text in results])
    
    result_path = OUTPUT_DIR / filename_stem / "result.txt"
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    total_time = time.time() - overall_start
    print(f'\n{"="*80}')
    print(f'✅ ALL DONE in {total_time:.2f}s')
    print(f'   - Parallel execution: {gather_time:.2f}s')
    print(f'   - Average per page: {total_time/total_pages:.2f}s')
    print(f'{"="*80}\n', flush=True)
    
    return {"filename": file.filename, "text": full_text, "processing_time": total_time}

@app.on_event("shutdown")
async def shutdown_event():
    executor.shutdown(wait=True)