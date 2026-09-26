import os
import re
import time
from typing import Dict, Any, List, Tuple
import requests

WIKI_HEADERS = {"User-Agent": "QuarkLlamaBot/1.0 (contact@quarkllama.ai)"}


def clean_topic_query(text: str) -> str:
    """Strips conversational fluff to identify core subject for search."""
    cleaned = re.sub(r'[^\w\s]', ' ', text)
    stop_phrases = [
        r'\bexplain\b', r'\bin simple words\b', r'\bfor a \d+ year old child\b',
        r'\bfor a child\b', r'\bfor kids\b', r'\bfor beginners\b',
        r'\bwhat is the meaning of\b', r'\bwhat is a\b', r'\bwhat is an\b', r'\bwhat is\b',
        r'\bwhat are\b', r'\bwhat was\b', r'\bwho was\b', r'\bwho is\b',
        r'\btell me about\b', r'\bhow does\b', r'\bhow do\b', r'\bwhy is\b',
        r'\bgive me \d+ tips to\b', r'\bgive me tips to\b', r'\bcan you\b',
        r'\bplease\b', r'\bdefine\b', r'\bmeaning of\b'
    ]
    q = cleaned
    for p in stop_phrases:
        q = re.sub(p, '', q, flags=re.IGNORECASE)
    stop_words = {'a', 'an', 'the', 'in', 'on', 'at', 'of', 'for', 'to', 'and', 'is', 'are', 'was'}
    words = [w for w in q.split() if w.lower() not in stop_words]
    core = ' '.join(words).strip()
    return core if len(core) >= 2 else text.strip()


def search_with_tavily(query: str, api_key: str) -> Dict[str, Any]:
    """Execute high-speed multi-website search via Tavily API."""
    from tavily import TavilyClient

    start_time = time.time()
    client = TavilyClient(api_key=api_key)
    res = client.search(
        query=query,
        search_depth="basic",
        max_results=3,
        include_answer=True
    )
    elapsed = int((time.time() - start_time) * 1000)

    answer = res.get("answer", "")
    results = res.get("results", [])
    
    sources = []
    combined_content = []
    for r in results:
        title = r.get("title", "Source")
        url = r.get("url", "")
        content = r.get("content", "").strip()
        sources.append({"title": title, "url": url})
        if content:
            combined_content.append(content)

    if not answer and combined_content:
        answer = " ".join(combined_content[:2])

    if len(answer) > 400:
        answer = answer[:400].rsplit(".", 1)[0] + "."

    return {
        "engine": "Tavily AI Search",
        "answer": answer.strip(),
        "sources": sources,
        "latency_ms": elapsed,
        "success": True
    }


def search_with_wikipedia(query: str) -> Dict[str, Any]:
    """Fallback search using Wikipedia REST & Opensearch APIs."""
    start_time = time.time()
    q_lower = query.lower()
    sources = []
    answer = ""

    # Check for comparison
    comp_match = re.search(r'(?:difference between|compare|vs|versus) (?:a |an )?(\w+)(?: and (?:a |an )?(\w+))?', q_lower)
    if comp_match:
        t1 = comp_match.group(1)
        t2 = comp_match.group(2) if comp_match.group(2) else ""
        s1 = _get_wiki_extract(t1)
        s2 = _get_wiki_extract(t2) if t2 else ""
        if s1 or s2:
            part1 = s1[:180].rsplit('.', 1)[0] + '.' if s1 else ""
            part2 = s2[:180].rsplit('.', 1)[0] + '.' if s2 else ""
            answer = f"⭐ [{t1.capitalize()}]: {part1}\n🪐 [{t2.capitalize()}]: {part2}"
            sources = [
                {"title": f"Wikipedia: {t1.capitalize()}", "url": f"https://en.wikipedia.org/wiki/{t1}"},
                {"title": f"Wikipedia: {t2.capitalize()}", "url": f"https://en.wikipedia.org/wiki/{t2}"}
            ]

    if not answer:
        topic = clean_topic_query(query)
        title = _search_wiki_title(topic) or _search_wiki_title(query)
        if title:
            raw = _get_wiki_extract(title)
            if raw:
                answer = raw[:350].rsplit('.', 1)[0] + '.' if len(raw) > 350 else raw
                sources = [{"title": f"Wikipedia: {title}", "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"}]

    elapsed = int((time.time() - start_time) * 1000)
    return {
        "engine": "Wikipedia Knowledge",
        "answer": answer.strip(),
        "sources": sources,
        "latency_ms": elapsed,
        "success": bool(answer)
    }


def _search_wiki_title(query: str) -> str:
    try:
        url = "https://en.wikipedia.org/w/api.php"
        params = {"action": "opensearch", "search": query.strip(), "limit": 3, "namespace": 0, "format": "json"}
        r = requests.get(url, params=params, headers=WIKI_HEADERS, timeout=3).json()
        if r and len(r) > 1 and r[1]:
            return r[1][0]
    except Exception:
        pass
    return ""


def _get_wiki_extract(title: str) -> str:
    if not title:
        return ""
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}"
        r = requests.get(url, headers=WIKI_HEADERS, timeout=3)
        if r.status_code == 200:
            return r.json().get("extract", "")
    except Exception:
        pass
    return ""


def get_web_information(query: str, tavily_api_key: str = None) -> Dict[str, Any]:
    """Tries Tavily search first if an API key is available, else uses Wikipedia."""
    key = tavily_api_key or os.environ.get("TAVILY_API_KEY", "").strip()
    if key:
        try:
            res = search_with_tavily(query, key)
            if res.get("success") and res.get("answer"):
                return res
        except Exception as e:
            # Fall through to Wikipedia if Tavily has an issue (e.g. invalid key/quota)
            pass

    return search_with_wikipedia(query)
