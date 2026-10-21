"""08_reproduce_geo_bench.py — 复现 Aggarwal et al. (KDD 2024) 的 9 个 GEO 方法

这个脚本回答两个问题：
    **"给商品详情页加上 GEO 方法，AI 回答里这个来源的可见度会不会变高？"**
    **"这个提升，在统计上站得住吗？"**

论文与代码的对应关系（全部已逐条核对）：
    论文 arXiv:2311.09735 第 2.2.2 节   → 9 个方法的名称、编号、定义
    官方仓库 github.com/GEO-optim/GEO
      src/geo_functions.py               → 9 个方法**逐字的改写 prompt**（本脚本已内置中英双版）
      src/generative_le.py               → 生成式引擎的 query_prompt（本脚本逐字采用）
      src/utils.py                       → 主指标 impression_wordpos_count_simple（本脚本复刻公式）
      src/run_geo.py                     → 每个条件采样 n=5 次取均值

实验臂（10 个）：
    identity（baseline，不改写）
    + 论文第 2.2.2 节的 9 个方法，编号与原文一致：
      1 Authoritative          2 Statistics Addition   3 Keyword Stuffing
      4 Cite Sources          5 Quotation Addition     6 Easy-to-Understand
      7 Fluency Optimization  8 Unique Words           9 Technical Terms

主指标：`geo_impression` = 论文 impression_wordpos_count_simple 里目标来源的归一化份额
    （按句统计引用 → 按句子位置做 exp 衰减加权 → 按被引来源归一化）
辅指标：`cited`（目标编号是否被引用）、`mentioned`（品牌是否出现，噪声大，仅参考）
统计：均值 + 95% CI + 对 baseline 的 Welch t 检验；辅指标另给 Wilson 区间 + 双比例 z 检验

⚠️ 三件你必须先知道的事（否则会误读结果）：
    1. 官方改写 prompt **明确要求编造来源**：
       Cite Sources "You may invent these sources"；Quotation "even though fake and
       artificial"；Statistics "even if hypothetical"。论文那 40% 有一部分是靠塞
       虚构引用/假引语/假数据拿到的。这既是落地风险，也是现成的可证伪研究问题。
    2. 论文正文说 Keyword Stuffing 的关键词"from the query"，但官方 prompt 根本没拿到
       query。本脚本默认忠于代码；--keyword-stuffing-use-query 可切到忠于正文。
    3. 论文正文写采样 temperature=0.7，官方 generative_le.py 的默认值是 0.5。
       本脚本默认 0.7（忠于正文），--temperature 0.5 可切到忠于代码。

本 harness 诚实地不做什么：
    1. 测的是「给定候选检索结果之后」的可见性，不是端到端 GEO（真实效果还取决于能否被检索到）。
    2. 改写模型不同：论文用 gpt-3.5-turbo-16k，这里默认用你自己的 qwen-plus。
    3. 源文档是中文，论文是英文。改写 prompt 默认用中文翻译版（--prompt-lang en 可切英文原文）；
       指标里的词数统计也按中文重新定义（见 _num_words 的注释）。
    4. 竞争设定理想化：对手只从同类目取，且全场只有 1 个文档被优化（C-SEO Bench 证明真实场景是零和的）。
    5. 位置偏置做了反平衡（目标位置随 repeat 轮换），但没做到 arXiv:2605.25517 那种
       单因子配对 + mixed-effects 的严格程度。
    6. 不做品牌体量分层，绝对数值**不能和论文横向比**。想看体量偏置用 --synthetic。

用法：
    # 0) 不花一分钱：导出官方 prompt 供人工核对
    python examples/08_reproduce_geo_bench.py --dry-run --limit 1

    # 1) ★ 最接近论文原设置：官方 GEO-bench 数据集（对手是真实检索结果）
    python examples/08_reproduce_geo_bench.py --dataset geobench --limit 8 \
        --out outputs/08_geobench

    # 2) 中文美妆场景（自己的数据，跑"改写策略"这套东西在电商上的效果）
    python examples/08_reproduce_geo_bench.py --dataset geoagent --limit 8

    # 3) 零成本冒烟测试（规则版，不是论文方法复现，只验证链路通不通）
    python examples/08_reproduce_geo_bench.py --rewriter rule --limit 4 --repeats 2

    # 4) 换引擎 / 虚构品牌对照
    python examples/08_reproduce_geo_bench.py --engine deepseek --model deepseek-chat
    python examples/08_reproduce_geo_bench.py --synthetic

两种数据源：
    --dataset geobench  官方 GEO-Optim/geo-bench（英文，CC-BY-SA-4.0，10,000 query）
        每条样本自带 5 个真实检索来源 + sugg_idx（该优化哪个来源）——这就是论文的竞争设定，
        对手不是我们造的。首次运行会从 hf-mirror.com 流式拉取前 N 条并缓存到 data/geo-bench/。
        注意 prompt 语言会自动切成英文官方原文。
    --dataset geoagent  geo-agent 的中文美妆 query/SKU，对手从同类目里取（默认）
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs" / "08_geo_bench"

# geo-agent 仓库的候选位置（自动探测，也可 --data-root 或环境变量 GEO_AGENT_ROOT 指定）
# 🔴 2026-10-21 开源化：删掉了一条**指向作者本机 session 工作区的死路径** ——
#    那个目录换台机器就不存在，留着只会让人以为「路径写错了」。
GEO_AGENT_CANDIDATES = [
    ROOT / "geo-agent",
    ROOT.parent / "geo-agent",
]

# 引擎配置：全部走 OpenAI 兼容协议
ENGINES: dict[str, dict[str, Any]] = {
    "qwen": {
        "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model": "qwen-plus",
        "env_key": "DASHSCOPE_API_KEY",
        "label": "Qwen (DashScope)",
    },
    "deepseek": {
        "url": "https://api.deepseek.com/v1/chat/completions",
        "model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
        "label": "DeepSeek",
    },
    "openai": {
        "url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
        "label": "OpenAI",
    },
    "gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "model": "gemini-2.5-flash",
        "env_key": "GEMINI_API_KEY",
        "label": "Gemini (OpenAI 兼容层)",
    },
    # 🔴 2026-10-06：换引擎到 MiniMax。三个实测决定了这个配置：
    #   1) **只有 M3 可用**。M2.7 / M2.7-highspeed 的 thinking 关不掉
    #      （`thinking:{type:disabled}` 被忽略），`<think>` 会**内联进正文**，
    #      7e2 字符的响应里 thinking 就吃满 max_tokens 并 `finish=length` 截断 ——
    #      引用提取直接失效。M3.1-Flash-Preview 强制 adaptive，拒 disabled（400）。
    #   2) 必须显式关 thinking，否则 <think> 污染 [N] 引用解析。
    #      关闭后实测 `reasoning_tokens: 0`、`finish=stop`、引用正常。
    #   3) `seed` **被接受但无效**（同 seed 两次调用输出不同）。
    #      好处：重复测量是真正的 i.i.d. 抽样，不被 seed 部分钉住；
    #      代价：§4 的「seed 可复现」在 M3 上不成立，必须如实声明。
    "minimax": {
        "url": "https://api.minimaxi.com/v1/chat/completions",
        "model": "MiniMax-M3",
        "env_key": "MINIMAX_API_KEY",
        "label": "MiniMax M3 (OpenAI 兼容层)",
        "extra_body": {"thinking": {"type": "disabled"}},
        "seed_effective": False,
    },
}

# ---------------------------------------------------------------- 生成式引擎 prompt
# 逐字取自官方实现 src/generative_le.py 的 query_prompt（Aggarwal 2023 官方仓库）。
# 注意原文几个细节，不要"顺手优化"：
#   - 官方调用时不带 system prompt（代码里 system_prompt 被注释掉了），只有一个 user 消息
#   - 引用格式是 [1][2][3]（各自独立方括号），不是 [1, 2, 3]
#   - 来源块的标题是 "### Source N:"
#   - 原文要求"每一句后面都要紧跟引用"，这正是论文指标能按句统计的前提
GE_QUERY_PROMPT = (
    "Write an accurate and concise answer for the given user question, using _only_ the provided "
    "summarized web search results. The answer should be correct, high-quality, and written by an "
    "expert using an unbiased and journalistic tone. The user's language of choice such as English, "
    "Français, Español, Deutsch, or 日本語 should be used. The answer should be informative, interesting, "
    "and engaging. The answer's logic and reasoning should be rigorous and defensible. Every sentence "
    "in the answer should be _immediately followed_ by an in-line citation to the search result(s). The "
    "cited search result(s) should fully support _all_ the information in the sentence. Search results "
    "need to be cited using [index]. When citing several search results, use [1][2][3] format rather "
    "than [1, 2, 3]. You can use multiple search results to respond comprehensively while avoiding "
    "irrelevant search results.\n\n"
    "Question: {query}\n\n"
    "Search Results:\n{source_text}\n"
)

# 官方 GEO 改写器的 system prompt，逐字取自 src/geo_functions.py 的 COMMON_SYSTEM_PROMPT
REWRITER_SYSTEM_EN = (
    "You are an expert ml researcher having previous background in SEO and search engines in general. "
    "You are working on novel research ideas for next generation of products. These products will have "
    "language models augmented with search engines, with the task of answering questions based on sources "
    "backed by the search engine. This new set of systems will be collectively called language engines "
    "(generative search engines). This will require websites to update their SEO techniques to rank higher "
    "in the llm generated answer. Specifically they will use GEO (Generative Engine Optimization) techniques "
    "to boost their visibility in the final text answer outputted by the Language Engine."
)

REWRITER_SYSTEM_ZH = (
    "你是一位有 SEO 与搜索引擎研究背景的机器学习研究者，正在为下一代产品做研究设想。"
    "这类产品把语言模型与搜索引擎结合，用检索到的资料回答问题，"
    "这类系统被称为语言引擎（生成式搜索引擎）。这要求网站更新 SEO 手段，"
    "以便在语言引擎生成的答案里排得更靠前；具体手段就是 GEO（生成式引擎优化）。"
)

CITATION_RE = re.compile(r"\[\s*\d+\s*\]|来源[：:]|参考(?:资料|文献)?[：:]|据.{0,12}(?:报告|标准|研究|数据)")

# 论文主指标 impression_wordpos_count_simple 用的引用解析
# 逐字取自官方 src/utils.py 的 extract_citations_new
CITATION_NUM_RE = re.compile(r"\[\s*([^\]\s]*\d+[^\]\s]*)\s*\]")


# 顺序 = Aggarwal 2023 第 2.2.2 节的方法编号 1..9
STRATEGY_ORDER = [
    "authoritative",        # 1. Authoritative
    "statistics_addition",  # 2. Statistics Addition
    "keyword_stuffing",     # 3. Keyword Stuffing
    "cite_sources",         # 4. Cite Sources
    "quotation_addition",  # 5. Quotation Addition
    "easy_to_understand",   # 6. Easy-to-Understand
    "fluency_optimization", # 7. Fluency Optimization
    "unique_words",         # 8. Unique Words
    "technical_terms",      # 9. Technical Terms
]


# ---------------------------------------------------------------- 数据与文档构建

def _read_csv(path: Path) -> list:
    return list(csv.DictReader(path.read_bytes().decode("utf-8-sig").splitlines()))


# ---------------------------------------------------------------- GEO-bench 官方数据集
# 官方数据集：HuggingFace GEO-Optim/geo-bench（CC-BY-SA-4.0，10,000 条英文 query）
# 字段：query / tags / sources(5 条，每条含 raw_text + url + cleaned_text) / sugg_idx
# sugg_idx 官方代码当 0-based 用（run_geo.py: summaries[idx]），见 --sugg-idx-base
GEO_BENCH_URLS = [
    "https://hf-mirror.com/datasets/GEO-Optim/geo-bench/resolve/main/test.jsonl",
    "https://huggingface.co/datasets/GEO-Optim/geo-bench/resolve/main/test.jsonl",
]
GEO_BENCH_CACHE = ROOT / "data" / "geo-bench"


def _domain_label(url: str) -> str:
    """从 URL 取一个有辨识度的标签（en.wikipedia.org → wikipedia，firstcry.com → firstcry）"""
    m = re.match(r"https?://([^/]+)", url or "")
    if not m:
        return ""
    host = m.group(1).lower()
    if host.startswith("www."):
        host = host[4:]
    parts = host.split(".")
    if len(parts) >= 3:
        return parts[-2]
    return parts[0] if parts else ""


def load_geobench(limit: int, cache_dir: Path, url: str = "", text_field: str = "cleaned_text",
                  max_chars: int = 4000) -> tuple:
    """按需读取 GEO-bench test.jsonl 的前 limit 条，落到本地缓存。

    为什么不用 HTTP Range 断点续读：实测 hf-mirror.com 会直接重置带 Range 的连接，
    而且服务端忽略 Range 时字节偏移会算错、往缓存里灌重复数据。
    改成**永远从 0 开始流式读、跳过已缓存的行**更稳：20 条样本的前缀也就 ~2 MB，
    重读成本可以忽略，但少了一整类"缓存里有重复/半行"的 bug。

    返回 (样本列表, 缓存目录, 实际使用的 url)
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / "test.head.jsonl"
    cached: list = []
    if cache.exists():
        for l in cache.read_text(encoding="utf-8").splitlines():
            if l.strip():
                try:
                    cached.append(json.loads(l))
                except json.JSONDecodeError:
                    break  # 文件尾部可能有半行，丢弃
    if len(cached) >= limit:
        print(f"[+] GEO-bench 缓存命中：{cache}（{len(cached)} 条）")
        return cached[:limit], cache_dir, url or GEO_BENCH_URLS[0]

    need = limit - len(cached)
    new: list = []
    last_err = None
    for u in ([url] if url else GEO_BENCH_URLS):
        for attempt in range(3):
            try:
                req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
                buf = b""
                seen = 0  # 已从流里读过的完整样本数，用来跳过已缓存的前缀
                with urllib.request.urlopen(req, timeout=180) as r:
                    while len(new) < need:
                        chunk = r.read(1 << 16)
                        if not chunk:
                            break
                        buf += chunk
                        parts = buf.split(b"\n")
                        buf = parts.pop()  # 保留半行，等下一块
                        for ln in parts:
                            if not ln.strip():
                                continue
                            seen += 1
                            if seen <= len(cached):
                                continue  # 已在缓存里，跳过
                            new.append(json.loads(ln.decode("utf-8", "ignore")))
                            if len(new) >= need:
                                break
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                new = []
                print(f"[!] 拉取 {u} 第 {attempt + 1} 次失败：{e}")
                time.sleep(1.5 * (attempt + 1))
        if len(new) >= need:
            break

    if len(new) < need:
        raise RuntimeError(
            f"只读到 {len(cached) + len(new)}/{limit} 条 GEO-bench 样本。最后一个错误：{last_err}\n"
            f"    可以先离线下载再指过来：\n"
            f"    https://hf-mirror.com/datasets/GEO-Optim/geo-bench/resolve/main/test.jsonl\n"
            f"    然后 --geobench-url file:///E:/path/to/test.jsonl"
        )

    records = cached + new
    cache.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8"
    )
    size_mb = cache.stat().st_size / 1024 / 1024
    print(f"[+] GEO-bench 缓存已更新：{cache}（{len(records)} 条，{size_mb:.1f} MB）")
    return records[:limit], cache_dir, (url or GEO_BENCH_URLS[0])


def build_items_geobench(args: argparse.Namespace) -> tuple:
    """用官方 GEO-bench 构造实验样本。

    这是**最接近论文原设置**的路径：官方就是"每个 query 5 个来源，优化 sugg_idx 指定的那个"。
    竞争对手不是我们造的，是数据集自带的真实检索结果。
    """
    samples, cache_dir, used_url = load_geobench(
        args.limit or 20, Path(args.cache_dir), args.geobench_url,
        max_chars=args.geobench_chars,
    )
    field = "raw_text" if args.geobench_field == "raw" else "cleaned_text"
    items: list[BenchItem] = []
    # 🔴 2026-10-06 修正：`sugg_idx` 是 **0-based**（实测 150 条全部 5 个来源，
    #    sugg_idx 取值集合恰为 {0,1,2,3,4}，0-based 解释下 0% 越界；
    #    1-based 解释下 34/150 = 22.7% 越界）。
    #    原先默认 `sugg_idx_base=1` 会**把目标文档取错**（少 1），且对那 34 条
    #    静默回退到 index 0 —— 也就是在优化**竞争者**而不是数据集指定的目标。
    #    现在：0-based 为默认，且越界**硬报错**而不是静默回退。
    bad_idx: list[tuple[int, int, int]] = []
    for i, s in enumerate(samples):
        srcs = s.get("sources") or []
        if len(srcs) < 2:
            continue
        si = int(s.get("sugg_idx", 0))
        if args.sugg_idx_base == 1:
            si -= 1
        if not (0 <= si < len(srcs)):
            bad_idx.append((i, int(s.get("sugg_idx", -99)), len(srcs)))
            if not args.allow_bad_sugg_idx:
                print(f"ERROR: 第 {i} 条 sugg_idx={s.get('sugg_idx')!r} 越界"
                      f"（该条有 {len(srcs)} 个来源，--sugg-idx-base="
                      f"{args.sugg_idx_base}）。\n"
                      f"       静默回退会把目标文档取成**竞争者**，污染整个实验。\n"
                      f"       已中止。确认基准后重跑，或加 --allow-bad-sugg-idx "
                      f"显式接受（会在报告里标出条数）。",
                      file=sys.stderr)
                raise SystemExit(3)
            si = 0
        target = srcs[si]

        def mk(d: dict) -> dict:
            txt = (d.get(field) or d.get("cleaned_text") or "")
            return {"doc_text": txt[: args.geobench_chars], "url": d.get("url", "")}

        items.append(BenchItem(
            query_text=s.get("query", ""),
            query_id=f"GB{i:04d}",
            sku={}, match_mode=args.match,
            competitors=[mk(d) for j, d in enumerate(srcs) if j != si],
            doc_text=(target.get(field) or target.get("cleaned_text") or "")[: args.geobench_chars],
            source_url=target.get("url", ""),
            tags=",".join(s.get("tags") or []),
        ))
    src = f"{cache_dir / 'test.head.jsonl'} via {used_url} (field={field}, {args.geobench_chars} chars)"
    print(f"[+] 数据源：官方 GEO-bench（{len(items)} 条，每条 1 目标 + "
          f"{len(items[0].competitors) if items else 0} 个真实检索对手）")
    return items, src



def _tags(s: str) -> list[str]:
    return [t.strip() for t in (s or "").replace("，", ",").split(";") if t.strip()]


# 用于从标题里提取"商品名"的通用词表（这些是品类/属性词，不是品牌名的一部分）
GENERIC_TITLE_WORDS = [
    "精华液", "精华", "面霜", "粉底液", "粉底", "乳液", "化妆水", "爽肤水", "面膜",
    "防晒霜", "防晒", "口红", "眼影", "散粉", "眉笔", "眉粉", "洗发水", "香水",
    "洁面乳", "洁面", "卸妆水", "卸妆", "原液", "眼霜", "隔离霜", "气垫", "素颜霜",
    "安瓶", "套装", "礼盒", "官方", "旗舰店", "正品", "包邮", "新版", "升级",
    "专用", "专属", "适用", "推荐", "热卖", "学生党", "平价", "国货", "补水", "保湿",
    "美白", "修护", "抗皱", "淡纹", "紧致", "控油", "舒缓", "维稳", "提亮", "淡斑",
    "露",
]


def _product_name(sku: dict[str, str]) -> str:
    """从标题里提取商品名（去掉品类词、功效词、肤质词、括号标签后的中文段）

    例："【敏感肌专属】雅诗兰黛小棕瓶精华 50ml 第七代修护抗皱" → "雅诗兰黛小棕瓶"
    这样才能匹配到引擎用中文写的回答，而不是只匹配 brand 字段里的英文。

    关键：**优先取第一段中文**，因为电商标题总是"品牌+品名"开头，后面才是营销话术。
    如果按"最长"来选，海蓝之谜会被提成"烂脸救星泛红"这种垃圾片段。
    只有第一段被剥干净后不足 2 字，才退回到取最长的一段。
    """
    title = sku.get("title", "") or ""
    # 整段删掉【敏感肌专属】() 这类营销标签，而不是只删括号
    # （只删括号的话，"敏感肌专属"会变成独立片段，反而被误当成商品名）
    title = re.sub(r"[【】\[\]（）()][^【】\[\]（）()]*[【】\[\]（）()]", " ", title)
    # 属性词 = 该 SKU 自己的功效/肤质/成分标签
    attr: set[str] = set()
    for field_name in ("function_tags", "skin_type_tags", "ingredients_keywords"):
        attr.update(_tags(sku.get(field_name, "")))

    runs = re.findall(r"[一-鿿]+", title)
    if not runs:
        return ""

    def strip(run: str) -> str:
        cand = run
        for w in sorted(set(GENERIC_TITLE_WORDS) | attr, key=len, reverse=True):
            cand = cand.replace(w, "")
        return re.sub(r"第[一二三四五六七八九十\d]+代", "", cand)

    first = strip(runs[0])
    if len(first) >= 2:
        return first
    best = first
    for run in runs[1:]:
        cand = strip(run)
        if len(cand) > len(best):
            best = cand
    return best


def _target_tokens(sku: dict[str, str], mode: str = "auto") -> list[str]:
    """目标识别词集合。至少包含 brand；auto 模式再加入从标题提取的商品名。

    为什么要集合而不是单个字符串：skus.csv 的 brand 字段是英文（ESTEE LAUDER），
    而引擎的回答通常写中文（雅诗兰黛）。只匹配 brand 会让 impression rate 恒为 0。
    实际命中了哪个词会记进 raw_trials.csv 的 matched_token 列，便于人工审计。
    """
    toks: list[str] = []
    brand = (sku.get("brand") or "").strip()
    if brand:
        toks.append(brand)
    if mode in ("auto", "title"):
        name = _product_name(sku)
        if name and name.lower() != brand.lower():
            toks.append(name)
    return toks



def find_geo_agent(explicit: str | None) -> Path | None:
    """定位 geo-agent 仓库（里面有 data/queries.csv 和 data/skus.csv）"""
    cands: list[Path] = []
    if explicit:
        cands.append(Path(explicit))
    if os.environ.get("GEO_AGENT_ROOT"):
        cands.append(Path(os.environ["GEO_AGENT_ROOT"]))
    cands.extend(GEO_AGENT_CANDIDATES)
    for c in cands:
        if (c / "data" / "queries.csv").exists() and (c / "data" / "skus.csv").exists():
            return c
    return None


# 找不到 geo-agent 时的内置兜底数据，保证脚本在任何目录都能跑
DEMO_QUERIES = [
    {"query_id": "D001", "query_text": "敏感肌能用的美白精华推荐", "subcategory": "精华", "weight": "1.0"},
    {"query_id": "D002", "query_text": "油皮粉底液推荐", "subcategory": "粉底", "weight": "1.0"},
    {"query_id": "D003", "query_text": "干皮面霜怎么选", "subcategory": "面霜", "weight": "1.0"},
]

DEMO_SKUS = [
    {
        "sku_id": "DSKU001",
        "title": "焕亮修护精华 30ml",
        "brand": "青研实验室",
        "subcategory": "精华",
        "price": "289",
        "original_price": "369",
        "shop_type": "旗舰店",
        "skin_type_tags": "敏感肌;干皮;混合皮",
        "function_tags": "美白;修护;保湿",
        "ingredients_keywords": "烟酰胺;二裂酵母;透明质酸",
        "has_nmpa": "yes",
    },
    {
        "sku_id": "DSKU002",
        "title": "持妆控油粉底液 30ml",
        "brand": "青研实验室",
        "subcategory": "粉底",
        "price": "199",
        "original_price": "259",
        "shop_type": "旗舰店",
        "skin_type_tags": "油皮;混油皮",
        "function_tags": "控油;持妆",
        "ingredients_keywords": "PCA锌;二氧化钛",
        "has_nmpa": "yes",
    },
]


@dataclass
class BenchItem:
    """一个实验样本：1 个目标文档 + 若干竞争对手

    两种数据源共用这个结构：
      --dataset geoagent  sku = geo-agent 的一条美妆 SKU，base_doc 由字段拼装
      --dataset geobench  sku = 占位（{}），doc_text 直接用官方数据集给的来源正文
    """
    query_text: str
    query_id: str
    sku: dict[str, str]
    match_mode: str = "auto"
    competitors: list[dict[str, str]] = field(default_factory=list)
    doc_text: str = ""          # geo-bench：来源正文（有值时 base_doc 直接返回它）
    source_url: str = ""        # geo-bench：目标来源的 URL
    tags: str = ""              # geo-bench：原始 tags

    @property
    def brand(self) -> str:
        return (self.sku.get("brand") or "").strip() or _domain_label(self.source_url)

    @property
    def target_tokens(self) -> list:
        if self.source_url:
            # geo-bench 的身份是"哪个来源"，主指标本来就是按来源编号算的，
            # 这里给一个域名标签只用于 mentioned 这个辅指标。
            lbl = _domain_label(self.source_url)
            return [lbl] if lbl else []
        return _target_tokens(self.sku, self.match_mode)

    def base_doc(self) -> str:
        """baseline 文档。geo-bench 直接用数据集原文；geo-agent 由 SKU 字段拼装"""
        if self.doc_text:
            return self.doc_text
        s = self.sku
        lines = [
            f"{s.get('brand','')} {s.get('title','')}",
            f"价格：¥{s.get('price','')}（原价 ¥{s.get('original_price','')}）　店铺类型：{s.get('shop_type','')}",
            f"适合肤质：{s.get('skin_type_tags','')}",
            f"核心功效：{s.get('function_tags','')}",
            f"主要成分：{s.get('ingredients_keywords','')}",
            "使用方法：取适量均匀涂抹于面部，早晚各一次。",
        ]
        return "\n".join(lines)

    def competitors_base_docs(self) -> list:
        """竞争对手的 baseline 文档（geo-agent 从 SKU 拼，geo-bench 用数据集原文）"""
        out = []
        for c in self.competitors:
            if isinstance(c, dict) and c.get("doc_text"):
                out.append(c["doc_text"])
            else:
                out.append(BenchItem("", "", c, self.match_mode).base_doc())
        return out


"""策略与指标实现（替换用）。"""


def _kw(sku: dict, n: int = 2) -> list:
    pool = _tags(sku.get("function_tags", "")) + _tags(sku.get("ingredients_keywords", ""))
    return [k for k in pool if k][:n] or ["该产品"]


def s_unique_words(doc: str, sku: dict) -> str:
    """Unique Words 的规则版近似：换用更书面的同义表达"""
    repl = {"推荐": "力荐", "便宜": "更具性价比", "好用": "表现出色", "不错": "表现优异",
            "很多": "大量", "比较": "相当", "可以": "能够", "舒服": "舒适",
            "补水": "补水保湿", "效果": "功效表现"}
    out = doc
    for a, b in repl.items():
        out = out.replace(a, b)
    return out


# ------------------------------------------------------------------
# 以下 9 个是 rule 模式的近似实现，只用于零成本冒烟测试（--rewriter rule）。
# 论文的 9 个方法本质是 LLM 改写（官方 prompt 在上方），规则版**不是**复现。
# ------------------------------------------------------------------

def s_cite_sources(doc: str, sku: dict) -> str:
    return doc + (
        "\n参考依据：GB/T 29680-2013《化妆品安全技术规范》；"
        "《化妆品监督管理条例》（2021 年 1 月 1 日施行）；本品已完成国家药监局备案。"
    )


def s_add_statistics(doc: str, sku: dict) -> str:
    return doc + (
        "\n使用数据：8 周对照测试（n=120）中，92% 受试者反馈使用后 4 周保湿度明显改善，"
        "平均改善幅度 34%；复购率 41%。"
    )


def s_quotation_addition(doc: str, sku: dict) -> str:
    return doc + (
        "\n专家观点：\"该配方在温和性上表现稳定，适合敏感肌人群长期使用。\""
        "——某三甲医院皮肤科主治医师"
    )


def s_keyword_stuffing(doc: str, sku: dict) -> str:
    kws = "、".join(_kw(sku, 3))
    filler = "　".join([f"{k}是选购{k}时的核心考量" for k in _kw(sku, 3)])
    return doc + f"\n关键词说明：{kws}。{filler}。{kws}。{kws}。"


def s_fluency_optimization(doc: str, sku: dict) -> str:
    return doc + (
        "\n选购提示：总的来说，如果你的主要需求是" + "、".join(_kw(sku, 2))
        + "，那么这款产品值得优先考虑；不过具体到肤质适配和使用习惯，仍建议先做小面积测试，"
        "确认没有不适后再全脸使用。"
    )


def s_easy_to_understand(doc: str, sku: dict) -> str:
    explain = {
        "烟酰胺": "常见的提亮成分", "二裂酵母": "发酵类修护成分",
        "透明质酸": "大家熟悉的补水保湿成分", "PCA锌": "控油成分",
        "二氧化钛": "物理遮盖成分", "A醇": "抗老成分", "六胜肽": "紧致成分",
    }
    parts = [f"{k}（{explain[k]}）" for k in _kw(sku, 3) if k in explain]
    body = "、".join(parts) if parts else "该配方以基础保湿修护为主"
    return doc + f"\n成分白话解释：{body}。简单说，就是日常护肤里容易理解、接受度高的常见成分。"


def s_authoritative_style(doc: str, sku: dict) -> str:
    return doc + (
        "\n专业背书：本品由皮肤科学实验室按 Pharmaceutical-grade 标准生产，"
        "经第三方检测机构检验合格。建议按周期持续使用以观察完整功效周期。"
        "我们保证正品，你不会后悔。"
    )


def s_technical_terms(doc: str, sku: dict) -> str:
    return doc + (
        "\n技术参数：INCI 命名按 CTFA 标准标注；水相体系含甘油、"
        "透明质酸钠（hyaluronic acid sodium salt）；防腐体系为苯氧乙醇 + 羟苯甲酯；"
        "建议 pH 5.5±0.5，常温避光保存。"
    )




# ---------------------------------------------------------------- 官方 9 个 GEO 方法
# 策略名、编号、定义 100% 取自 Aggarwal et al. 2023 论文第 2.2.2 节；
# prompt 逐字取自官方仓库 https://github.com/GEO-optim/GEO 的 src/geo_functions.py。
#
# ⚠️ 两处官方实现与论文正文对不上的地方（本脚本如实保留并标注，不要"修正"）：
#   1. 论文正文说 Keyword Stuffing 是 "include more keywords **from the query**"，
#      但官方 prompt 里根本没拿到 query，只让模型自己加 SEO 关键词。
#      → 用 --keyword-stuffing-use-query 打开会把 query 补进 prompt（默认关闭 = 忠于代码）。
#   2. 论文正文写采样 temperature=0.7，官方 generative_le.py 的函数默认值是 0.5。
#      → 本脚本默认 0.7（忠于正文），想跟代码一致就 --temperature 0.5。
#
# ⚠️⚠️ 必须知道的一件事（也是本研究最值得写的观察之一）：
# 官方 prompt **明确指示模型编造来源**：
#     Cite Sources:   "You may invent these sources but ensure they sound plausible"
#     Quotation:      "Add more quotes in the source, **even though fake and artificial**"
#     Statistics:     "Add positive, compelling statistics (**even if hypothetical**)"
# 也就是说 Aggarwal 2023 报的那 40% 提升里，有一部分是靠往文档里塞
# **虚构的引用、假专家引语、假数据** 拿到的。这既是这个方法在真实业务里
# 最大的落地风险（平台方一旦核查就是处罚），也是一个现成的、可证伪的研究问题：
# **把"编造"换成"真实但可得"的证据，可见度提升还剩多少？**

_CITE_SOURCES_ZH = """请修改下面的来源文本，加入来自可信来源的引用。你可以虚构这些来源，但要让它们听起来合理、不要误导读者。引用不要写成研究论文那种形式，而要用换过说法的自然语言，例如："根据最新报告，这个产品将成为下一个爆款……"
在改写过程中，要保证来源文本的核心内容不被改动。初始来源与最终来源的长度应当相同，来源各个部分的结构（行距、项目符号等）应保持完整。

记住最终目标：当这个来源与一系列摘要一起呈现时，能让读者给予它更多注意力。所以要用自然语言多引用一些来源，但不要改动内容。

来源：
```
{summary}
```

补充要求：引用全文 5-6 处即可，务必与内容高度相关，且读起来自然。"""

_CITE_SOURCES_EN = """Revise the following source to include citations from credible sources. You may invent these sources but ensure they sound plausible and do not mislead the reader. Citations should not be research paper style, but rather should be in rephrased words. For example: "According to Google's latest report this product is going to be next big thing....'
In the process, ensure that the core content of the source remains unaltered. The length of initial source and final source should be the same, and the structure of individual parts of the source (such as line spacing bullet points, should remain intact)

Remember the end-goal is that readers give more attention to this source, when presented with a series of summaries, so cite more sources in natural language but do not alter content.

Source:
```
{summary}
```

Remember the end-goal is that readers give more attention to this source, when presented with a series of summaries, so cite more sources in natural language but do not alter content. Also don't overdo citing, 5-6 citations in the whole source are enough provided they are very relevant and and text looks natural."""

_QUOTATION_ZH = """请修改下面的来源文本，加入更多引语，且不改变核心内容。引语必须与内容相关、能提供有价值的信息，不得误导或歪曲原文。引语应当来自可敬的来源，并让读者感到这个网站（来源）比其他引用更具影响力和重要性。

来源：
```
{summary}
```

要求：尽量多加引语，即使是虚构、不自然的也没关系。思路是："加入权威人物的直接引述或声明，可能提升来源的可信度。"
保持文本格式与内容相近，例如行距、项目符号和整体结构应保持不变。
不允许增加或删除内容。"""

_QUOTATION_EN = """Modify the following source by including more quotes, without changing the core content. Ensure the quotes are relevant and contribute valuable information to the source, and do not mislead or distort the original information. The quotes should be reputable, and should create a sense in readers that this website (source) is more influentical and important than other citations.

Source:
```
{summary}
```

Add more quotes in the source, even though fake and artificial. The idea is: "Including direct quotes or statements from authoritative figures might increase the credibility of the source."
Keep the format and content of text similar.
For example, line spacing, bullet points and overall structure should remain the same.
No addition or deletion of content is allowed."""

_AUTHORITATIVE_ZH = """请把下面的来源文本改写成权威风格，不增删任何核心信息。改写后应体现自信、专业与果断，同时保持原意与相关性。

来源应当以断言式的口吻陈述，让读者相信这是比其他来源摘要更有价值的信息。最终目标是通过自信的表述提升该来源被引用的概率，也就是断言这是质量最好的信息。
但内容与结构必须保持不变：只允许改写单行或 2-3 句，内容保持一致。

来源：
```
{summary}
```

记住要保持权威感，但文本格式与内容保持一致。例如行距、项目符号和整体结构应保持不变。
不允许增加或删除内容。
注意要通过文本本身的自信表述让读者相信这是一个重要来源。可以加入诸如"只有我们是正品""我们保证"等措辞，以及第二人称表达如"你不会后悔"等。"""

_AUTHORITATIVE_EN = """Transform the following source into an authoritative style without adding or removing any core information. The revised source should reflect confidence, expertise, and assertiveness, while maintaining the original content's meaning and relevance.
The source should be assertive in its statements, such that reader believes that this is more valuable source of information than other provided summaries. End Goal is to increase the citation of this source, by assertively saying that this is the best quality inforamtion.
However, the content and structure of the source should remain the same. That means, only individual lines and/or 2-3 sentences can be paraphrased, while keeping the content same.

Source:
```
{summary}
```

Remember to be authoritative, but keep the format and content of text the same.
For example, line spacing, bullet points and overall structure should remain the same.
No addition or deletion of content is allowed.
Note we have to convince the reader that this is an important source by saying it authoritatively in the source itself.
For example the addition of phrases such as "only we are authentic etc", 'we guarantee', use of second pronouns such as "you will not regret" etc is expected within the source content itself."""

_SIMPLE_LANGUAGE_ZH = """请用简单易懂的语言简化下面的来源，同时确保关键信息仍然传达完整。过程中不要遗漏、增加或改动任何核心信息。

记住最终目标：当这个来源与一系列摘要一起呈现时，能让读者给予它更多注意力。所以要让语言更容易理解，但不要删除任何信息。
新来源的长度应与原文相同。实际上你只需要重写单个表述，让它们更清晰易懂。

来源：
```
{summary}
```
"""

_SIMPLE_LANGUAGE_EN = """Simplify the following source, using simple, easy-to-understand language while ensuring the key information is still conveyed. Do not omit, add, or alter any core information in the process.

Remember the end-goal is that readers give more attention to this source, when presented with a series of summaries, so make the language easier to understand, but do not delete any information.
The length of the new source should be the same as the original. Effectively you have to rephrase just individual statements so they become more clear to understand.

Source:
```
{summary}
```
"""

_TECHNICAL_ZH = """请让下面的来源更具技术性，在需要的地方给出更多术语和事实，同时确保关键信息仍然传达完整。过程中不要遗漏、增加或改动任何核心信息。

记住最终目标是让知识更丰富的读者在看到一系列摘要时给予这个来源更多注意力，所以要让语言包含更多技术信息，或让已有信息以更技术化的方式呈现。但不要增加或删除内容。初始来源的词数应与最终来源相同。
新来源的长度应与原文相同。实际上你只需要重写单个表述，让它们包含更丰富的技术信息。

来源：
{summary}
"""

_TECHNICAL_EN = """Make the following source more technical, using giving more technical terms and facts where needed while ensuring the key information is still conveyed. Do not omit, add, or alter any core information in the process.

Remember the end-goal is that very knowledgeable readers give more attention to this source, when presented with a series of summaries, so make the language such that it has more technical information or existing information is presented in more technical fashion. However, Do not add or delete any content . The number of words in the initial source should be the same as that in the final source.
The length of the new source should be the same as the original. Effectively you have to rephrase just individual statements so they have  more enriching technical information in them.

Source:
{summary}
"""

_FLUENT_ZH = """请改写下面的来源，使其更流畅，同时不改动核心内容。句子之间应当自然衔接，语言要清晰、生动，同时保留原有信息。

来源：{summary}
"""

_FLUENT_EN = """Rewrite the following source to make it more fluent without altering the core content. The sentences should flow smoothly from one to the next, and the language should be clear and engaging while preserving the original information.

Source: {summary}"""

_UNIQUE_WORDS_ZH = """请改写下面的来源，加入更多独特、少见的用词，同时不改动核心信息。确保这些用词是为内容增色，而不是随意堆砌，并且原意得到保留。

来源：{summary}
"""

_UNIQUE_WORDS_EN = """Revise the following source by incorporating more unique and rare words, without altering the core information. Ensure that these words enhance the content and are not used arbitrarily, and the original meaning is preserved.

Source: {summary}"""

_STATS_ZH = """这是需要优化的来源：
```
{summary}
```

## 任务：
在文本中多个合适的位置加入正面、有说服力的统计数据（即使是假设的也可以）。统计指客观事实，例如营销上的 x% 增长、科学文本中的数字、有趣的数值事实。
先找出可以加入统计、数字或客观事实的位置，例如"在关于 xx 的句子中，可以加入关于 yy 的 zz 的统计数据"。请使用确切数字而不是 xx、yy、zz。
然后用三反引号输出加入统计之后的来源文本。

## 需遵循的准则：
1. 统计与数字应当"悄无声息"地内嵌在句子中，不要新增独立段落或大块文字。
2. 除了加入统计的那些行，不要改动任何文本内容。
3. 除了你加入的统计，不要增加或删除内容。停在与初始来源对应的最后一行，即使那行不完整。
4. 只输出优化后的来源文本，不需要给任何解释、推理或结论。
5. 先找出可以加入统计、数字或客观事实的位置，并使用确切数字。然后用三反引号输出更新后的来源。

## 输出格式：
1. 要加入的统计
2. 要加入的统计。
....
k. 要加入的统计。

更新后的输出：
```
<Output>
```
"""

_STATS_EN = """Here is the source:
```
{summary}
```

## Task:
Add positive, compelling statistics (even if hypothetical) at multiple relevant places in the text. Statsistics means objective facts such as x% growth in marketing, numbers in scientific texts, interesting numerical facts. First identify the places where statistics, numbers or objective facts can be added. Eg: "In sentence about xx, stats about yy of zz can be added" However use exact numbers instead of xx, yy or zz. Then in triple ticks output the updated source, which would have stats included.

## Guidelines to follow:
1. Statsistics and Numbers should be subtly added inline within the sentneces. No explicit paragraphs or big chunks of text should be added.
2. Do not update any text content except for the lines where you are adding statistics.
3. Do not add or delete content except the statistics you are adding. Stop at the last line corresponding to the inital source, even if it is incomplete.
4. Just output the optimized source text. No need to give any explanation or reasoning or conclusion.
5. First identify the places where statistics, numbers or objective facts can be added. Eg: "In sentence about xx, stats about yy of zz can be added". However use exact numbers instead of xx, yy or zz. Then in triple ticks output the updated source, which would have stats included.

## Output Format:
1. Stat to be added
2. Stat to be added.
....
k. Stat to be added.

Updated Output:
```
<Output>
```
"""

_SEO_ZH = """这是需要更新的来源：
```
{summary}
```

## 任务：
按 SEO 原则在来源中加入**新的**关键词以优化内容。注意不能使用来源里已经出现的关键词，只能加入新的关键词。

## 需遵循的准则：
1. 通过在不同位置加入相关关键词来为来源做 SEO 优化，这些关键词必须是新的、与已有不同的。
2. 先找出可以加入的关键词，例如"在关于 zzz 的句子中，加入关键词 xxx"。请使用真实关键词与真实句子，例如"在关于光合作用的句子中，加入关键词叶绿素。"
3. 新关键词最多 10 个，必须与来源中已有的不同。
4. 最后用三反引号输出加入关键词后的来源。

## 输出格式：
1. 在关于关键词 zzz 的句子中，加入关键词 xxx
2. 在关于关键词 zzz 的句子中，加入关键词 xxx
....
k. 在关于关键词 zzz 的句子中，加入关键词 xxx

现在我输出更新后的文本：
更新后的输出：
```
<Output>
```"""

_SEO_EN = """Here is the source that you need to update:
```
{summary}
```

## Task:
Add NEW keywords in the source that optimize the content in accordance with SEO principles. Note you cannot use the keywords already present in the source. You have to only include the new keywords.

## Guidelines to follow:
1. Remember to optimize source for SEO, by adding relevant keywords at different places. These keywords should be new, different from those already present in source.
2. First identify the keywords that can be added. Eg: "In sentence about zzz, add keyword xxx". However, use actual keyword instead of xxx and actual sentence instead of zzz. For example: "In sentence about photosynthesis, add keyword Chlorophyll."
3. Maximum new keywords should be 10. Remember keywords should be DIFFERENT from those already present in source.
4. Finally, in triple ticks output the updated source, which would have the keywords included.

## Output Format:
1. In sentence about keyword zzz, add keyword xxx
2. In sentence about keyword zzz, add keyword xxx
....
k. In sentence about keyword zzz, add keyword xxx

Now I will output the updated text:
Updated Output:
```
<Output>
```"""

# key -> (英文名, 论文中的定义, 是否需要新增内容, 规则版改写函数, 中文 prompt, 英文 prompt)
# "是否需要新增内容" 依据论文第 2.2.2 节：except for methods 3, 4, and 5,
# the remaining methods enhance the presentation of existing content ... without requiring extra content
STRATEGIES = {
    "authoritative": (
        "Authoritative",
        "Modifies text style of the source content to be more persuasive and authoritative",
        False, s_authoritative_style, _AUTHORITATIVE_ZH, _AUTHORITATIVE_EN,
    ),
    "statistics_addition": (
        "Statistics Addition",
        "Modifies content to include quantitative statistics instead of qualitative discussion, wherever possible",
        True, s_add_statistics, _STATS_ZH, _STATS_EN,
    ),
    "keyword_stuffing": (
        "Keyword Stuffing",
        "Modifies content to include more keywords from the query, as expected in classical SEO optimization",
        True, s_keyword_stuffing, _SEO_ZH, _SEO_EN,
    ),
    "cite_sources": (
        "Cite Sources",
        "Adds relevant citations from credible sources",
        True, s_cite_sources, _CITE_SOURCES_ZH, _CITE_SOURCES_EN,
    ),
    "quotation_addition": (
        "Quotation Addition",
        "Adds relevant quotations from credible sources",
        True, s_quotation_addition, _QUOTATION_ZH, _QUOTATION_EN,
    ),
    "easy_to_understand": (
        "Easy-to-Understand",
        "Simplifies the language of website",
        False, s_easy_to_understand, _SIMPLE_LANGUAGE_ZH, _SIMPLE_LANGUAGE_EN,
    ),
    "fluency_optimization": (
        "Fluency Optimization",
        "improves the fluency of website text",
        False, s_fluency_optimization, _FLUENT_ZH, _FLUENT_EN,
    ),
    "unique_words": (
        "Unique Words",
        "involves adding unique and rare words wherever possible",
        False, s_unique_words, _UNIQUE_WORDS_ZH, _UNIQUE_WORDS_EN,
    ),
    "technical_terms": (
        "Technical Terms",
        "involves adding technical terms wherever possible",
        False, s_technical_terms, _TECHNICAL_ZH, _TECHNICAL_EN,
    ),
}

# 论文正文里的方法编号（1..9），用于和原文对照
PAPER_NUMBER = {
    "authoritative": 1, "statistics_addition": 2, "keyword_stuffing": 3,
    "cite_sources": 4, "quotation_addition": 5, "easy_to_understand": 6,
    "fluency_optimization": 7, "unique_words": 8, "technical_terms": 9,
}


def strategy_prompt(arm: str, doc: str, query: str, lang: str, use_query: bool) -> str:
    """构造官方 prompt。{summary} 是待改写的来源文本。"""
    if arm == "baseline":
        return ""
    prompt = STRATEGIES[arm][5] if lang == "en" else STRATEGIES[arm][4]
    if use_query and arm == "keyword_stuffing":
        # 论文正文说关键词要"from the query"，但官方 prompt 没传 query。
        # --keyword-stuffing-use-query 打开时把 query 补进去（这是本脚本的补充，不是官方实现）。
        prompt += f"\n\n用户当前搜索的 query 是：{query}"
    return prompt.replace("{summary}", doc)


# ---------------------------------------------------------------- LLM 改写器
# 论文的 9 个方法本质上都是 "gpt-3.5-turbo-16k, temperature=0.0" 的改写 prompt。
# 这里换成你自己的 key（默认 qwen-plus）来跑同一套 prompt —— 这是"复现方法"，
# 而不是"换了个模型做别的实验"。偏离点：改写模型不同（论文用 gpt-3.5-turbo-16k）。
rewriter_fn = None      # (arm, prompt) -> text；None 表示不做 LLM 改写（dry-run / rule 模式）
rewriter_lang = "zh"
rewriter_use_query = False
_rewriter_cache: dict = {}
_rewriter_lock = __import__("threading").Lock()


def make_rewriter(engine: str, model: str, api_key: str, cache_path: Path, max_tokens: int):
    """返回一个带缓存的改写函数。

    为什么必须缓存：官方代码也缓存（geo_optimizations_cache.json）。
    一个 arm × 一个 SKU 的改写结果是固定的，不该在每次 --repeats 时重复付费。
    """
    global _rewriter_cache
    if cache_path.exists():
        try:
            _rewriter_cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            _rewriter_cache = {}
    system = REWRITER_SYSTEM_EN if rewriter_lang == "en" else REWRITER_SYSTEM_ZH

    def _rewrite(arm: str, prompt: str) -> str:
        key = f"{model}||{arm}||{prompt}"
        with _rewriter_lock:
            if key in _rewriter_cache:
                return _rewriter_cache[key]
        # 官方 temperature=0.0：改写要确定性，否则同臂不同 repeat 的差异会混入改写噪声
        text, _ = call_engine(
            engine, model, api_key, system, prompt,
            temperature=0.0, max_tokens=max_tokens, seed=7, retries=3,
        )
        text = _strip_code_fence(text)
        with _rewriter_lock:
            _rewriter_cache[key] = text
            try:
                cache_path.write_text(json.dumps(_rewriter_cache, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
            except Exception:
                pass
        return text

    return _rewrite


def _strip_code_fence(text: str) -> str:
    """对齐官方 geo_functions.py 里的 get_summary()：把内容从 ``` 代码块里取出来。"""
    text = text.replace("```\n```", "```")
    b = text.rfind("```")
    if b == -1:
        return text.strip()
    if text.count("```") < 2:
        a = b + 3
        b = -1
    else:
        a = text[:b].rfind("```") + 3
    if b - a < 50:
        a = b if len(text) - b > 200 else a
        b = -1
    if a <= 2:
        a = 0
    out = (text[a:b] if b != -1 else text[a:]).strip()
    if out.lower().startswith("updated") or out.startswith("更新后的输出"):
        out = "\n".join(out.splitlines()[1:]).strip()
    return out or text.strip()


def apply_arm_rule(arm: str, item) -> str:

    """规则版改写：零成本、可复现，用于冒烟测试。
    ⚠️ 这不是论文方法的复现 —— 论文的 9 个方法都是 LLM 改写，规则版只是"长得像"的替代品。
       要复现请用 --rewriter llm（默认）。"""
    doc = item.base_doc()
    if arm == "baseline":
        return doc
    return STRATEGIES[arm][3](doc, item.sku)


def apply_arm(arm: str, item) -> str:
    """LLM 改写版（默认）。真正的论文方法复现路径。
    rewriter_fn 注入为 (arm, prompt) -> text；为 None 时退回规则版。"""
    if rewriter_fn is None:
        return apply_arm_rule(arm, item)
    doc = item.base_doc()
    if arm == "baseline":
        return doc
    prompt = strategy_prompt(arm, doc, item.query_text, rewriter_lang, rewriter_use_query)
    out = rewriter_fn(arm, prompt)
    return out if out.strip() else doc



# ---------------------------------------------------------------- 指标

def _tokenize_words(text: str) -> list:
    """中文按字 + 英文按词切，用来算位置"""
    return re.findall(r"[一-鿿]|[A-Za-z0-9]+", text.lower())


def _split_sentences(text: str) -> list:
    """按段落切分后再切句。官方用 nltk.word_tokenize/sent_tokenize（只支持英文），
    这里改用中英通用的标点切分，其余逻辑与官方一致。"""
    sents = []
    for para in re.split(r"\n\s*\n", text):
        for s in re.split(r"(?<=[。！？!?；;])\s*|\n+", para):
            s = s.strip()
            if s:
                sents.append(s)
    return sents


def _num_words(sent: str) -> int:
    """官方 get_num_words: 统计长度 > 2 的 token 数。

    ⚠️ 必要的偏离：官方用 nltk.word_tokenize，只对英文有效；论文处理的也是英文网页。
    本 harness 的源文档是中文，逐字切分后每个 token 长度都是 1，
    `len(x) > 2` 永远不会成立 —— 指标会**静默塌成全 0**。
    所以这里等价地定义：CJK 字符按 1 个词计，英文/数字按 len>2 计。
    这不是"改公式"，是让"句子的信息量"这个量在中文下有非零意义。
    """
    cjk = len(re.findall(r"[一-鿿]", sent))
    latin = len([x for x in re.findall(r"[A-Za-z0-9]+", sent.lower()) if len(x) > 2])
    return cjk + latin


# 官方代码在"回答里一个引用都没有"时返回均分 1/n（见 utils.py 的 else 分支），
# 而不是全 0。这是个真实的实现歧义：没有任何引用时，可见度到底该是 0 还是均分？
# 官方选了均分。本脚本默认忠于官方，但用 --metric-empty zero 可以切成 0 做敏感性分析。
metric_empty_mode = "uniform"


def impression_score_all(response: str, n_sources: int) -> list:
    """返回所有来源的归一化 impression 份额（论文主指标）

    逐字对齐官方 impression_wordpos_count_simple 的公式：
        对第 i 句中的每个引用 c：
            score = 该句词数 * exp(-i / (总句数 - 1)) / 该句引用数
            scores[c-1] += score
        最后按总分归一化
    """
    sents = _split_sentences(response)
    if not sents:
        return [0.0] * n_sources
    scores = [0.0] * n_sources
    n = len(sents)
    for i, sent in enumerate(sents):
        cits = [int(x) for x in re.findall(r"\d+", " ".join(CITATION_NUM_RE.findall(sent)))]
        if not cits:
            continue
        base = _num_words(sent)
        decay = math.exp(-1.0 * i / (n - 1)) if n > 1 else 1.0
        for c in cits:
            if 1 <= c <= n_sources:
                scores[c - 1] += base * decay / len(cits)
    total = sum(scores)
    if total > 0:
        return [x / total for x in scores]
    # 一个引用都没有：官方在这里返回均分 1/n（utils.py 的 else 分支）
    if metric_empty_mode == "uniform" and n_sources > 0:
        return [1.0 / n_sources] * n_sources
    return [0.0] * n_sources


def impression_wordpos_count_simple(response: str, n_sources: int, target_idx: int = 1) -> float:
    """目标来源的归一化 impression 份额（0-1）。官方返回整个向量再取 idx。"""
    share = impression_score_all(response, n_sources)
    return share[target_idx - 1] if 1 <= target_idx <= n_sources else 0.0



def measure(response: str, tokens: list, target_idx: int = 1, n_sources: int = 1) -> dict:
    """从引擎回答里测指标

    geo_impression  = 论文主指标 impression_wordpos_count_simple 里目标来源的份额（0-1）
    cited           = 目标来源的编号是否被引用（二值）
    mentioned       = 品牌/商品名是否在文中出现（二值，噪声大，仅作参考）

    为什么以 geo_impression 为主指标：它是论文自己用来支撑 40% 结论的那个指标。
    mentioned 之所以不可靠，参见 arXiv:2606.20065：情感框定变化的频率
    约为"是否被提及"的 6.7 倍 —— 提及是个高噪声信号。

    🔴 `cited_pos_ratio` —— 2026-10-06 新增，**纠正一个实现与预注册定义的错配**
    ------------------------------------------------------------------------
    2604.25707 的 Eq 2 里有 `0.15 · (1 − first_position_ratio)`，与
    `0.20 · min(ref_count/3, 1)` 并列。13 号笔记 §3.2 写明 absorption 是
    **「引用页级」**度量，§(b)1 又把两者描述成「被多次引用、**排位靠前**」
    —— 同一句话里两个成分都关于**引用**，不是关于品牌提及。

    而本函数原先的 `first_pos` 量的是**品牌首次提及**在文中的词位，且只在
    `mentioned=1` 时有定义。实测后果（2026-10-06 阶段 A'，MiniMax-M3，
    1,980 格）：`mentioned=1` 只有 **7.1%**（140/1,980），而 `cited=1` 有
    **69.9%**（1,385/1,980）。也就是说预注册的**共同主因变量**
    `first_position_ratio` 只覆盖了 7.1% 的格子 —— 配对分析基本无从谈起。
    覆盖率的差（7.1% vs 69.9%）本身就是「量错了对象」的指纹。

    正确口径是**目标来源首次被引用的位置**：
    - 找第一个形如 `[target_idx]` 的引用标记；
    - 位置按**比率**给（`名次 / 总词数`，0..1），因为 Eq 2 里它就叫
      `ratio` 且参与 `1 −` 运算，必须是无量纲的；
    - 未被引用时为 -1（与 `first_pos` 一致，用缺失而不是 0 —— 0 会被
      当成「第一个词就出现」，是最容易混过去的假值）。

    🔴 两个口径**都保留**：`first_pos`（提及位置）不改名不改值，
       这样已落盘的 1,980 格历史数据仍可复现旧口径；
       `cited_pos_ratio` 是新增列。**不静默替换**。
    """
    low = response.lower()
    matched = ""
    hit_idx = -1
    for t in tokens:
        if t and t.lower() in low:
            i = low.find(t.lower())
            if hit_idx < 0 or i < hit_idx:
                hit_idx, matched = i, t
    mentioned = 1 if hit_idx >= 0 else 0

    words = _tokenize_words(response)
    first_pos = int(round(hit_idx / max(len(low), 1) * len(words))) + 1 if (mentioned and words) else -1
    paid = max(0.0, 1.0 - (first_pos - 1) / max(len(words), 1)) if mentioned else 0.0

    # 🔴 引用位置口径（Eq 2 对齐），未被引用 = -1
    cited_pos_ratio = -1.0
    m = re.search(rf"\[\s*{target_idx}\s*\]", response)
    if m and words:
        cited_pos_ratio = round(
            (m.start() / max(len(response), 1)) * len(words) / max(len(words), 1), 4)

    # 论文主指标：把目标来源挪到第 1 位再算，避免索引换算
    share = impression_score_all(response, n_sources)
    geo_imp = share[target_idx - 1] if 1 <= target_idx <= n_sources else 0.0

    return {
        "mentioned": mentioned,
        "cited": int(bool(re.search(rf"\[\s*{target_idx}\s*\]", response))),
        "geo_impression": round(geo_imp, 4),
        "first_pos": first_pos,
        "cited_pos_ratio": cited_pos_ratio,
        "paid_impression": round(paid, 4),
        "citation_marker": int(bool(CITATION_RE.search(response))),
        "matched_token": matched,
        "response_chars": len(response),
    }


# ---------------------------------------------------------------- 统计

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 区间（小样本下比 Wald 区间靠谱得多）"""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def two_proportion_ztest(k1: int, n1: int, k2: int, n2: int) -> tuple:
    """双比例 z 检验。返回 (z, 双尾 p)。k1/n1 = 策略臂，k2/n2 = baseline"""
    if n1 == 0 or n2 == 0:
        return (0.0, 1.0)
    p1, p2 = k1 / n1, k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return (0.0, 1.0)
    z = (p1 - p2) / se
    return (z, 2 * (1 - _norm_cdf(abs(z))))


def _stdev(xs: list) -> float:
    """样本标准差（ddof=1）。少于 2 个样本返回 0。"""
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def _welch_df(s1: float, n1: int, s2: float, n2: int) -> float:
    """Welch–Satterthwaite 自由度"""
    if n1 < 2 or n2 < 2:
        return 0.0
    a, b = s1 ** 2 / n1, s2 ** 2 / n2
    if a + b == 0:
        return 0.0
    return (a + b) ** 2 / (a * a / (n1 - 1) + b * b / (n2 - 1))


def _betacf(a: float, b: float, x: float) -> float:
    """连分式展开（Lentz 算法），用于不完全 Beta 函数"""
    MAXIT, EPS, FPMIN = 300, 3.0e-12, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """正则化不完全 Beta 函数 I_x(a, b)"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * _betacf(b, a, 1.0 - x) / b


def _t_sf(t: float, df: float) -> float:
    """t 分布双尾 p 值"""
    if df <= 0:
        return 1.0
    return _betai(df / 2.0, 0.5, df / (df + t * t))



# ---------------------------------------------------------------- 引擎调用

@dataclass
class CallStats:
    calls: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    lock: Any = field(default_factory=lambda: __import__("threading").Lock())

    def add(self, p: int, c: int) -> None:
        with self.lock:
            self.calls += 1
            self.prompt_tokens += p
            self.completion_tokens += c


class RateLimiter:
    """全局最小请求间隔（跨线程）。

    为什么需要它（2026-10-06 阶段 A' 补齐时实测）
    --------------------------------------------
    MiniMax 的 429 是**配额制**的，不是**并发制**的：把 worker 从 8 降到 3
    只是把同样多的请求挤进更少的时间槽，**单位时间请求数没变**，429 照样来。
    真正决定成败的是「每分钟打多少个请求」。线程数管并发，限速器管速率，
    两者是不同的杠杆 —— 之前只调了前者，所以补齐跑了几分钟 CPU 只涨 0.08s
    （全在退避里睡觉）。

    注意限速器只控制**请求发起**的节奏；单次调用的耗时仍由网络与输出生成决定。
    """

    def __init__(self, min_interval: float = 0.0):
        self.min_interval = float(min_interval)
        self._lock = __import__("threading").Lock()
        self._next_at = 0.0
        self.n_waits = 0
        self.waited_sec = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.time()
            t = now if now > self._next_at else self._next_at
            self._next_at = t + self.min_interval
            self.n_waits += 1
        d = t - now
        if d > 0:
            self.waited_sec += d
            time.sleep(d)


def call_engine(
    engine: str, model: str, api_key: str, system: str, user: str,
    temperature: float, max_tokens: int, seed: int, timeout: int = 90,
    retries: int = 3, backoff: float = 1.5, limiter: "RateLimiter | None" = None,
) -> tuple[str, dict]:
    """调用引擎并返回 (文本, usage)。

    退避策略（2026-10-06 为阶段 A' 补齐而加）
    ----------------------------------------
    2026-10-06 阶段 A'（MiniMax-M3）首轮 18.6% 传输层失败，其中 360 次是
    HTTP 429 限流。原来的退避是 `1.5 * 2**attempt`（1.5s/3s/6s），对 429
    来说太短，而且**多线程会同步重试** —— 8 个 worker 同时睡 1.5s 再一起
    冲，正好把限流窗口顶回去，形成自维持的失败。

    三处修正：
    ① `backoff` 提到参数（默认 1.5 保持旧行为不变）；
    ② 429/529 走**更长的**专用退避（`backoff * 3`）—— 限流需要秒级让路，
       而参数错误/网络瞬断用短退避就够；
    ③ 加抖动，破坏多线程的同步重试。

    🔴 重试**只针对传输层**（429/5xx/网络），绝不针对内容质量 ——
       §3 禁止的是「改写过不了信息保持门就重试」，那是挑结果造偏差。
       这里重试的是「压根没拿到测量」，拿到就停。
    """
    cfg = ENGINES[engine]
    payload: dict[str, Any] = {
        "model": model or cfg["model"],
        "messages": (
            [{"role": "user", "content": user}]
            if not system else
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        ),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # seed 能显著降低方差，是可复现性的关键（Qwen / OpenAI 均支持）
    # ⚠️ 但 **MiniMax M3 接受 seed 却不用**（2026-10-06 实测：同 seed 两次输出不同）。
    #    所以 M3 上重复测量是真正的 i.i.d. 抽样 —— 引擎侧非确定性全部进测量噪声。
    if seed is not None:
        payload["seed"] = seed
    # 引擎专属附加字段（如 MiniMax 关闭 thinking）
    extra_body = cfg.get("extra_body")
    if extra_body:
        payload.update(extra_body)

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_err: Exception | None = None
    for attempt in range(retries):
        if limiter is not None:
            limiter.wait()
        try:
            req = urllib.request.Request(
                cfg["url"],
                data=data,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            text = body["choices"][0]["message"]["content"] or ""
            usage = body.get("usage") or {}
            return text, {
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            }
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8")[:300]
            except Exception:
                pass
            last_err = RuntimeError(f"HTTP {e.code}: {detail}")
            # 400/401/403 = 参数/鉴权；422 = 内容审核或不可处理输入。
            # 🔴 2026-10-06：422 原先不在这个集合里，于是 MiniMax 的
            #    `input new_sensitive` 拦截被当成瞬时错误重试 5 次 ——
            #    那是**确定性**拒绝（实测同一 query 15/15 个 rep 全被拦），
            #    重试只是白烧配额。加进不可重试集合。
            if e.code in (400, 401, 403, 422):
                break
            # 限流/过载走更长退避（见 docstring）
            mult = 3.0 if e.code in (429, 529) else 1.0
            time.sleep(mult * backoff * (2 ** attempt)
                       + random.uniform(0, backoff))
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(backoff * (2 ** attempt) + random.uniform(0, backoff))
    raise RuntimeError(str(last_err) if last_err else "unknown error")


# ---------------------------------------------------------------- 主流程

def _relevance(query: dict[str, str], sku: dict[str, str]) -> int:
    """query 与 SKU 的属性重合度。

    为什么要这个：query 问"美白精华"却配到"修护抗皱"的 SKU 时，文档根本不回答问题，
    baseline 的 impression rate 会接近 0，臂间对比就变成纯噪声。必须让文档能回答问题。
    """
    score = 0
    if (query.get("subcategory") or "").strip() == (sku.get("subcategory") or "").strip():
        score += 2
    if (query.get("category") or "").strip() == (sku.get("category") or "").strip():
        score += 1
    q_func = (query.get("target_function") or "").strip()
    if q_func and q_func in (sku.get("function_tags") or ""):
        score += 2
    q_skin = (query.get("target_skin_type") or "").strip()
    if q_skin and q_skin in (sku.get("skin_type_tags") or ""):
        score += 2
    return score


def build_items(args: argparse.Namespace) -> tuple:
    if args.dataset == "geobench":
        return build_items_geobench(args)

    root = find_geo_agent(args.data_root)

    sku_file = "skus_synthetic.csv" if args.synthetic else "skus.csv"

    if root is None:
        print("[!] 找不到 geo-agent 仓库，改用内置 demo 数据（3 query × 2 SKU）")
        print("    指定真实数据：--data-root <geo-agent 路径> 或设环境变量 GEO_AGENT_ROOT\n")
        pairs = [(q, s) for q in DEMO_QUERIES for s in DEMO_SKUS]
        items = [
            BenchItem(query_text=q["query_text"], query_id=q["query_id"], sku=s, match_mode=args.match)
            for q, s in pairs
        ]
        return items, "builtin-demo"

    queries = _read_csv(root / "data" / "queries.csv")
    skus = _read_csv(root / "data" / sku_file)
    print(f"[+] 数据源：{root / 'data'}  ({sku_file}, {len(queries)} query × {len(skus)} SKU)")

    items: list[BenchItem] = []
    n_unmatched = 0
    for q in queries:
        sub = (q.get("subcategory") or "").strip()
        pool = [s for s in skus if (s.get("subcategory") or "").strip() == sub] or \
               [s for s in skus if (s.get("category") or "").strip() == (q.get("category") or "").strip()] or skus
        if args.relevance_match:
            pool = sorted(pool, key=lambda s: (-_relevance(q, s), s.get("sku_id", "")))
        if _relevance(q, pool[0]) == 0:
            n_unmatched += 1
        for t_i in range(args.docs_per_query):
            if t_i >= len(pool):
                break
            target = pool[t_i]
            # 竞争对手取同类目里紧随其后的几个 —— 真实引擎会检索出多个候选再挑几个引用，
            # 只注入一份文档的话"是否被提及"必然是 1.0，指标会顶到天花板而失去区分力
            comps = [
                s for s in pool
                if s.get("sku_id") != target.get("sku_id")
            ][: args.competitors]
            items.append(BenchItem(
                query_text=q["query_text"], query_id=q["query_id"],
                sku=target, match_mode=args.match, competitors=comps,
            ))
    if n_unmatched:
        print(f"[!] 有 {n_unmatched}/{len(queries)} 个 query 找不到属性匹配的 SKU，"
              f"这些配对的 baseline 会偏低（已知局限）")
    k = len(items[0].competitors) if items else 0
    print(f"[+] 竞争设置：每个 query 注入 1 个目标文档 + {k} 个同类目竞争对手，"
          f"目标位置按 repeat 轮换（位置偏置反平衡）")
    return items, str(root / "data" / sku_file)


def parse_arms(spec: str) -> list[str]:
    if spec.strip() == "all":
        return ["baseline", *STRATEGY_ORDER]
    arms = [a.strip() for a in spec.split(",") if a.strip()]
    bad = [a for a in arms if a != "baseline" and a not in STRATEGIES]
    if bad:
        sys.exit(f"[!] 未知策略：{bad}\n    可用：baseline, {', '.join(STRATEGY_ORDER)}")
    if "baseline" not in arms:
        arms = ["baseline", *arms]
    return arms


def write_arms_preview(items: list, arms: list, out: Path, limit: int = 1) -> Path:
    """把每个策略的改写结果 + 实际发出的 prompt 落盘。

    复现时必须先人工过一眼这个文件：如果策略改错了，后面跑再多调用也是白跑。
    LLM 模式下这里会真实调用改写器（结果会进缓存），所以第一次跑就有真实改写文本。
    """
    p = out / "arms_preview.md"
    lines = [
        "# 各策略改写结果 + 实际 prompt 预览",
        "",
        "> **先读这个文件，再花钱跑实验。** 每个策略下面有两段：",
        "> 1. 实际发给改写器的 prompt（官方 prompt 逐字版 / 中文翻译版）",
        "> 2. 改写后的文档 —— 确认策略真的按论文定义生效了",
        "",
        "> 策略名、编号、定义取自 Aggarwal et al. 2023 第 2.2.2 节；",
        "> prompt 取自官方仓库 https://github.com/GEO-optim/GEO 的 `src/geo_functions.py`。",
        "",
        f"> ⚠️ 官方 prompt 明确要求**编造**来源（\"You may invent these sources\"、"
        "\"even though fake and artificial\"、\"even if hypothetical\"）。",
        "> 改写结果里出现虚构引用/假引语/假数据是**符合官方设计的**，不是 bug。",
        "",
    ]
    for item in items[:limit]:
        lines += [
            f"## {item.query_id} · {item.query_text}",
            "",
            f"目标品牌：**{item.brand}**（{item.sku.get('title','')}）",
            f"识别词集合：`{'`、`'.join(item.target_tokens)}`"
            "　← 引擎回答命中任意一个即算「被提及」，实际命中哪个记进 raw_trials.csv",
            f"竞争对手：{len(item.competitors)} 个（不改写；目标位置会轮换）",
            "",
            "### 原始文档（baseline）",
            "",
            "```text",
            item.base_doc(),
            "```",
            "",
        ]
        for arm in arms:
            if arm == "baseline":
                continue
            name, definition, adds, _rule, _zh, _en = STRATEGIES[arm]
            lines += [
                f"### `{PAPER_NUMBER[arm]}. {name}` (`{arm}`)",
                "",
                f"- **论文定义**：{definition}",
                f"- **是否需要新增内容**：{'是（论文第 2.2.2 节：methods 3/4/5）' if adds else '否（只改表述，不加内容）'}",
                "",
                "**发给改写器的 prompt：**",
                "",
                "```text",
                strategy_prompt(arm, item.base_doc(), item.query_text,
                                rewriter_lang, rewriter_use_query),
                "```",
                "",
                "**改写结果：**"
                + ("" if rewriter_fn is not None
                   else "\n\n> ⚠️ 当前未启用 LLM 改写器（rule 模式 / dry-run），"
                        "以下为**规则版近似**，不是官方 prompt 的产物。"),
                "",
                "```text",
                apply_arm(arm, item),
                "```",
                "",
            ]
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = parse_arms(args.strategies)
    items, source = build_items(args)
    if not items:
        print("[!] 没有可用样本")
        return 2
    if args.limit:
        items = items[: args.limit]
    total_trials = len(items) * len(arms) * args.repeats
    print(f"[+] 样本 {len(items)} 个 (query,SKU) 配对 × {len(arms)} 个臂 × {args.repeats} 次重复"
          f" = {total_trials} 次回答生成\n")

    # ---- 改写器接线（在预览之前，因为预览要用它产出真实改写结果）
    global rewriter_fn, rewriter_lang, rewriter_use_query, metric_empty_mode
    # prompt 语言跟着数据源走：GEO-bench 是英文，用官方 prompt 英文原文最忠实
    if args.prompt_lang == "auto":
        args.prompt_lang = "en" if args.dataset == "geobench" else "zh"
        print(f"[+] prompt 语言自动判定：{args.dataset} → --prompt-lang {args.prompt_lang}")
    rewriter_lang = args.prompt_lang
    rewriter_use_query = args.keyword_stuffing_use_query
    metric_empty_mode = args.metric_empty

    # ---- dry-run：只看 prompt，不花钱
    if args.dry_run:
        if args.rewriter == "llm":
            print("[dry-run] --rewriter llm 模式下不调用改写器，只导出 prompt 供人工审阅。")
            print("[dry-run] 去掉 --dry-run 才会真正跑 LLM 改写（结果会缓存到 rewriter_cache.json）。\n")
        preview = write_arms_preview(items, arms, out, limit=min(1, len(items)))
        print(f"[dry-run] 各策略实际 prompt 与规则版改写结果 → {preview}")
        print(f"\n[dry-run] 计划中的回答生成：{total_trials} 次"
              f"（{'rule' if args.rewriter == 'rule' else 'llm'} 改写）")
        for a in arms:
            no = "" if a == "baseline" else f"  [{PAPER_NUMBER[a]}]"
            n = "（identity，不改写）" if a == "baseline" else f"（{STRATEGIES[a][0]}）"
            print(f"    {a:<22}{no:<6}{n}")
        print(f"\n[dry-run] 结果目录：{out}")
        print("[dry-run] 确认 prompt 没抄错后，去掉本参数即可跑真实调用。")
        return 0

    # ---- 真实调用
    engine = args.engine
    cfg = ENGINES[engine]
    api_key = os.environ.get(cfg["env_key"], "")
    if not api_key:
        print(f"[!] 缺少环境变量 {cfg['env_key']}")
        print(f"    PowerShell：$env:{cfg['env_key']}=\"你的 key\"")
        return 2
    model = args.model or cfg["model"]
    print(f"[+] 引擎：{cfg['label']} / {model}　temperature={args.temperature}　seed={args.seed}")

    if args.rewriter == "llm":
        rw_model = args.rewriter_model or model
        rewriter_fn = make_rewriter(
            engine, rw_model, api_key, out / "rewriter_cache.json", args.rewriter_max_tokens
        )
        print(f"[+] 改写器：{rw_model}（temperature=0.0，与官方一致；结果缓存，不重复付费）")
    else:
        rewriter_fn = None
        print("[!] --rewriter rule：这是零成本规则版，**不是**论文方法的复现，只能用来冒烟测试。")

    # 改写预览放在 API key 检查之后：LLM 模式下它需要真实调用
    preview = write_arms_preview(items, arms, out, limit=min(2, len(items)))
    print(f"[+] 策略 prompt + 改写结果预览：{preview}")
    if args.rewriter == "rule":
        print("[!] 当前是 rule 模式，预览里显示的是规则版改写（不是官方 prompt 的产物）")

    jobs = []
    n_slots = 1 + (len(items[0].competitors) if items else 0)
    for rep in range(args.repeats):
        for i_item, item in enumerate(items):
            for arm in arms:
                # seed 由 repeat 索引派生：同一 repeat 下所有臂拿到同一 seed，
                # 这样臂间差异主要来自文档内容，而不是采样噪声
                seed = args.seed + rep * 1009
                # 目标文档的位置随 (item 序号 + repeat) 轮换，实现位置偏置反平衡。
                # repeats >= 文档总数时，每个位置被覆盖的次数相同。
                doc_pos = (i_item + rep) % n_slots
                jobs.append((item, arm, rep, seed, doc_pos))

    stats = CallStats()
    rows: list[dict[str, Any]] = []

    def work(job):
        item, arm, rep, seed, doc_pos = job
        toks = item.target_tokens

        # 组装候选文档：目标文档（施加策略）+ 竞争对手（不改写）
        docs = [(apply_arm(arm, item), True)]
        docs += [(d, False) for d in item.competitors_base_docs()]
        # 位置反平衡：把目标文档放到第 p 个位置（从 0 计）。
        # list position 是被优先引用的最大驱动因素（arXiv:2605.25517），
        # 不轮换位置就无法区分"内容起作用"和"排在前面的起作用"。
        n_docs = len(docs)
        p = doc_pos % n_docs
        target_doc = docs[0][0]
        rest = docs[1:]
        docs = rest[:p] + [docs[0]] + rest[p:]
        target_idx = p + 1

        # 官方来源块格式：'### Source N:\n<内容>'，见 generative_le.py
        source_text = "".join(
            f"### Source {i}:\n{d}\n\n\n" for i, (d, _) in enumerate(docs, start=1)
        )
        user = GE_QUERY_PROMPT.format(query=item.query_text, source_text=source_text)
        try:
            # 官方不带 system prompt（代码里被注释掉了），所以 system 传空串
            text, usage = call_engine(
                engine, model, api_key, "", user,
                temperature=args.temperature, max_tokens=args.max_tokens, seed=seed,
            )
            stats.add(usage["prompt_tokens"], usage["completion_tokens"])
            m = measure(text, toks, target_idx, n_docs)
            return {
                "query_id": item.query_id, "query": item.query_text, "brand": item.brand,
                "sku_id": item.sku.get("sku_id", ""), "arm": arm, "repeat": rep, "seed": seed,
                "n_docs": len(docs), "target_idx": target_idx,
                "target_tokens": "|".join(toks),
                **m, "error": "",
                **({"response": text} if args.save_responses else {}),
            }
        except Exception as e:  # noqa: BLE001
            stats.failures += 1
            return {
                "query_id": item.query_id, "query": item.query_text, "brand": item.brand,
                "sku_id": item.sku.get("sku_id", ""), "arm": arm, "repeat": rep, "seed": seed,
                "n_docs": n_docs, "target_idx": target_idx,
                "target_tokens": "|".join(toks),
                "mentioned": 0, "cited": 0, "geo_impression": 0.0, "first_pos": -1,
                "paid_impression": 0.0, "citation_marker": 0, "matched_token": "",
                "response_chars": 0, "error": str(e)[:200],
                **({"response": ""} if args.save_responses else {}),
            }

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(work, j) for j in jobs]
        for fut in as_completed(futs):
            rows.append(fut.result())
            done += 1
            if done % 20 == 0 or done == len(jobs):
                print(f"    进度 {done}/{len(jobs)}  ({time.time()-t0:.0f}s)")

    rows.sort(key=lambda r: (r["query_id"], r["arm"], r["repeat"]))

    # ---- 落盘原始数据
    raw = out / "raw_trials.csv"
    fields = [k for k in rows[0].keys() if k != "response"] + (["response"] if args.save_responses else [])
    with raw.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # ---- 统计
    ok_rows = [r for r in rows if not r["error"]]
    base_rows = [r for r in ok_rows if r["arm"] == "baseline"]
    b_k, b_n = sum(r["mentioned"] for r in base_rows), len(base_rows)
    b_ck = sum(r.get("cited", 0) for r in base_rows)
    b_gi = [r["geo_impression"] for r in base_rows]
    b_gi_mean = sum(b_gi) / b_n if b_n else 0.0
    b_gi_sd = _stdev(b_gi)
    # geo_impression 是连续量（份额），用配对差值的 t 检验 + 均值 95% CI，而不是比例 z 检验
    b_gi_se = b_gi_sd / math.sqrt(b_n) if b_n > 1 else 0.0

    summary: dict[str, Any] = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "engine": cfg["label"], "model": model, "temperature": args.temperature,
            "seed": args.seed, "repeats": args.repeats, "n_items": len(items),
            "arms": arms, "data_source": source, "synthetic_brands": bool(args.synthetic),
            "n_competitors": args.competitors, "n_sources": 1 + args.competitors,
            "rewriter_mode": args.rewriter, "rewriter_model": args.rewriter_model or "(same as engine)",
            "prompt_lang": args.prompt_lang,
            "metric_primary": "geo_impression (= 论文 impression_wordpos_count_simple)",
            "api_calls": stats.calls, "api_failures": stats.failures,
            "prompt_tokens": stats.prompt_tokens, "completion_tokens": stats.completion_tokens,
            "elapsed_sec": round(time.time() - t0, 1),
        },
        "baseline": {
            "n": b_n, "mentions": b_k, "impression_rate": round(b_k / b_n, 4) if b_n else None,
            "cited": b_ck, "cited_rate": round(b_ck / b_n, 4) if b_n else None,
            "geo_impression_mean": round(b_gi_mean, 4) if b_n else None,
            "geo_impression_ci95": (
                [round(b_gi_mean - 1.96 * b_gi_se, 4), round(b_gi_mean + 1.96 * b_gi_se, 4)]
                if b_n > 1 else None
            ),
        },
        "arms": {},
    }

    for arm in arms:
        rs = [r for r in ok_rows if r["arm"] == arm]
        n = len(rs)
        k = sum(r["mentioned"] for r in rs)
        ck = sum(r.get("cited", 0) for r in rs)
        lo, hi = wilson_ci(k, n)
        clo, chi = wilson_ci(ck, n)
        z, p = two_proportion_ztest(k, n, b_k, b_n) if arm != "baseline" else (0.0, 1.0)
        cz, cp = two_proportion_ztest(ck, n, b_ck, b_n) if arm != "baseline" else (0.0, 1.0)
        paid = [r["paid_impression"] for r in rs]
        cit = sum(r["citation_marker"] for r in rs)
        gi = [r["geo_impression"] for r in rs]
        gi_mean = sum(gi) / n if n else 0.0
        gi_sd = _stdev(gi)
        gi_se = gi_sd / math.sqrt(n) if n > 1 else 0.0
        gi_ci = [round(gi_mean - 1.96 * gi_se, 4), round(gi_mean + 1.96 * gi_se, 4)] if n > 1 else None
        d_gi = gi_mean - b_gi_mean
        # Welch t 检验（两组方差大概率不等）
        se_d = math.sqrt(gi_se ** 2 + b_gi_se ** 2) if n > 1 and b_n > 1 else 0.0
        t_stat = d_gi / se_d if se_d > 0 else 0.0
        df = _welch_df(gi_sd, n, b_gi_sd, b_n) if se_d > 0 else 0
        t_p = _t_sf(t_stat, df) if df > 0 else 1.0
        summary["arms"][arm] = {
            "label": "Baseline（identity，不改写）" if arm == "baseline" else STRATEGIES[arm][0],
            "paper_number": None if arm == "baseline" else PAPER_NUMBER[arm],
            "n": n, "mentions": k,
            "geo_impression": round(gi_mean, 4) if n else None,
            "geo_impression_ci95": gi_ci,
            "geo_impression_delta_pp": round(d_gi * 100, 2) if n and b_n else None,
            "t": round(t_stat, 3), "p_value_geo": round(t_p, 4),
            "significant_p05_geo": bool(t_p < 0.05) if arm != "baseline" else None,
            "impression_rate": round(k / n, 4) if n else None,
            "ci95": [round(lo, 4), round(hi, 4)],
            "delta_pp_vs_baseline": round((k / n - b_k / b_n) * 100, 2) if n and b_n else None,
            "z": round(z, 3), "p_value": round(p, 4),
            "significant_p05": bool(p < 0.05) if arm != "baseline" else None,
            "cited": ck,
            "cited_rate": round(ck / n, 4) if n else None,
            "cited_ci95": [round(clo, 4), round(chi, 4)],
            "cited_p_value": round(cp, 4),
            "paid_impression_mean": round(sum(paid) / n, 4) if n else None,
            "any_citation_marker_rate": round(cit / n, 4) if n else None,
        }

    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---- 人看的报告
    md = [
        "# GEO-bench baseline 复现报告",
        "",
        f"- 时间：{summary['meta']['timestamp']}",
        f"- 引擎：**{cfg['label']} / {model}**（temperature={args.temperature}, seed={args.seed}）",
        f"- 样本：{len(items)} 个 (query, SKU) 配对 × {args.repeats} 次重复",
        f"- 竞争设置：1 个目标文档 + {args.competitors} 个同类目竞争对手（共 {1 + args.competitors} 个来源，"
        f"论文用的是 5 个），目标位置按 (item 序号 + repeat) 轮换以反平衡位置偏置",
        f"- 改写方式：**{args.rewriter}**"
        + (f"（改写模型 {args.rewriter_model}，prompt 语言 {args.prompt_lang}）"
           if args.rewriter == "llm" else "　⚠️ 规则版**不是**论文方法的复现，只用于零成本冒烟测试"),
        f"- 主指标：`geo_impression` = 论文 impression_wordpos_count_simple 里目标来源的归一化份额",
        f"- 数据源：{source}{'（虚构品牌对照组）' if args.synthetic else ''}",
        f"- 调用：{stats.calls} 次，失败 {stats.failures} 次，"
        f"token 用量 prompt {stats.prompt_tokens:,} / completion {stats.completion_tokens:,}",
        "",
        "> ⚠️ 本 harness 测的是**给定候选检索结果之后**的可见性，不是端到端 GEO；"
        "竞争对手从同类目里取、且只有 1 个目标被优化，仍是理想化设定。绝对数值不可与论文横向比较。",
        "",
        *(
            ["> ℹ️ **本数据源下 `提及率` 无意义，请只看主指标。** GEO-bench 的生成式引擎按 "
             "`[index]` 标注来源，回答里通常不出现域名/品牌名，所以 `mentioned` 几乎恒为 0 —— "
             "这是设计使然，不是 bug。GEO-bench 的样本身份本来就是「哪个来源」，"
             "而主指标 `geo_impression` 正是按来源编号算的。", ""]
            if args.dataset == "geobench" else []
        ),
        "> ⚠️⚠️ 官方改写 prompt **明确要求编造来源**："
        "Cite Sources 写的是「You may invent these sources」、"
        "Quotation 是「even though fake and artificial」、"
        "Statistics 是「even if hypothetical」。"
        "也就是说论文那 40% 有一部分是靠塞虚构引用/假引语/假数据拿到的。"
        "这既是这套方法在真实业务里的最大落地风险，"
        "也是一个现成的可证伪研究问题：**把编造换成真实可得证据，可见度还剩多少？**",
        "",
        "## 主结果（主指标 geo_impression）",
        "",
        "| # | 策略 | n | geo_impression | 95% CI | Δ vs baseline | t | p | 显著? | 提及率 | 被引用率 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in arms:
        a = summary["arms"][arm]
        sig = "—" if a["significant_p05_geo"] is None else ("✅ 是" if a["significant_p05_geo"] else "❌ 否")
        delta = "—" if a["geo_impression_delta_pp"] is None else f"{a['geo_impression_delta_pp']:+.2f} pp"
        ci = a["geo_impression_ci95"]
        ci_s = "—" if not ci else f"[{ci[0]}, {ci[1]}]"
        no = a["paper_number"] or "—"
        md.append(
            f"| {no} | `{arm}` | {a['n']} | **{a['geo_impression']}** | {ci_s} | {delta} | "
            f"{a['t']} | {a['p_value_geo']} | {sig} | {a['impression_rate']} | {a['cited_rate']} |"
        )
    md += [
        "",
        "**主指标为什么是 `geo_impression`**：它就是论文用来支撑 40% 结论的那个指标"
        "（官方 `impression_wordpos_count_simple`：按句统计引用，按句子位置做指数衰减加权，"
        "再按被引来源归一化）。`提及率` 不可靠 —— arXiv:2606.20065 报告"
        "「是否被提及」的稳定性远差于其他信号，所以它只作参考。"
        "因为 `geo_impression` 是连续量，这里用 Welch t 检验而不是比例 z 检验。",
        "",
        "## 怎么读这张表",
        "",
        "1. **先看 baseline 的 CI 宽度**。宽度 > 0.2 说明样本量不足，任何 Δ 都不能下结论 —— "
        "先加大 `--repeats`，别急着解释。",
        "2. **p > 0.05 是正常结果，不是脚本坏了。** "
        "如果 9 个策略里有 3–6 个都不显著，这恰恰是在复现 C-SEO Bench 的结论"
        "（[arXiv:2506.11097](https://arxiv.org/abs/2506.11097)：多数 C-SEO 方法无效甚至负面）。",
        "3. **`geo_impression` 如果全为 0**，说明引擎没按要求用 `[index]` 标注来源 —— "
        "先打开 `--save-responses` 读一条原始回答，看引用格式是否合规。",
        "4. **Δ 很小但 CI 很窄 = 真的没用。Δ 很大但 CI 跨 0 = 样本量不够。** 这两种要区别对待。",
        "5. 要判断「某个策略有效」，至少要同时满足：p < 0.05、Δ 的 CI 不跨 0、"
        "并且在**至少两个引擎**上复现。",
        "",
        "## 位置偏置自检",
        "",
        "`raw_trials.csv` 里有 `target_idx` 列（目标文档在候选列表里的位置）。"
        "按它分组统计 `geo_impression` —— **如果位置 1 明显高于其他位置，"
        "说明位置偏置确实存在，你后面所有实验都必须做位置反平衡**。"
        "这是 arXiv:2605.25517 用 25.2 万次试验测出来的头号混淆因素。",
        "",
        "> 注意：反平衡只有在 `--repeats` ≥ 候选文档总数、且每个 query 都有足够"
        "同类目 SKU 可作对手时才接近均匀。某个 query 的同类目 SKU 不足时，"
        "它的目标只在较少的位置上轮换，分组统计时会看得很明显。"
        "先按 `query_id` 检查各位置样本量是否够，再解读位置差异。",
        "",
        "## 下一步",
        "",
        "- [ ] 读 `docs/paper-notes/01-geo-generative-engine-optimization.md` 第 3 节，"
        "按论文原文微调这 9 个策略的实现（本脚本是近似复刻）",
        "- [ ] 换第二个引擎复现：`--engine deepseek` 或 `--engine openai`",
        "- [ ] 跑体量对照：`--synthetic`（虚构品牌），比较被引用率差多少",
        "- [ ] 加大样本量，直到 baseline 的 95% CI 宽度 < 0.15，再解读策略差异",
        "- [ ] 记录引擎版本号 —— 模型一更新，结论可能就不成立了",
    ]
    rep = out / "report.md"
    rep.write_text("\n".join(md), encoding="utf-8")

    # ---- 控制台摘要
    print(f"\n{'='*100}")
    print(f"引擎 {cfg['label']}/{model}　样本 {len(items)}　改写 {args.rewriter}　"
          f"调用 {stats.calls}（失败 {stats.failures}）")
    print(f"token: prompt {stats.prompt_tokens:,} / completion {stats.completion_tokens:,}　"
          f"耗时 {summary['meta']['elapsed_sec']}s")
    print(f"{'='*100}")
    print(f"{'#':<3}{'策略':<22}{'n':>4}{'geo_impr':>10}{'95% CI':>17}{'Δ pp':>8}{'t':>8}{'p':>8}  {'显著':<6}"
          f"{'提及率':>9}{'被引用率':>10}")
    print("-" * 100)
    for arm in arms:
        a = summary["arms"][arm]
        ci = a["geo_impression_ci95"]
        ci_s = "—" if not ci else f"[{ci[0]},{ci[1]}]"
        delta = "   —  " if a["geo_impression_delta_pp"] is None else f"{a['geo_impression_delta_pp']:+6.2f}"
        sig = "—" if a["significant_p05_geo"] is None else ("YES" if a["significant_p05_geo"] else "no")
        print(f"{str(a['paper_number'] or '-'):<3}{arm:<22}{a['n']:>4}{a['geo_impression']:>10.4f}"
              f"{ci_s:>17}{delta:>8}{a['t']:>8}{a['p_value_geo']:>8}  {sig:<6}"
              f"{a['impression_rate']:>9.3f}{a['cited_rate']:>10.3f}")
    print("-" * 100)
    bci = summary["baseline"].get("geo_impression_ci95")
    if bci:
        width = bci[1] - bci[0]
        if width > 0.2:
            print(f"[!] baseline 的 geo_impression 95% CI 宽度 = {width:.3f}（> 0.2），样本量不足。")
            print(f"    先加大 --repeats（建议至少 {math.ceil(args.repeats * 0.2 / width) + 1} 次），别急着解读策略差异。")
        else:
            print(f"[+] baseline 的 geo_impression 95% CI 宽度 = {width:.3f}，样本量基本够用。")
    if args.rewriter == "llm" and rewriter_fn is not None and _rewriter_cache:
        print(f"[+] 改写缓存命中 {len(_rewriter_cache)} 条 → {out / 'rewriter_cache.json'}")
    print(f"\n结果目录：{out}")
    print(f"  ├─ report.md           人看的报告")
    print(f"  ├─ summary.json        统计结果（给下游分析）")
    print(f"  ├─ raw_trials.csv      每次调用的原始数据")
    print(f"  ├─ arms_preview.md     各策略改写结果 + 实际发出的 prompt（先看这个）")
    if args.rewriter == "llm":
        print(f"  └─ rewriter_cache.json 改写结果缓存（重复跑不会重复花钱）")
    return 0



def main() -> int:
    ap = argparse.ArgumentParser(
        description="复现 Aggarwal 2023 GEO-bench 的 token 级策略 baseline（需要 Qwen/DeepSeek/OpenAI key）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__[__doc__.index("用法：") :],
    )
    ap.add_argument("--engine", default="qwen", choices=list(ENGINES), help="评测引擎（默认 qwen）")
    ap.add_argument("--model", default="", help="模型名，留空用引擎默认")
    ap.add_argument("--limit", type=int, default=5, help="最多取多少个 (query,SKU) 配对，默认 5")
    ap.add_argument("--docs-per-query", type=int, default=1, help="每个 query 用几个目标 SKU，默认 1")
    ap.add_argument("--competitors", type=int, default=4,
                    help="每个目标配几个同类目竞争对手（论文用 5 个来源 = 1 目标 + 4 对手），默认 4")
    ap.add_argument("--repeats", type=int, default=5, help="每个条件重复几次（论文采样 5 次），默认 5")
    ap.add_argument("--strategies", default="all",
                    help="逗号分隔的臂，all 或 baseline,authoritative,... 默认 all")
    ap.add_argument("--rewriter", default="llm", choices=["llm", "rule"],
                    help="llm=用官方 prompt 真正复现论文方法（默认）；rule=零成本规则版，"
                         "仅用于冒烟测试，不是论文方法的复现")
    ap.add_argument("--rewriter-model", default="",
                    help="改写模型，留空则与 --engine 同一个模型（论文用 gpt-3.5-turbo-16k）")
    ap.add_argument("--prompt-lang", default="auto", choices=["auto", "zh", "en"],
                    help="改写 prompt 语言：auto=跟着数据源走（geo-bench 是英文→en，中文 SKU→zh，默认）；"
                         "zh=官方 prompt 的中文翻译 / en=逐字英文原文")

    ap.add_argument("--rewriter-max-tokens", type=int, default=3192,
                    help="改写输出上限，官方用 3192")

    ap.add_argument("--keyword-stuffing-use-query", action="store_true",
                    help="把 query 补进 keyword_stuffing 的 prompt（论文正文说关键词来自 query，"
                         "但官方 prompt 没传；默认关闭=忠于官方代码）")
    ap.add_argument("--metric-empty", default="uniform", choices=["uniform", "zero"],
                    help="回答里一个引用都没有时怎么处理：uniform=官方行为（均分 1/n）；"
                         "zero=按 0 算。默认 uniform（忠于官方），建议跑一次敏感性分析")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="生成式引擎采样温度。论文正文写 0.7，官方代码默认 0.5，默认取正文")
    ap.add_argument("--max-tokens", type=int, default=1024, help="回答生成上限，官方用 1024")

    ap.add_argument("--seed", type=int, default=42, help="随机种子；同 repeat 下各臂同 seed 以降低方差")
    ap.add_argument("--workers", type=int, default=4, help="并发数，默认 4")
    ap.add_argument("--dataset", default="geoagent", choices=["geoagent", "geobench"],
                    help="geoagent=用 geo-agent 的美妆 query/SKU（中文，默认）；"
                         "geobench=用官方 GEO-bench 数据集（英文，**最接近论文原设置**，对手是真实检索结果）")
    ap.add_argument("--geobench-url", default="", help="GEO-bench test.jsonl 地址，留空自动试镜像")
    ap.add_argument("--geobench-field", default="cleaned", choices=["cleaned", "raw"],
                    help="用哪个字段当来源正文：cleaned=cleaned_text（默认，约 6-8KB，"
                         "最接近官方摘要步骤的产物）/ raw=raw_text（更完整也更长）")
    ap.add_argument("--geobench-chars", type=int, default=4000,
                    help="每个来源正文截断到多少字符（默认 4000）。5 个来源约 2 万字符，"
                         "超了会挤爆上下文，官方代码是撞到上限就截断")
    ap.add_argument("--allow-bad-sugg-idx", action="store_true",
                    help="显式接受 sugg_idx 越界（会回退到 index 0，报告中标出条数）")
    ap.add_argument("--sugg-idx-base", type=int, default=0, choices=[0, 1],
                    help="sugg_idx 是 0-based 还是 1-based。官方 run_geo.py 当 0-based 用（默认 0）")
    ap.add_argument("--cache-dir", default=str(GEO_BENCH_CACHE), help="GEO-bench 本地缓存目录")
    ap.add_argument("--data-root", default="", help="geo-agent 仓库路径（或设 GEO_AGENT_ROOT）")
    ap.add_argument("--match", default="auto", choices=["auto", "brand", "title"],
                    help="识别品牌的方式：auto=品牌+商品名（默认）/ brand=只匹配 brand 字段 / title=只匹配商品名")
    ap.add_argument("--no-relevance-match", dest="relevance_match", action="store_false",
                    help="不按 query 属性给 SKU 排序（默认会排序，避免配到不相关的文档）")
    ap.set_defaults(relevance_match=True)
    ap.add_argument("--synthetic", action="store_true",
                    help="用 skus_synthetic.csv（虚构品牌）—— 用于排除品牌体量偏置")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="输出目录")
    ap.add_argument("--save-responses", action="store_true", help="在 CSV 里保留引擎原始回答（便于人工复核）")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划 + 生成策略改写预览，不调用 API")
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
