# %%
# use cudf.pandas 
import cudf
import pandas as pd 
import logging
import hashlib
import re
import json
from bisect import bisect_right

import sys 
sys.path.append("/share/ju/matt/sensing-ai-risks")
from utils.logger import setup_logger

logger = setup_logger("processing.keyword")
logger.setLevel(logging.INFO)

# %%
articles_path = "/share/ju/matt/sensing-ai-risks/data/global_subset/all_articles.parquet"
logger.info(f"Loading articles from {articles_path}")
articles = pd.read_parquet(articles_path)
logger.success(f"Loaded {len(articles)} articles with {articles.shape[1]} columns")


# %%
logger.info(
    f"Columns available: {len(articles.columns)} | Example columns: {list(articles.columns)[:10]}"
)

# %%
logger.info("Computing keyword flags (methods and domains)")
articles['contains_ai'] = articles['article_text'].str.lower().str.contains("artificial intelligence")

# %%
articles['contains_ml'] = articles['article_text'].str.lower().str.contains("machine learning")


# %%
articles['contains_computer'] = articles['article_text'].str.lower().str.contains("computer")

# %%
articles['contains_model'] = articles['article_text'].str.lower().str.contains("model")

# %%
articles['contains_city'] = articles['article_text'].str.lower().str.contains("city|cities", regex=True)

# %%
articles['contains_urban'] = articles['article_text'].str.lower().str.contains("urban")

# %%
articles['contains_climate'] = articles['article_text'].str.lower().str.contains("climate")

# %%
articles['contains_earth'] = articles['article_text'].str.lower().str.contains("earth")

# %%
articles['contains_environment'] = articles['article_text'].str.lower().str.contains("environment")

# %%
articles['contains_transportation'] = articles['article_text'].str.lower().str.contains("transport")

# %%
method_cols = ['contains_ai', 'contains_ml', 'contains_computer', 'contains_model']
domain_cols = ['contains_city', 'contains_urban', 'contains_climate', 'contains_earth', 'contains_environment', 'contains_transportation']

method_counts = {col: int(articles[col].sum()) for col in method_cols}
domain_counts = {col: int(articles[col].sum()) for col in domain_cols}
logger.info(f"Method keyword matches: {method_counts}")
logger.info(f"Domain keyword matches: {domain_counts}")

# %%
# Build a single compiled regex to find any relevant term (case-insensitive)
def build_relevant_regex() -> re.Pattern:
    phrases = [
        r"artificial\s+intelligence",
        r"machine\s+learning",
        r"computer",
        r"model",
        r"(?:city|cities)",
        r"urban",
        r"climate",
        r"earth",
        r"environment",
        r"transport",
    ]
    pattern = r"(" + r"|".join(phrases) + r")"
    return re.compile(pattern, flags=re.IGNORECASE)


RELEVANT_REGEX = build_relevant_regex()

# %%
# an article is relevant if at least ONE of method_cols is True AND at least ONE of domain_cols is True
articles['relevant'] = (articles[method_cols].any(axis=1) & articles[domain_cols].any(axis=1))
num_relevant = int(articles['relevant'].sum())
total_articles = len(articles)
logger.info(
    f"Computed relevance: {num_relevant}/{total_articles} articles ({(num_relevant/total_articles):.1%})"
)

# %%
articles['total_methods_mentioned'] = articles[method_cols].sum(axis=1)
articles['total_domains_mentioned'] = articles[domain_cols].sum(axis=1)

articles['total_mentions'] = articles['total_methods_mentioned'] + articles['total_domains_mentioned']

# sort by total_mentions
articles = articles.sort_values(by='total_mentions', ascending=False)
logger.info("Sorted articles by total keyword mentions")


# %%
relevant = articles[articles['relevant']]
logger.info(f"Selected relevant subset with {len(relevant)} rows")

# %%
logger.info(
    f"Relevant columns: {len(relevant.columns)} | Example columns: {list(relevant.columns)[:10]}"
)

# %%
# keep article_path; it will be used as a stable key for tracing and IDs

# %%
relevant = relevant.as_cpu_object()
logger.info("Converted relevant subset to CPU object")

# add a stable per-article ID derived from article_path (if present)
if 'article_path' in relevant.columns:
    relevant['article_id'] = relevant['article_path'].apply(
        lambda p: hashlib.sha1(str(p).encode('utf-8')).hexdigest()
    )
    logger.info("Added 'article_id' as sha1(article_path)")

# %%
# Generate merged ±100-word buffers around each relevant match and store as JSON
def generate_relevant_blocks(text: str, compiled_regex: re.Pattern, window_words: int = 100) -> list[str]:
    if not isinstance(text, str) or not text:
        return []

    # Tokenize by contiguous non-space sequences and keep positions
    token_matches = list(re.finditer(r"\S+", text))
    if not token_matches:
        return []

    token_starts = [m.start() for m in token_matches]
    intervals: list[tuple[int, int]] = []

    for m in compiled_regex.finditer(text):
        # locate token index that contains the match start
        idx = max(0, min(len(token_starts) - 1, bisect_right(token_starts, m.start()) - 1))
        start_token = max(0, idx - window_words)
        end_token = min(len(token_matches) - 1, idx + window_words)
        start_char = token_matches[start_token].start()
        end_char = token_matches[end_token].end()
        intervals.append((start_char, end_char))

    if not intervals:
        return []

    # Merge overlapping/adjacent intervals
    intervals.sort(key=lambda x: x[0])
    merged: list[list[int]] = []
    for s, e in intervals:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    # Extract text blocks
    blocks = [text[s:e] for s, e in merged]
    return blocks


logger.info("Generating relevant text blocks (±100 words) for each relevant article")
_blocks = relevant['article_text'].apply(lambda t: generate_relevant_blocks(t, RELEVANT_REGEX, 100))
relevant['relevant_blocks'] = _blocks.apply(lambda blks: json.dumps(blks, ensure_ascii=False))
relevant['relevant_blocks_count'] = _blocks.apply(len)
logger.info(
    f"Generated blocks for {len(relevant)} articles; total blocks: {int(relevant['relevant_blocks_count'].sum())}"
)

# %%
# Compute token counts for each block and totals; log concise stats
def _count_tokens_whitespace(text: str) -> int:
    if not isinstance(text, str) or not text:
        return 0
    return len(re.findall(r"\S+", text))


_block_token_counts = _blocks.apply(lambda blks: [_count_tokens_whitespace(b) for b in blks])
relevant['relevant_blocks_token_counts'] = _block_token_counts.apply(lambda counts: json.dumps(counts))
relevant['total_relevant_tokens'] = _block_token_counts.apply(sum)

_total_blocks = int(relevant['relevant_blocks_count'].sum())
_total_tokens = int(relevant['total_relevant_tokens'].sum())
_avg_tokens_per_block = (_total_tokens / _total_blocks) if _total_blocks else 0
logger.info(
    f"Token stats — total blocks: {_total_blocks}, total tokens: {_total_tokens}, avg tokens/block: {_avg_tokens_per_block:.1f}"
)

# log the maximum number of tokens per block
logger.info(f"Maximum number of tokens per block: {int(relevant['total_relevant_tokens'].max())}")

# %%
# write relevant articles subset to its own parquet 
relevant_parquet_path = "../data/global_subset/relevant_articles.parquet"
logger.info(f"Writing relevant articles to parquet (dropping 'article_text'): {relevant_parquet_path}")
to_save = relevant.copy()
if 'article_text' in to_save.columns:
    to_save = to_save.drop(columns=['article_text'])
to_save.to_parquet(relevant_parquet_path)
logger.success("Wrote relevant articles parquet")

# %%
logger.info(
    f"Final relevant dataset: {len(relevant)} rows, {len(relevant.columns)} columns"
)

# %%
sample_csv_path = "/share/ju/matt/sensing-ai-risks/data/global_subset/relevant_articles_sample.csv"
logger.info(f"Writing sample CSV to {sample_csv_path}")
relevant.sample(n=10, random_state=42).to_csv(sample_csv_path, index=False)
logger.success("Wrote sample CSV")

# %%



