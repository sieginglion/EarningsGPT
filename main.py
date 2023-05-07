import logging
import re
from os import environ
from pathlib import Path
from typing import Any

import requests as r
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import TokenTextSplitter

load_dotenv()
logging.basicConfig(level=logging.INFO)

FMP_KEY = environ['FMP_KEY']
OPENAI_KEY = environ['OPENAI_KEY']
TOP_K_COMPANIES = 2


def get_symbols() -> list[str]:
    res = r.post(
        'https://scanner.tradingview.com/america/scan',
        json={
            'columns': ['name'],
            'filter': [
                {
                    'left': 'exchange',
                    'operation': 'in_range',
                    'right': ['AMEX', 'NASDAQ', 'NYSE'],
                },
                {'left': 'is_primary', 'operation': 'equal', 'right': True},
                {'left': 'type', 'operation': 'in_range', 'right': ['dr', 'stock']},
            ],
            'range': [0, TOP_K_COMPANIES * 2],
            'sort': {'sortBy': 'market_cap_basic', 'sortOrder': 'desc'},
        },
    )
    return [e['d'][0] for e in res.json()['data']]


def get_transcript(symbol: str) -> str:
    res = r.get(
        f'https://financialmodelingprep.com/api/v3/earning_call_transcript/{symbol}?apikey={FMP_KEY}'
    )
    if not (data := res.json()):
        res.status_code = 404
        res.raise_for_status()
    return data[0]['content']


def summarize_article(article: str) -> str:
    splitter = TokenTextSplitter(
        chunk_overlap=137,
        chunk_size=2731,
        model_name='gpt-3.5-turbo',
    )
    chunks = splitter.split_text(article)
    threads = [
        [
            SystemMessage(
                content='You are a financial analyst. Generate a comprehensive summary for the following article:'
            ),
            HumanMessage(content=chunk),
            HumanMessage(content='The comprehensive summary:'),
        ]
        for chunk in chunks
    ]
    llm = ChatOpenAI(
        max_retries=2,
        model_name='gpt-3.5-turbo',
        openai_api_key=OPENAI_KEY,
        temperature=0.5,
    )  # type: ignore
    results = llm.generate(threads).generations
    summary = '\n'.join(e[0].text for e in results)
    return summarize_article(summary) if len(results) > 1 else summary


def get_summary(symbol: str) -> str:
    return summarize_article(get_transcript(symbol))


def load_summary(symbol: str) -> str:
    path = Path('cache') / (symbol + '.txt')
    if not path.is_file():
        path.write_text(get_summary(symbol))
    return path.read_text()


def get_summaries() -> list[str]:
    summaries = []
    for symbol in get_symbols():
        try:
            summary = re.sub(r'\n+', ' ', load_summary(symbol).strip())
            summaries.append(f'{symbol}:\n{summary}\n\n')
            if len(summaries) == TOP_K_COMPANIES:
                break
        except r.HTTPError:
            logging.error(f'{symbol} not found')
    return summaries


class AtomicTextsSplitter(TokenTextSplitter):
    def __init__(self, max_chunk_size, model_name) -> None:
        super().__init__(
            chunk_overlap=0, chunk_size=max_chunk_size, model_name=model_name
        )

    def split_texts(self, texts: list[str]) -> list[str]:
        chunk = ''
        chunk_size = 0
        chunks = []
        for text in texts:
            text_size = len(self._tokenizer.encode(text))
            if chunk_size + text_size <= self._chunk_size:
                chunk += text
                chunk_size += text_size
            else:
                chunks.append(chunk)
                chunk = [text]
                chunk_size = text_size
        if chunk:
            chunks.append(chunk)
        return chunks


def extract_most_related_content(articles: list[str], query: str) -> str:
    splitter = AtomicTextsSplitter(max_chunk_size=2731, model_name='gpt-3.5-turbo')
    chunks = splitter.split_texts(articles)
    threads = [
        [
            SystemMessage(
                content='You are a financial analyst. Extract the content related to the question from the article.'
            ),
            HumanMessage(content=f'Article: {chunk}'),
            HumanMessage(content=f'Question: {query}'),
            HumanMessage(content='Content related to the question:'),
        ]
        for chunk in chunks
    ]
    llm = ChatOpenAI(
        max_retries=2,
        model_name='gpt-3.5-turbo',
        openai_api_key=OPENAI_KEY,
        temperature=0.5,
    )  # type: ignore
    results = llm.generate(threads).generations
    contents = [e[0].text for e in results]
    return (
        extract_most_related_content(contents, query)
        if len(contents) > 1
        else contents[0]
    )


summaries = get_summaries()
extract_most_related_content(summaries, 'What are the companies focusing on?')
