import logging
from pathlib import Path

import requests as r
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import TokenTextSplitter

FMP_KEY = ''
OPENAI_KEY = ''
TOP_K_COMPANIES = 10


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


def get_transcripts(symbols: list[str]) -> dict[str, str]:
    symbol_to_transcript = {}
    for symbol in symbols:
        try:
            symbol_to_transcript[symbol] = get_transcript(symbol)
        except r.HTTPError:
            logging.error(f'{symbol} not found')
        if len(symbol_to_transcript) == TOP_K_COMPANIES:
            break
    return symbol_to_transcript


def summarize_article(article: str) -> str:
    splitter = TokenTextSplitter(
        chunk_overlap=102,
        chunk_size=2048,
        model_name='gpt-3.5-turbo',
    )
    paragraphs = splitter.split_text(article)
    jobs = [
        [
            SystemMessage(
                content='You are a summarizer. You summarize an article thoroughly. Here is the article:'
            ),
            HumanMessage(content=paragraph),
            HumanMessage(content='Thorough Summary:'),
        ]
        for paragraph in paragraphs
    ]
    llm = ChatOpenAI(
        max_retries=2,
        model_name='gpt-3.5-turbo',
        openai_api_key=OPENAI_KEY,
        temperature=0.5,
    )  # type: ignore
    results = llm.generate(jobs).generations
    summary = '\n'.join(e[0].text for e in results)
    if len(results) > 1:
        return summarize_article(summary)
    return summary


def get_summary(symbol: str) -> str:
    transcript = get_transcript(symbol)
    return summarize_article(transcript)


# get_summary('MSFT')


def load_summary(symbol: str) -> str:
    path = Path('cache') / (symbol + '.txt')
    if not path.is_file():
        path.write_text(get_summary(symbol))
    return path.read_text()


def load_summaries() -> dict[str, str]:
    symbols = get_symbols()
    symbol_to_summary = {}
    for symbol in symbols:
        try:
            symbol_to_summary[symbol] = load_summary(symbol)
        except r.HTTPError:
            logging.error(f'{symbol} not found')
        if len(symbol_to_summary) == TOP_K_COMPANIES:
            break
    return symbol_to_summary

load_summary('MSFT')
