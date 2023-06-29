import re
from collections import defaultdict

import requests as r
from diskcache import Cache
from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.text_splitter import TokenTextSplitter
from more_itertools import first_true
from openai import ChatCompletion
from pydantic import BaseSettings


class Settings(BaseSettings):
    FMP_API_KEY: str
    OPENAI_API_BASE: str
    OPENAI_API_KEY: str
    OPENAI_API_VERSION: str
    OPENAI_DEPLOYMENT: str
    TOP_K: int


cache = Cache('.cache')
settings = Settings(_env_file='.env')


@cache.memoize()
def get_name_to_symbols(top_k: int):
    res = r.post(
        'https://scanner.tradingview.com/america/scan',
        json={
            'columns': ['name', 'description'],
            'filter': [
                {
                    'left': 'typespecs',
                    'operation': 'has_none_of',
                    'right': ['etn', 'etf'],
                }
            ],
            'range': [0, top_k * 2],
            'sort': {'sortBy': 'market_cap_basic', 'sortOrder': 'desc'},
            'markets': ['america'],
        },
    )
    name_to_symbols: dict[str, list[str]] = defaultdict(list)
    for e in res.json()['data']:
        symbol, name = e['d']
        symbol = symbol.replace('/', '-')
        name_to_symbols[name].append(symbol)
    return name_to_symbols


@cache.memoize()
def get_transcript(symbol: str):
    res = r.get(
        f'https://financialmodelingprep.com/api/v3/earning_call_transcript/{symbol}?apikey={settings.FMP_API_KEY}'
    )
    if res.ok and (data := res.json()):
        return data[0]['content']


def get_name_to_transcript(name_to_symbols: dict[str, list[str]]):
    name_to_transcript: dict[str, str] = {}
    for name, symbols in name_to_symbols.items():
        if transcript := first_true(get_transcript(symbol) for symbol in symbols):
            name_to_transcript[name] = transcript
            if len(name_to_transcript) == settings.TOP_K:
                break
    return name_to_transcript


def map_context(context: str, chunk_size: int, template: str):
    sys_msg = 'You are an experienced stock investor.'
    splitter = TokenTextSplitter(
        chunk_overlap=round(chunk_size * 0.05),
        chunk_size=chunk_size,
        model_name='gpt-3.5-turbo',
    )
    llm = AzureChatOpenAI(
        client=ChatCompletion,
        deployment_name=settings.OPENAI_DEPLOYMENT,
        openai_api_base=settings.OPENAI_API_BASE,
        openai_api_key=settings.OPENAI_API_KEY,
        openai_api_version=settings.OPENAI_API_VERSION,
        request_timeout=30,
        temperature=0.5,
    )
    prompt = ChatPromptTemplate.from_messages(
        messages=[
            SystemMessagePromptTemplate.from_template(template=sys_msg),
            HumanMessagePromptTemplate.from_template(template=template),
        ]
    )
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    _ = re.sub(r'\s+', ' ', context)
    _ = splitter.split_text(_)
    _ = map(chain.run, _)
    _ = filter(lambda x: 'null' not in x.lower(), _)
    return list(_)


def summarize_transcript(transcript: str):
    instruction = [
        'Summarize the earnings call transcript.',
    ]
    template = [
        'Transcript: \'\'\'{transcript}\'\'\'',
        'Instruction: ' + ' '.join(instruction),
    ]
    results = map_context(transcript, 2731, '\n'.join(template))
    return ' '.join(results)


def get_name_to_summary(name_to_transcript: dict[str, str]):
    name_to_summary: dict[str, str] = {}
    for name, transcript in name_to_transcript.items():
        key = ('summary', name)
        if not (summary := cache.get(key)):
            summary = summarize_transcript(transcript)
            cache[key] = summary
        name_to_summary[name] = summary
    return name_to_summary


def extract_content(context: str, question: str):
    instruction = [
        'Extract content related to the question from the context.',
    ]
    template = [
        'Context: \'\'\'{context}\'\'\'',
        'Question: ' + question,
        'Instruction: ' + ' '.join(instruction),
    ]
    results = map_context(context, 3277, '\n'.join(template))
    return ' '.join(results)


def get_name_to_content(name_to_summary: dict[str, str], question: str):
    return {
        name: content
        for name, summary in name_to_summary.items()
        if (content := extract_content(summary, question))
    }


def reduce_content(name_to_content: dict[str, str]):
    return ' '.join(f'{name}: {content}' for name, content in name_to_content.items())


def answer_question(context: str, question: str):
    instruction = [
        'Answer the question according to the context.',
    ]
    template = [
        'Context: \'\'\'{context}\'\'\'',
        'Question: ' + question,
        'Instruction: ' + ' '.join(instruction),
    ]
    results = map_context(context, 3277, '\n'.join(template))
    return ' '.join(results)


question = ''

if __name__ == '__main__':
    _ = get_name_to_symbols(settings.TOP_K)
    _ = get_name_to_transcript(_)
    _ = get_name_to_summary(_)
    _ = get_name_to_content(_, question)
    _ = reduce_content(_)
    _ = answer_question(_, question)
    print(_)
