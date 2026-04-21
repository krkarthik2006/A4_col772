from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset
from utils import format_mmlu


class MMLUPro:
    LOCAL_DATASET_SOURCE = 'data/dataset.jsonl'
    LOCAL_DATASET_PATH = Path(__file__).resolve().parent / 'dataset.jsonl'
    LANGUAGE_ALIASES = {
        'en': {'en', 'eng', 'english'},
        'hindi': {'hindi', 'hi'},
        'bengali': {'bengali', 'bn'},
        'kannada': {'kannada', 'kn'},
        'tamil': {'tamil', 'ta'},
    }

    def __init__(self, language='en', split='validation', dataset_name=None, logger=None):
        self.language = language
        self.split = split
        self.dataset_name = dataset_name or self.LOCAL_DATASET_SOURCE
        self._dataset = None
        self._logger = logger

    def get_source_dataset_name(self):
        return self.dataset_name

    @classmethod
    def _canonical_language(cls, language):
        tag = str(language).strip().lower()
        for canonical, aliases in cls.LANGUAGE_ALIASES.items():
            if tag in aliases:
                return canonical
        return tag

    @classmethod
    def _language_matches(cls, row_language, requested_language):
        row_tag = cls._canonical_language(row_language)
        requested_tag = cls._canonical_language(requested_language)
        if requested_tag == 'en':
            return row_tag == 'en'
        return row_tag == requested_tag

    @classmethod
    def _read_local_jsonl(cls, dataset_path):
        rows = []
        with dataset_path.open('r', encoding='utf-8') as fp:
            for line in fp:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _load_with_optional_language_config(self):
        dataset_path = self.LOCAL_DATASET_PATH
        if not dataset_path.exists():
            raise FileNotFoundError(
                f'Local dataset not found at {dataset_path}. '
                'Generate it first using data/create_dataset.py'
            )

        rows = self._read_local_jsonl(dataset_path)
        filtered_rows = [
            row for row in rows
            if self._language_matches(row.get('language', ''), self.language)
        ]

        if self._logger:
            self._logger.info(
                'Loaded %d rows from %s for language=%s (split=%s ignored for local JSONL)',
                len(filtered_rows),
                dataset_path,
                self.language,
                self.split,
            )

        return Dataset.from_list(filtered_rows)

    def get_dataset(self, refresh=False):
        if refresh or self._dataset is None:
            self._dataset = self._load_with_optional_language_config()
        return self._dataset

    def get_prompt(self, question, choices):
        return format_mmlu(question, choices)

    @staticmethod
    def _normalize_options(options):
        if isinstance(options, list):
            return options
        return list(options)

    @staticmethod
    def _answer_to_letter(answer):
        if isinstance(answer, int):
            return 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[answer]

        answer_text = str(answer).strip().upper()
        if answer_text.isdigit():
            return 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[int(answer_text)]
        for letter in 'ABCD':
            if letter in answer_text:
                return letter
        return answer_text[:1]

    @classmethod
    def get_answer_idx(cls, row):
        answer_idx = row.get('answer_idx')
        if answer_idx is not None:
            return int(answer_idx)

        answer_letter = cls._answer_to_letter(row.get('answer', ''))
        if answer_letter:
            return max(0, ord(answer_letter) - ord('A'))
        return 0

    @classmethod
    def get_answer_letter(cls, row):
        if row.get('answer') is not None:
            return cls._answer_to_letter(row['answer'])

        answer_idx = row.get('answer_idx')
        if answer_idx is None:
            return ''
        return 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[int(answer_idx)]

    def get_prompt_from_row(self, row):
        question = row['question']
        options = self._normalize_options(row['options'])
        return self.get_prompt(question, options)

    def to_unified_row(self, row):
        raw_options = row.get('options', row.get('choices', []))
        options = self._normalize_options(raw_options)
        answer_letter = self.get_answer_letter(row)
        answer_idx = self.get_answer_idx(row)

        return {
            'question': row['question'],
            'options': options,
            'answer': answer_letter,
            'answer_idx': answer_idx,
            'subject': row.get('subject', row.get('src')),
            'language': self._canonical_language(row.get('language', self.language)),
            'source_dataset': row.get('source', self.get_source_dataset_name()),
        }

    def row_to_messages(self, row):
        unified_row = self.to_unified_row(row)
        prompt = self.get_prompt(
            unified_row['question'], unified_row['options'])
        answer_letter = unified_row['answer']

        return {
            'messages': [
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': f'#### ANSWER: {answer_letter}'},
            ],
            'question': unified_row['question'],
            'options': unified_row['options'],
            'answer': answer_letter,
            'answer_idx': unified_row['answer_idx'],
            'subject': unified_row['subject'],
            'language': unified_row['language'],
            'source_dataset': unified_row['source_dataset'],
        }

    def get_unified_dataset(self, refresh=False):
        dataset = self.get_dataset(refresh=refresh)
        return dataset.map(
            self.to_unified_row,
            remove_columns=dataset.column_names,
        )

    def get_messages_dataset(self, refresh=False):
        dataset = self.get_dataset(refresh=refresh)
        return dataset.map(
            self.row_to_messages,
            remove_columns=dataset.column_names,
        )

    def load_mmmlu(self, messages=False, refresh=False):
        if messages:
            return self.get_messages_dataset(refresh=refresh)
        return self.get_dataset(refresh=refresh)

    def load_mmlu_pro(self, messages=False, unified=True, refresh=False):
        if messages is True:
            return self.get_messages_dataset(refresh=refresh)
        if unified:
            return self.get_unified_dataset(refresh=refresh)
        return self.get_dataset(refresh=refresh)
