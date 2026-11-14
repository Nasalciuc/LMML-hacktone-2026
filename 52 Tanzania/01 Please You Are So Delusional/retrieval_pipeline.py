"""Retrieval pipeline for the Tanzania agents task (repaired)."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flag_utils import derive_flag, load_verification_codes

# BM25 parameters; the title is scored as its own field and weighted up,
# because it is the only place that says which notice is current.
K1 = 1.2
B = 0.75
TITLE_WEIGHT = 3.0

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "at", "by", "can", "do", "does", "for", "from",
    "give", "in", "is", "it", "of", "on", "or", "should", "the", "they",
    "to", "under", "what", "when", "where", "which", "who", "with",
}


@dataclass(frozen=True)
class Document:
    """One searchable source document."""

    document_id: str
    title: str
    text: str


def load_corpus(corpus_path: Path) -> list[Document]:
    """Read source documents from a JSON Lines corpus.

    :param corpus_path: Corpus file supplied with the task.
    :return: Searchable documents in file order.
    """
    documents: list[Document] = []
    with corpus_path.open(encoding="utf-8") as corpus_file:
        for line in corpus_file:
            row = json.loads(line)
            documents.append(Document(row["document_id"], row["title"], row["text"]))
    return documents


def load_questions(questions_path: Path) -> list[dict[str, str]]:
    """Read the ordered retrieval questions.

    :param questions_path: JSON question file supplied with the task.
    :return: Question objects in official order.
    """
    with questions_path.open(encoding="utf-8") as questions_file:
        questions = json.load(questions_file)
    if not isinstance(questions, list):
        raise ValueError("questions.json must contain a list.")
    return questions


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_PATTERN.findall(text.lower()) if t not in STOPWORDS]


class BM25Field:
    """BM25 index over one text field of every document."""

    def __init__(self, texts: list[str]) -> None:
        self.term_counts = [Counter(tokenize(text)) for text in texts]
        self.lengths = np.array([sum(c.values()) for c in self.term_counts], dtype=float)
        self.avg_length = float(self.lengths.mean())
        n_docs = len(texts)
        doc_freq = Counter(term for counts in self.term_counts for term in counts)
        self.idf = {
            term: math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }

    def scores(self, query: str) -> np.ndarray:
        norm = K1 * (1 - B + B * self.lengths / self.avg_length)
        total = np.zeros(len(self.term_counts))
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = np.array([counts.get(term, 0) for counts in self.term_counts], dtype=float)
            total += idf * tf * (K1 + 1) / (tf + norm)
        return total


def retrieve_documents(
    documents: list[Document],
    questions: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Return the best whole source document for every question.

    :param documents: Complete source documents.
    :param questions: Ordered task questions.
    :return: Top source document ID for each question.
    """
    titles = BM25Field([document.title for document in documents])
    bodies = BM25Field([document.text for document in documents])

    results: list[dict[str, str]] = []
    for question in questions:
        query = question["question"]
        scores = TITLE_WEIGHT * titles.scores(query) + bodies.scores(query)
        best = int(np.argmax(scores))
        results.append(
            {
                "question_id": question["question_id"],
                "document_id": documents[best].document_id,
            }
        )
    return results


def parse_arguments() -> argparse.Namespace:
    """Parse command-line file locations.

    :return: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("corpus.jsonl"))
    parser.add_argument("--questions", type=Path, default=Path("questions.json"))
    parser.add_argument("--output", type=Path, default=Path("retrieved_ids.json"))
    return parser.parse_args()


def main() -> None:
    """Run retrieval, save IDs, and print the derived flag."""
    arguments = parse_arguments()
    documents = load_corpus(arguments.corpus)
    questions = load_questions(arguments.questions)
    results = retrieve_documents(documents, questions)
    arguments.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    flag = derive_flag(results, questions, load_verification_codes(arguments.corpus))
    print(f"Saved {len(results)} results to {arguments.output}.")
    print(flag)


if __name__ == "__main__":
    main()
