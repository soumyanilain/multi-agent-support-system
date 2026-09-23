import json
import os
import re
from pathlib import Path

import faiss
from dotenv import load_dotenv
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, ValidationError
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)

TOP_K_PER_QUERY = 3
FINAL_CONTEXT_SIZE = 5
NUM_ALTERNATIVE_QUERIES = 3

# Reciprocal Rank Fusion smoothing constant. 60 is the value used in
# the original RRF paper and the common default.
RRF_K = 60


# ============================================================
# RESPONSE SCHEMA
# ============================================================

class SpecialistResponse(BaseModel):
    category: str
    resolution: str
    sources: list[str]


# ============================================================
# SPECIALIST RAG PIPELINE
# ============================================================

class RAGPipeline:

    def __init__(self):

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY was not found. "
                "Please add it to the .env file."
            )

        self.client = Groq(
            api_key=api_key
        )

        print("Loading embedding model...")

        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

        self.documents = self.load_documents()

        self.chunks = self.chunk_documents()

        self.embeddings = self.create_embeddings()

        self.index = self.create_faiss_index()

        self.bm25 = self.create_bm25_index()

        # Cache query expansions so the evaluation script can compare
        # dense-only against fusion using identical query sets, and so
        # repeated questions do not re-hit the LLM rate limit.
        self._query_cache = {}

        print("RAG Pipeline ready.")


    # ========================================================
    # LOAD KNOWLEDGE BASE
    # ========================================================

    def load_documents(self):

        documents = []

        if not KNOWLEDGE_BASE_DIR.exists():
            raise FileNotFoundError(
                f"Knowledge base folder not found: "
                f"{KNOWLEDGE_BASE_DIR}"
            )

        for file_path in sorted(
            KNOWLEDGE_BASE_DIR.glob("*.md")
        ):

            text = file_path.read_text(
                encoding="utf-8"
            )

            if text.strip():

                documents.append(
                    {
                        "text": text,
                        "source": file_path.name
                    }
                )

        if not documents:
            raise FileNotFoundError(
                "No Markdown files were found "
                "in the knowledge_base folder."
            )

        print(
            f"Loaded {len(documents)} knowledge-base documents."
        )

        return documents


    # ========================================================
    # CHUNK DOCUMENTS
    # ========================================================

    def chunk_documents(self):

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = []

        for document in self.documents:

            document_chunks = splitter.split_text(
                document["text"]
            )

            for text in document_chunks:

                chunks.append(
                    {
                        "text": text,
                        "source": document["source"]
                    }
                )

        if not chunks:
            raise ValueError(
                "No chunks were created from "
                "the knowledge base."
            )

        print(
            f"Created {len(chunks)} knowledge-base chunks."
        )

        return chunks


    # ========================================================
    # CREATE EMBEDDINGS
    #
    # Embeddings are L2-normalized so that the FAISS inner
    # product index returns cosine similarity. This gives a
    # bounded 0-1 relevance score that the Specialist Agent
    # can threshold to detect out-of-scope questions.
    # ========================================================

    def create_embeddings(self):

        texts = [
            chunk["text"]
            for chunk in self.chunks
        ]

        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True
        )

        embeddings = embeddings.astype(
            "float32"
        )

        faiss.normalize_L2(
            embeddings
        )

        print(
            "Created embeddings with "
            f"dimension {embeddings.shape[1]}."
        )

        return embeddings


    # ========================================================
    # CREATE FAISS INDEX
    # ========================================================

    def create_faiss_index(self):

        dimension = self.embeddings.shape[1]

        # Inner product on normalized vectors == cosine similarity.
        index = faiss.IndexFlatIP(
            dimension
        )

        index.add(
            self.embeddings
        )

        print(
            f"FAISS index contains "
            f"{index.ntotal} vectors."
        )

        return index


    # ========================================================
    # CREATE BM25 KEYWORD INDEX
    # ========================================================

    def tokenize(self, text):
        """Lowercase word tokenizer used by the BM25 index."""

        return re.findall(
            r"[a-z0-9]+",
            text.lower()
        )


    def create_bm25_index(self):
        """Build a BM25 keyword index over the same chunks."""

        tokenized_chunks = [
            self.tokenize(chunk["text"])
            for chunk in self.chunks
        ]

        bm25 = BM25Okapi(
            tokenized_chunks
        )

        print(
            f"BM25 index contains "
            f"{len(tokenized_chunks)} chunks."
        )

        return bm25


    # ========================================================
    # QUERY EXPANSION
    #
    # Results are cached per question. This keeps the
    # dense-only and fusion evaluation runs comparable, and
    # avoids re-spending tokens against the LLM rate limit.
    # ========================================================

    def generate_query_variations(
        self,
        question
    ):

        if question in self._query_cache:
            return self._query_cache[question]

        prompt = f"""
You are helping a technical support system
retrieve information from a knowledge base.

Generate exactly {NUM_ALTERNATIVE_QUERIES}
alternative search queries for the user's question.

The alternative queries should:

- Preserve the original meaning.
- Use different wording.
- Include useful technical terms.
- Help retrieve relevant knowledge-base passages.
- Not invent facts.

Return ONLY valid JSON in this format:

{{
    "queries": [
        "alternative query 1",
        "alternative query 2",
        "alternative query 3"
    ]
}}

USER QUESTION:
{question}
"""

        try:

            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You generate search query "
                            "variations for document retrieval."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                response_format={
                    "type": "json_object"
                }
            )

            content = (
                response.choices[0]
                .message.content
            )

            data = json.loads(content)

            queries = data.get(
                "queries",
                []
            )

            cleaned_queries = []

            for query in queries:

                if isinstance(query, str):

                    query = query.strip()

                    if (
                        query
                        and query.lower()
                        != question.lower()
                    ):
                        cleaned_queries.append(
                            query
                        )

            variations = cleaned_queries[
                :NUM_ALTERNATIVE_QUERIES
            ]

            self._query_cache[question] = variations

            return variations

        except Exception as error:

            print(
                "Warning: Could not generate "
                f"query variations: {error}"
            )

            # Deliberately not cached. A rate limit or network error
            # is transient, so a later call for the same question
            # should be allowed to try again. Retrieval still works
            # using the original question alone.
            return []


    # ========================================================
    # INDIVIDUAL RETRIEVERS
    # ========================================================

    def dense_ranking(self, query, top_k):
        """Return chunk indices ranked by cosine similarity."""

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        )

        query_embedding = query_embedding.astype(
            "float32"
        )

        faiss.normalize_L2(
            query_embedding
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        ranked = []

        for score, index_number in zip(
            scores[0],
            indices[0]
        ):

            if index_number == -1:
                continue

            ranked.append(
                (
                    int(index_number),
                    float(score)
                )
            )

        return ranked


    def bm25_ranking(self, query, top_k):
        """Return chunk indices ranked by BM25 keyword score."""

        scores = self.bm25.get_scores(
            self.tokenize(query)
        )

        ordered = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        ranked = []

        for index_number in ordered[:top_k]:

            if scores[index_number] <= 0:
                continue

            ranked.append(
                (
                    int(index_number),
                    float(scores[index_number])
                )
            )

        return ranked


    # ========================================================
    # ADVANCED RAG:
    # HYBRID FUSION RETRIEVAL (BM25 + DENSE, MERGED WITH RRF)
    #
    # Dense embeddings capture meaning but blur exact terms.
    # Support tickets are full of literal tokens -- "Wi-Fi",
    # "keyboard", "synchronizing" -- that BM25 matches exactly.
    # Each retriever produces its own ranking and Reciprocal
    # Rank Fusion merges them by rank position, so neither
    # retriever's score scale has to be normalized against the
    # other.
    # ========================================================

    def retrieve_multi_query(
        self,
        question,
        top_k_per_query=TOP_K_PER_QUERY,
        final_k=FINAL_CONTEXT_SIZE,
        use_fusion=True
    ):
        """
        Multi-query expansion feeding a hybrid fusion retriever.

        The question is rewritten into several alternative phrasings.
        Each phrasing is run through both a dense retriever and a
        BM25 retriever, and every resulting ranking is merged with
        Reciprocal Rank Fusion:

            rrf_score(chunk) = sum over rankings of 1 / (K + rank)

        Setting use_fusion=False falls back to dense-only retrieval,
        which is how the evaluation script produces a like-for-like
        comparison between the two strategies.
        """

        alternative_queries = self.generate_query_variations(
            question
        )

        # Always include the original question.
        queries = [question] + alternative_queries

        # chunk index -> accumulated RRF score
        fused_scores = {}

        # chunk index -> best cosine similarity seen, used as the
        # confidence signal reported to the Specialist Agent.
        cosine_scores = {}

        # chunk index -> the query phrasing that retrieved it
        matched_queries = {}

        rankings = []

        for query in queries:

            dense = self.dense_ranking(
                query,
                top_k_per_query
            )

            rankings.append(
                (query, dense)
            )

            for index_number, score in dense:

                if (
                    index_number not in cosine_scores
                    or score > cosine_scores[index_number]
                ):
                    cosine_scores[index_number] = score

            if use_fusion:

                keyword = self.bm25_ranking(
                    query,
                    top_k_per_query
                )

                rankings.append(
                    (query, keyword)
                )

        # Reciprocal Rank Fusion across every ranking produced.
        for query, ranking in rankings:

            for rank, (index_number, _score) in enumerate(
                ranking,
                start=1
            ):

                fused_scores[index_number] = (
                    fused_scores.get(index_number, 0.0)
                    + 1.0 / (RRF_K + rank)
                )

                matched_queries.setdefault(
                    index_number,
                    query
                )

        ranked_indices = sorted(
            fused_scores,
            key=lambda i: fused_scores[i],
            reverse=True
        )

        results = []

        for index_number in ranked_indices[:final_k]:

            results.append(
                {
                    "text": self.chunks[index_number]["text"],

                    "source": self.chunks[index_number]["source"],

                    # Cosine similarity is kept separately from the
                    # fusion score because the Specialist Agent
                    # thresholds it to detect out-of-scope questions.
                    # BM25-only hits have no cosine score, so they
                    # contribute nothing to confidence.
                    "score": cosine_scores.get(index_number, 0.0),

                    "rrf_score": round(
                        fused_scores[index_number],
                        5
                    ),

                    "matched_query": matched_queries.get(
                        index_number,
                        question
                    )
                }
            )

        # generate_answer reads results[0]["score"] as the confidence,
        # so make sure the highest cosine similarity is first.
        results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return results


    # ========================================================
    # GENERATE GROUNDED ANSWER
    # ========================================================

    def generate_answer(
        self,
        question,
        retrieved_results
    ):

        if not retrieved_results:

            return {
                "category": "Unknown",
                "resolution": (
                    "No relevant information "
                    "was found in the knowledge base."
                ),
                "sources": [],
                "confidence": 0.0
            }

        context_parts = []

        for index, result in enumerate(
            retrieved_results,
            start=1
        ):

            context_parts.append(
                f"""
SOURCE {index}: {result["source"]}

{result["text"]}
"""
            )

        context = "\n".join(
            context_parts
        )

        prompt = f"""
You are a technical support Specialist Agent.

Answer the user's support request using ONLY
the information provided in the knowledge-base
context below.

Do not use outside knowledge.

The response must contain exactly these fields:

1. category
2. resolution
3. sources

Possible categories include:

- Account Access
- Email
- Hardware
- Network
- Security
- Software

The sources must contain only knowledge-base
filenames that support the answer.

Return ONLY valid JSON in exactly this structure:

{{
    "category": "Account Access",
    "resolution": "Your grounded resolution here.",
    "sources": [
        "account_access.md"
    ]
}}

USER QUESTION:
{question}

KNOWLEDGE BASE CONTEXT:
{context}
"""

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a technical support "
                        "agent that answers only from "
                        "provided knowledge-base context."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            response_format={
                "type": "json_object"
            }
        )

        content = (
            response.choices[0]
            .message.content
        )

        try:

            data = json.loads(
                content
            )

        except json.JSONDecodeError as error:

            raise RuntimeError(
                "Groq returned invalid JSON."
            ) from error

        try:

            validated_response = (
                SpecialistResponse(
                    **data
                )
            )

        except ValidationError as error:

            raise RuntimeError(
                "Groq response did not match "
                "the required response schema."
            ) from error

        # Only allow sources that actually
        # appeared in retrieved context.
        retrieved_sources = {
            result["source"]
            for result in retrieved_results
        }

        valid_sources = [
            source
            for source in validated_response.sources
            if source in retrieved_sources
        ]

        # If the model did not provide valid
        # sources, use the retrieved sources.
        if not valid_sources:

            valid_sources = sorted(
                retrieved_sources
            )

        return {
            "category": (
                validated_response.category
            ),
            "resolution": (
                validated_response.resolution
            ),
            "sources": valid_sources,

            # Best cosine similarity across all retrieved
            # chunks. The Specialist Agent thresholds this
            # to detect questions the knowledge base does
            # not cover.
            "confidence": round(
                retrieved_results[0]["score"],
                3
            )
        }


    # ========================================================
    # PUBLIC ANSWER FUNCTION
    # ========================================================

    def answer(
        self,
        question,
        use_fusion=True
    ):

        if not isinstance(
            question,
            str
        ):

            raise TypeError(
                "Question must be a string."
            )

        question = question.strip()

        if not question:

            raise ValueError(
                "Question cannot be empty."
            )

        retrieved_results = self.retrieve_multi_query(
            question,
            use_fusion=use_fusion
        )

        return self.generate_answer(
            question,
            retrieved_results
        )


# ============================================================
# LAZY PIPELINE INSTANCE
#
# The pipeline is built on first use rather than at import
# time, so importing this module does not block on loading
# the embedding model or fail when no API key is present.
# ============================================================

_pipeline = None


def get_pipeline():

    global _pipeline

    if _pipeline is None:
        _pipeline = RAGPipeline()

    return _pipeline


# ============================================================
# FUNCTION USED BY SERVER.PY
# ============================================================

def answer(question, use_fusion=True):
    """
    Public function used by the Specialist
    A2A server.

    Args:
        question:   User's support question.
        use_fusion: False falls back to dense-only retrieval,
                    used by the evaluation script.

    Returns:
        Dictionary containing:
        category
        resolution
        sources
        confidence
    """

    return get_pipeline().answer(
        question,
        use_fusion=use_fusion
    )