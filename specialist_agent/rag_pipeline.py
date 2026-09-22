import json
import os
from pathlib import Path

import faiss
from dotenv import load_dotenv
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, ValidationError
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

        index = faiss.IndexFlatL2(
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
    # ADVANCED RAG:
    # MULTI-QUERY RETRIEVAL
    # ========================================================

    def generate_query_variations(
        self,
        question
    ):

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

            return cleaned_queries[
                :NUM_ALTERNATIVE_QUERIES
            ]

        except Exception as error:

            print(
                "Warning: Could not generate "
                f"query variations: {error}"
            )

            return []


    # ========================================================
    # MULTI-QUERY FAISS RETRIEVAL
    # ========================================================

    def retrieve_multi_query(
        self,
        question,
        top_k_per_query=TOP_K_PER_QUERY,
        final_k=FINAL_CONTEXT_SIZE
    ):

        alternative_queries = (
            self.generate_query_variations(
                question
            )
        )

        # Always include the original question.
        queries = [
            question
        ] + alternative_queries

        candidate_chunks = {}

        for query in queries:

            query_embedding = (
                self.embedding_model.encode(
                    [query],
                    convert_to_numpy=True
                )
            )

            query_embedding = (
                query_embedding.astype(
                    "float32"
                )
            )

            distances, indices = (
                self.index.search(
                    query_embedding,
                    top_k_per_query
                )
            )

            for distance, index_number in zip(
                distances[0],
                indices[0]
            ):

                if index_number == -1:
                    continue

                index_number = int(
                    index_number
                )

                distance = float(
                    distance
                )

                # If the same chunk is retrieved
                # by multiple queries, keep the
                # best distance.
                if (
                    index_number
                    not in candidate_chunks
                    or distance
                    < candidate_chunks[
                        index_number
                    ]["distance"]
                ):

                    candidate_chunks[
                        index_number
                    ] = {
                        "text": self.chunks[
                            index_number
                        ]["text"],

                        "source": self.chunks[
                            index_number
                        ]["source"],

                        "distance": distance,

                        "matched_query": query
                    }

        ranked_results = sorted(
            candidate_chunks.values(),
            key=lambda item: item["distance"]
        )

        return ranked_results[:final_k]


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
                "sources": []
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
            "sources": valid_sources
        }


    # ========================================================
    # PUBLIC ANSWER FUNCTION
    # ========================================================

    def answer(
        self,
        question
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

        retrieved_results = (
            self.retrieve_multi_query(
                question
            )
        )

        return self.generate_answer(
            question,
            retrieved_results
        )


# ============================================================
# SINGLE PIPELINE INSTANCE
# ============================================================

rag_pipeline = RAGPipeline()


# ============================================================
# FUNCTION USED BY SERVER.PY
# ============================================================

def answer(question):
    """
    Public function used by the Specialist
    A2A server.

    Args:
        question: User's support question.

    Returns:
        Dictionary containing:
        category
        resolution
        sources
    """

    return rag_pipeline.answer(
        question
    )
