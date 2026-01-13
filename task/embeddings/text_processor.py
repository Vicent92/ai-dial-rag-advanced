from enum import StrEnum
import os

import psycopg2
from psycopg2.extras import RealDictCursor

from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.utils.text import chunk_text


class SearchMode(StrEnum):
    EUCLIDIAN_DISTANCE = "euclidean"  # Euclidean distance (<->)
    COSINE_DISTANCE = "cosine"  # Cosine distance (<=>)


class TextProcessor:
    """Processor for text documents that handles chunking, embedding, storing, and retrieval"""

    def __init__(self, embeddings_client: DialEmbeddingsClient, db_config: dict):
        self.embeddings_client = embeddings_client
        self.db_config = db_config

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password']
        )

    def _truncate_table(self):
        """Truncate the vectors table"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("TRUNCATE TABLE vectors RESTART IDENTITY")
            conn.commit()
        finally:
            conn.close()

    def _save_chunk(self, document_name: str, text: str, embedding: list[float]):
        """Save a text chunk with its embedding to the database"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                embedding_str = str(embedding)
                cursor.execute(
                    """
                    INSERT INTO vectors (document_name, text, embedding)
                    VALUES (%s, %s, %s::vector)
                    """,
                    (document_name, text, embedding_str)
                )
            conn.commit()
        finally:
            conn.close()

    def process_text_file(
        self,
        file_path: str,
        chunk_size: int = 300,
        overlap: int = 40,
        dimensions: int = 1536,
        truncate: bool = True
    ):
        """
        Process a text file: load, chunk, embed, and store in database.

        Args:
            file_path: Path to the text file
            chunk_size: Size of text chunks
            overlap: Character overlap between chunks
            dimensions: Embedding dimensions
            truncate: Whether to truncate the table before inserting
        """
        if truncate:
            self._truncate_table()

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        chunks = chunk_text(content, chunk_size, overlap)
        document_name = os.path.basename(file_path)

        embeddings = self.embeddings_client.get_embeddings(chunks, dimensions)

        for index, chunk in enumerate(chunks):
            embedding = embeddings.get(index)
            if embedding:
                self._save_chunk(document_name, chunk, embedding)

        print(f"Processed {len(chunks)} chunks from {document_name}")

    def search(
        self,
        query: str,
        mode: SearchMode = SearchMode.COSINE_DISTANCE,
        top_k: int = 5,
        min_score: float = 0.5,
        dimensions: int = 1536
    ) -> list[str]:
        """
        Search for relevant text chunks using vector similarity.

        Args:
            query: User query text
            mode: Search mode (cosine or euclidean distance)
            top_k: Number of results to return
            min_score: Minimum similarity threshold
            dimensions: Embedding dimensions

        Returns:
            List of relevant text chunks
        """
        embeddings = self.embeddings_client.get_embeddings([query], dimensions)
        query_embedding = embeddings.get(0)

        if not query_embedding:
            return []

        embedding_str = str(query_embedding)

        if mode == SearchMode.COSINE_DISTANCE:
            distance_operator = "<=>"
        else:
            distance_operator = "<->"

        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                query_sql = f"""
                    SELECT text, embedding {distance_operator} %s::vector AS distance
                    FROM vectors
                    WHERE embedding {distance_operator} %s::vector < %s
                    ORDER BY distance
                    LIMIT %s
                """
                cursor.execute(query_sql, (embedding_str, embedding_str, min_score, top_k))
                results = cursor.fetchall()

            return [row['text'] for row in results]
        finally:
            conn.close()

