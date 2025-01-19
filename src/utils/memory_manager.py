import sqlite3
import logging
import uuid
from enum import Enum
from pathlib import Path
from typing import List, Any, Dict, Optional, Tuple, Union
from io import BytesIO

from PIL import Image

from src.utils.file_storage import FileStorage
from src.utils.file_storage_backends import LocalStorageBackend
from src.utils.embedding_helper import EmbeddingHelper
from src.utils.types import FileType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryOutputType(str, Enum):
    """
    Represents the type of memory to filter on when doing similarity searches.
    TEXT:  Memories that have no file (i.e., text-only).
    IMAGE: Memories stored with FileType.IMAGE
    AUDIO: Memories stored with FileType.AUDIO
    """

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"

class MemoryManager:
    """
    A simple SQLite-based database class that stores:
      - Memories in a regular table (`memories`)
      - File metadata in a separate table (`files`)
      - Embeddings in a virtual sqlite-vec table (`memories_vec`).

    It requires `text` (text) for **all** memories (text, image, audio).

    You can:
      - Insert memories (text, image, audio).
      - Delete memories (and associated files).
      - Perform similarity (k-NN) searches using the sqlite-vec extension,
        with optional filtering by memory output type.
      - Use a single convenience method `select_similar` to automatically
        detect whether the query is text, an image (PIL.Image.Image), or audio (BytesIO).
    """

    def __init__(
        self,
        embedding_helper: EmbeddingHelper,
    ):
        """
        :param embedding_helper: An instance of your EmbeddingHelper class (for text/image/audio).
        """
        self.embedding_helper = embedding_helper
        self.vector_dim = embedding_helper.vector_dim

        # We'll place the DB in ~/.xeno/database.sqlite
        xeno_dir = Path.home() / ".xeno"
        xeno_dir.mkdir(parents=True, exist_ok=True)  # Ensure the folder exists

        self.db_file = xeno_dir / "memories.sqlite"
        self.memory_table_name = "memories"
        self.files_table_name = "memory_files"
        self.memory_vector_table_name = "memories_vec"

        self.conn = self._create_connection()
        self._create_tables()

        # Local file storage
        local_backend = LocalStorageBackend(base_directory=xeno_dir / "memory_files")
        self.local_storage = FileStorage(backend=local_backend)

    def _create_connection(self) -> sqlite3.Connection:
        """
        Creates and returns a SQLite connection with the sqlite-vec extension loaded.
        """
        try:
            conn = sqlite3.connect(self.db_file)
            conn.row_factory = sqlite3.Row
            conn.enable_load_extension(True)
            import sqlite_vec  # Must be installed separately

            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            logger.info(
                "SQLite connection established and sqlite-vec extension loaded."
            )
            return conn
        except sqlite3.Error as e:
            logger.error(f"SQLite connection error: {e}")
            raise

    def _create_tables(self):
        """
        Creates (if not exist):
          - `memories`
          - `files`
          - `memories_vec`

        Ensures `text` is NOT NULL for `memories`.
        """
        create_memories = f"""
        CREATE TABLE IF NOT EXISTS {self.memory_table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            text TEXT NOT NULL,
            file_id INTEGER
        );
        """
        create_files = f"""
        CREATE TABLE IF NOT EXISTS {self.files_table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL,
            ref TEXT NOT NULL
        );
        """
        create_memories_vec = f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {self.memory_vector_table_name}
        USING vec0(
            id INTEGER PRIMARY KEY,
            embedding FLOAT[{self.vector_dim}]
        );
        """
        try:
            with self.conn:
                self.conn.execute(create_memories)
                self.conn.execute(create_files)
                self.conn.execute(create_memories_vec)
            logger.info(
                f"Tables '{self.memory_table_name}', '{self.files_table_name}', "
                f"and '{self.memory_vector_table_name}' created or verified."
            )
        except sqlite3.Error as e:
            logger.error(f"Error creating tables: {e}")
            raise

    def close(self):
        """
        Closes the database connection.
        """
        if self.conn:
            self.conn.close()
            logger.info("SQLite connection closed.")

    #
    # Public Insert Methods
    #
    def insert_text(self, text: str) -> int:
        """
        Inserts a text memory:
          1) Embeds the text via self.embedding_helper
          2) Stores text in `memories`
          3) Stores embedding in `memories_vec`
        """
        # Embed the text
        try:
            embed_tensor = self.embedding_helper.create_text_embedding(text)
        except Exception as e:
            raise Exception(f"An error occurred while embedding the text: {str(e)}")

        # Some embedding helpers return shape [D], others [1, D].
        if len(embed_tensor.shape) == 2:
            embedding = embed_tensor[0].tolist()
        else:
            embedding = embed_tensor.tolist()

        # Insert into memories (file_id=None for text)
        obs_id = self._insert_memory(text=text, file_id=None)

        # Insert embedding
        self._insert_embedding(obs_id, embedding)

        logger.info(f"Inserted text memory id={obs_id}, text='{text[:30]}'.")
        return obs_id

    def insert_image(self, text: str, image: Image.Image) -> int:
        """
        Inserts an memory with an image (PIL Image). Also embeds the image internally.

        :param text: A textual description or other required text for the memory.
        :param image:   A PIL.Image.Image object.
        """
        # Embed the image
        try:
            embed_tensor = self.embedding_helper.create_image_embedding(image)
        except Exception as e:
            raise Exception(f"An error occurred while embedding the image: {str(e)}")

        # Some embedding helpers return shape [D], others [1, D].
        if len(embed_tensor.shape) == 2:
            embedding = embed_tensor[0].tolist()
        else:
            embedding = embed_tensor.tolist()

        # Convert the PIL image to bytes for storage.
        img_buffer = BytesIO()
        image.save(img_buffer, format="PNG")
        image_bytes = img_buffer.getvalue()

        # Store the file (type=IMAGE) -> get back file_id
        file_id, file_ref = self._insert_file(FileType.IMAGE, image_bytes)

        # Insert the memory
        obs_id = self._insert_memory(text=text, file_id=file_id)

        # Insert embedding
        self._insert_embedding(obs_id, embedding)

        logger.info(f"Inserted image memory id={obs_id}, file_ref={file_ref}.")
        return obs_id

    def insert_audio(self, text: str, audio: BytesIO) -> int:
        """
        Inserts an memory with an audio file (BytesIO). Also embeds the audio internally.

        :param text: Required text associated with this memory.
        :param audio:   A BytesIO object containing raw audio data (e.g., WAV).
        """
        # Embed the audio
        try:
            embed_tensor = self.embedding_helper.create_audio_embedding(audio)
        except Exception as e:
            raise Exception(f"An error occurred while embedding audio: {str(e)}")

        if len(embed_tensor.shape) == 2:
            embedding = embed_tensor[0].tolist()
        else:
            embedding = embed_tensor.tolist()

        # Convert the BytesIO into raw bytes for storage.
        audio_bytes = audio.getvalue()

        # Insert file (type=AUDIO)
        file_id, file_ref = self._insert_file(FileType.AUDIO, audio_bytes)

        # Insert memory row
        obs_id = self._insert_memory(text=text, file_id=file_id)

        # Insert embedding
        self._insert_embedding(obs_id, embedding)

        logger.info(f"Inserted audio memory id={obs_id}, file_ref={file_ref}.")
        return obs_id

    #
    # Internal Helpers
    #
    def _insert_file(self, file_type: FileType, data: bytes) -> Tuple[int, str]:
        """
        Saves file data locally and inserts into the `files` table.
        Returns: (file_id, file_ref)
        """
        extension_map = {
            FileType.IMAGE: "png",
            FileType.AUDIO: "wav",
        }
        extension = extension_map.get(file_type, "bin")
        file_ref = f"{uuid.uuid4()}.{extension}"

        # Save the file to local storage
        self.local_storage.save_file(file_ref, data)

        # Insert into files table
        insert_file_sql = f"""
        INSERT INTO {self.files_table_name} (type, ref) VALUES (?, ?)
        """
        try:
            with self.conn:
                cursor = self.conn.execute(insert_file_sql, (file_type.value, file_ref))
                file_id = cursor.lastrowid
        except sqlite3.Error as e:
            logger.error(f"Error inserting into files table: {e}")
            raise

        return file_id, file_ref

    def _insert_memory(self, text: str, file_id: Optional[int]) -> int:
        """
        Inserts into `memories` table. `text` must not be None.
        """
        insert_obs_sql = f"""
        INSERT INTO {self.memory_table_name} (text, file_id)
        VALUES (?, ?)
        """
        try:
            with self.conn:
                cursor = self.conn.execute(insert_obs_sql, (text, file_id))
                obs_id = cursor.lastrowid
        except sqlite3.Error as e:
            logger.error(f"Error inserting memory: {e}")
            raise
        return obs_id

    def _insert_embedding(self, obs_id: int, embedding: List[float]):
        """
        Inserts the embedding into memories_vec. Ensures dimension matches.
        """
        if len(embedding) != self.vector_dim:
            raise ValueError(
                f"Embedding length ({len(embedding)}) != vector_dim ({self.vector_dim})."
            )
        embedding_str = "[" + ", ".join(str(x) for x in embedding) + "]"
        insert_vec_sql = f"""
        INSERT INTO {self.memory_vector_table_name} (id, embedding)
        VALUES (?, ?)
        """
        try:
            with self.conn:
                self.conn.execute(insert_vec_sql, (obs_id, embedding_str))
        except sqlite3.Error as e:
            logger.error(f"Error inserting embedding for obs_id={obs_id}: {e}")
            raise

    #
    # Delete
    #
    def delete(self, obs_id: int):
        """
        Deletes the memory by ID (including embedding).
        Also deletes any associated file in the DB and from local storage.
        """
        # 1) Find if there's an associated file
        select_file_sql = f"""
        SELECT o.file_id, f.ref
        FROM {self.memory_table_name} o
        LEFT JOIN {self.files_table_name} f ON o.file_id = f.id
        WHERE o.id = ?
        """
        try:
            row = self.conn.execute(select_file_sql, (obs_id,)).fetchone()
            if not row:
                logger.info(
                    f"No memory found with id={obs_id}. Nothing to delete."
                )
                return
        except sqlite3.Error as e:
            logger.error(f"Error fetching memory to delete: {e}")
            raise

        file_id = row["file_id"]
        file_ref = row["ref"]

        # 2) Delete from memories_vec
        sql_vec_del = f"DELETE FROM {self.memory_vector_table_name} WHERE id = ?"
        # 3) Delete from memories
        sql_obs_del = f"DELETE FROM {self.memory_table_name} WHERE id = ?"

        try:
            with self.conn:
                self.conn.execute(sql_vec_del, (obs_id,))
                self.conn.execute(sql_obs_del, (obs_id,))
        except sqlite3.Error as e:
            logger.error(f"Error deleting memory (id={obs_id}): {e}")
            raise

        # 4) If there's a file, remove it from DB and local storage
        if file_id is not None:
            try:
                if file_ref:
                    self.local_storage.delete_file(file_ref)
                sql_file_del = f"DELETE FROM {self.files_table_name} WHERE id = ?"
                with self.conn:
                    self.conn.execute(sql_file_del, (file_id,))
                logger.info(
                    f"Deleted file id={file_id} ref={file_ref} from DB/storage."
                )
            except sqlite3.Error as e:
                logger.error(f"Error deleting file reference (id={file_id}): {e}")
                raise
            except Exception as e:
                logger.error(f"Error deleting physical file '{file_ref}': {e}")
                raise

        logger.info(f"Deleted memory with id={obs_id}.")

    #
    # Similarity Search
    #
    def _select_similar(
        self,
        query_vector: List[float],
        top_k: int = 4,
        output_types: Optional[List[MemoryOutputType]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Performs a similarity (k-NN) search using the `memories_vec` virtual table
        and returns up to `top_k` matches, joined with `memories` + `files`.

        :param query_vector: The query vector (length = self.vector_dim).
        :param top_k:        The number of top similar results to retrieve.
        :param output_types: Which output memory types to include in the results
                             (e.g. [MemoryOutputType.TEXT, MemoryOutputType.IMAGE]).
                             If None or empty, returns all.
        :return:             A list of dicts, each containing joined row + `distance`.
        """
        if len(query_vector) != self.vector_dim:
            raise ValueError(
                f"Query vector length ({len(query_vector)}) != vector_dim ({self.vector_dim})."
            )

        # Build query
        query_str = "[" + ", ".join(str(x) for x in query_vector) + "]"

        base_sql = f"""
        SELECT
            o.id AS memory_id,
            o.timestamp,
            o.text,
            f.id AS file_id,
            f.type AS file_type,
            f.ref AS file_ref,
            distance(ov.embedding, :query) AS distance
        FROM {self.memory_vector_table_name} ov
        JOIN {self.memory_table_name} o ON o.id = ov.id
        LEFT JOIN {self.files_table_name} f ON o.file_id = f.id
        WHERE ov.embedding MATCH :query
          AND k = :top_k
        """

        # Build filter for output_types
        filter_sql = self._build_output_type_filter(output_types)
        base_sql += filter_sql

        base_sql += """
        ORDER BY distance
        LIMIT :top_k
        """

        params: Dict[str, Any] = {"query": query_str, "top_k": top_k}

        try:
            cur = self.conn.cursor()
            cur.execute(base_sql, params)
            rows = cur.fetchall()
            results = []
            for r in rows:
                row_dict = dict(r)
                row_dict["distance"] = float(row_dict["distance"])
                results.append(row_dict)
            return results
        except sqlite3.Error as e:
            logger.error(f"Error during similarity search: {e}")
            raise

    def _build_output_type_filter(
        self, output_types: Optional[List[MemoryOutputType]]
    ) -> str:
        """
        Builds the SQL filter snippet to include only the specified output types.

        If output_types is None or empty, returns an empty string (i.e., no filter).
        Otherwise, it will generate something like:
            AND (
              (o.file_id IS NULL) OR
              (f.type = "image")    OR
              (f.type = "audio")
            )
        depending on the selection.
        """
        if not output_types:
            return ""  # No filtering

        conditions = []
        if MemoryOutputType.TEXT in output_types:
            # text means no file
            conditions.append("(o.file_id IS NULL)")
        if MemoryOutputType.IMAGE in output_types:
            conditions.append("(f.type = 'image')")
        if MemoryOutputType.AUDIO in output_types:
            conditions.append("(f.type = 'audio')")

        if not conditions:
            return ""

        return " AND ( " + " OR ".join(conditions) + " )"

    #
    # Single Convenience Similarity Method
    #
    def select_similar(
        self,
        data: Union[str, Image.Image, BytesIO],
        top_k: int = 4,
        output_types: Optional[List[MemoryOutputType]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convenience method to perform similarity search from one of:
          - A text string
          - A PIL image
          - A BytesIO object (assumed to be audio)

        Internally detects type, calls the appropriate embedding helper method,
        then calls _select_similar with the resulting vector.
        """
        # 1) Determine input type
        if isinstance(data, str):
            # Text
            try:
                embed_tensor = self.embedding_helper.create_text_embedding(data)
            except Exception as e:
                raise Exception(f"Error embedding query text: {e}")

        elif isinstance(data, Image.Image):
            # PIL image
            try:
                embed_tensor = self.embedding_helper.create_image_embedding(data)
            except Exception as e:
                raise Exception(f"Error embedding query image: {e}")

        elif isinstance(data, BytesIO):
            # Audio
            try:
                embed_tensor = self.embedding_helper.create_audio_embedding(data)
            except Exception as e:
                raise Exception(f"Error embedding query audio: {e}")

        else:
            raise ValueError(
                "Unsupported data type. Please pass str (text), PIL.Image.Image, or BytesIO (audio)."
            )

        # 2) Convert embedding tensor to a Python list
        if len(embed_tensor.shape) == 2:
            query_vector = embed_tensor[0].tolist()
        else:
            query_vector = embed_tensor.tolist()

        # 3) Perform similarity search
        return self._select_similar(query_vector, top_k=top_k, output_types=output_types)


#
# Example Usage
#
if __name__ == "__main__":
    import wave

    # 1) Create your embedding model
    embedding_helper = EmbeddingHelper()

    # 2) Initialize DB with that embedding model
    memory = MemoryManager(embedding_helper=embedding_helper)

    # -- Insert text
    obs_text_id = memory.insert_text("Hello from text!")
    print(f"Inserted text memory ID={obs_text_id}")

    # -- Insert an image (red 224x224 PNG in memory)
    red_img = Image.new("RGB", (224, 224), color=(255, 0, 0))
    obs_img_id = memory.insert_image(red_img, text="A red square image")
    print(f"Inserted image memory ID={obs_img_id}")

    # -- Insert an audio file (1 second of silence, 16-bit, 16kHz)
    silent_wav = BytesIO()
    with wave.open(silent_wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(16000)
        w.writeframes(b"\x00" * 16000 * 2)  # 1 second of silence

    obs_audio_id = memory.insert_audio(silent_wav, text="One second of silence")
    print(f"Inserted audio memory ID={obs_audio_id}")

    # -- Single convenience method for similarity (text example)
    results_text = memory.select_similar(
        "hello world",
        top_k=5,
        output_types=[MemoryOutputType.TEXT, MemoryOutputType.IMAGE],
    )
    print("\nSimilarity search (text query) results:")
    for r in results_text:
        print(
            f"ObsID={r['memory_id']}, Distance={r['distance']:.4f}, "
            f"FileType={r['file_type']}, Content={r['text']}"
        )

    # -- Single convenience method for similarity (image example)
    results_image = memory.select_similar(
        red_img,  # The same PIL image we inserted
        top_k=5,
        output_types=[MemoryOutputType.IMAGE, MemoryOutputType.AUDIO],
    )
    print("\nSimilarity search (image query) results:")
    for r in results_image:
        print(
            f"ObsID={r['memory_id']}, Distance={r['distance']:.4f}, "
            f"FileType={r['file_type']}, Content={r['text']}"
        )

    # -- Single convenience method for similarity (audio example)
    # Pass the same BytesIO object
    results_audio = memory.select_similar(
        silent_wav,
        top_k=5,
        output_types=[MemoryOutputType.TEXT, MemoryOutputType.AUDIO],
    )
    print("\nSimilarity search (audio query) results:")
    for r in results_audio:
        print(
            f"ObsID={r['memory_id']}, Distance={r['distance']:.4f}, "
            f"FileType={r['file_type']}, Content={r['text']}"
        )

    # Cleanup
    memory.close()
