import sqlite3
import logging
import uuid
from enum import Enum
from pathlib import Path
from typing import List, Any, Dict, Optional, Tuple

from src.utils.file_storage import FileStorage
from src.utils.file_storage_backends import LocalStorageBackend
from src.utils.embedding_helper import EmbeddingHelper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileType(str, Enum):
    IMAGE = "image" # PNG
    AUDIO = "audio" # WAV


class EpisodicMemory:
    """
    A simple SQLite-based database class that stores:
      - Observations in a regular table (`observations`)
      - File metadata in a separate table (`files`)
      - Embeddings in a virtual sqlite-vec table (`observations_vec`).

    Allows for:
      - Inserting observations (text, image bytes, audio bytes).
      - Deleting observations (and associated files).
      - Similarity (k-NN) searches using the sqlite-vec extension.
    """

    def __init__(
        self,
        embedding_helper: EmbeddingHelper,
        vector_dim: int = 4,
    ):
        """
        :param embedding_helper: An instance of your EmbeddingHelper class (for text/image/audio).
        :param vector_dim:      Dimensionality of the embeddings (must match the embedding dim).
        """
        self.embedding_helper = embedding_helper
        self.vector_dim = vector_dim

        # We'll place the DB in ~/.xeno/database.sqlite
        xeno_dir = Path.home() / ".xeno"
        xeno_dir.mkdir(parents=True, exist_ok=True)  # Ensure the folder exists

        self.db_file = xeno_dir / "database.sqlite"
        self.observation_table_name = "observations"
        self.files_table_name = "files"
        self.observation_vector_table_name = "observations_vec"

        self.conn = self._create_connection()
        self._create_tables()

        # Local file storage
        local_backend = LocalStorageBackend(base_directory=xeno_dir / "files")
        self.local_storage = FileStorage(backend=local_backend)

    def _create_connection(self) -> sqlite3.Connection:
        """
        Creates and returns a SQLite connection with the sqlite-vec extension loaded.
        """
        try:
            conn = sqlite3.connect(self.db_file)
            conn.row_factory = sqlite3.Row
            conn.enable_load_extension(True)
            import sqlite_vec  # Must be installed
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            logger.info("SQLite connection established and sqlite-vec extension loaded.")
            return conn
        except sqlite3.Error as e:
            logger.error(f"SQLite connection error: {e}")
            raise

    def _create_tables(self):
        """
        Creates (if not exist):
          - `observations`
          - `files`
          - `observations_vec`
        with `content` as nullable now (so file-based observations may not have text).
        """
        create_observations = f"""
        CREATE TABLE IF NOT EXISTS {self.observation_table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            content TEXT,               -- now nullable
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
        create_observations_vec = f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {self.observation_vector_table_name}
        USING vec0(
            id INTEGER PRIMARY KEY,
            embedding FLOAT[{self.vector_dim}]
        );
        """
        try:
            with self.conn:
                self.conn.execute(create_observations)
                self.conn.execute(create_files)
                self.conn.execute(create_observations_vec)
            logger.info(
                f"Tables '{self.observation_table_name}', '{self.files_table_name}', "
                f"and '{self.observation_vector_table_name}' created or verified."
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
        Inserts a text observation:
          1) Embeds the text via self.embedding_helper
          2) Stores content in `observations`
          3) Stores embedding in `observations_vec`
        """
        # Embed the text
        try:
            embed_tensor = self.embedding_helper.create_text_embedding(text)  # shape [1, D]
        except Exception as e:
            raise Exception(f"An error occured while embedding the text: {str(e)}")

        embedding = embed_tensor[0].tolist()  # convert to python list

        # Insert into observations (no file_id)
        obs_id = self._insert_observation(content=text, file_id=None)

        # Insert embedding
        self._insert_embedding(obs_id, embedding)

        logger.info(f"Inserted text observation id={obs_id}, content='{text[:30]}'.")
        return obs_id

    def insert_image(self, image_bytes: bytes) -> int:
        """
        Inserts an observation with an image file. Also embeds the image internally.
        """
        # Embed the image
        try:
            embed_tensor = self.embedding_helper.create_image_embedding(image_bytes)
        except Exception as e:
            raise Exception(f"An error occured while embedding the image: {str(e)}")

        embedding = embed_tensor.tolist()

        # We store the file (type=IMAGE) -> get back file_id
        file_id, file_ref = self._insert_file(FileType.IMAGE, image_bytes)

        # Insert the observation (content=None) since it's image-based
        obs_id = self._insert_observation(content=None, file_id=file_id)

        # Insert embedding
        self._insert_embedding(obs_id, embedding)

        logger.info(f"Inserted image observation id={obs_id}, file_ref={file_ref}.")
        return obs_id

    def insert_audio(self, audio_bytes: bytes) -> int:
        """
        Inserts an observation with an audio file. Also embeds the audio internally.
        """
        # Embed the audio
        try:
            embed_tensor = self.embedding_helper.create_audio_embedding(audio_bytes)
        except Exception as e:
            raise Exception(f"An error occured while embedding text: {str(e)}")
        
        embedding = embed_tensor.tolist()

        # Insert file (type=AUDIO)
        file_id, file_ref = self._insert_file(FileType.AUDIO, audio_bytes)

        # Insert observation row
        obs_id = self._insert_observation(content=None, file_id=file_id)

        # Insert embedding
        self._insert_embedding(obs_id, embedding)

        logger.info(f"Inserted audio observation id={obs_id}, file_ref={file_ref}.")
        return obs_id

    #
    # Internal Helpers
    #
    def _insert_file(self, file_type: FileType, data: bytes) -> Tuple[int, str]:
        """
        Saves file data locally and inserts into the `files` table.
        Returns: (file_id, file_ref)
        """
        # Generate a unique file_ref
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

    def _insert_observation(self, content: Optional[str], file_id: Optional[int]) -> int:
        """
        Inserts into `observations` table. Content may be None if file-based.
        """
        insert_obs_sql = f"""
        INSERT INTO {self.observation_table_name} (content, file_id)
        VALUES (?, ?)
        """
        try:
            with self.conn:
                cursor = self.conn.execute(insert_obs_sql, (content, file_id))
                obs_id = cursor.lastrowid
        except sqlite3.Error as e:
            logger.error(f"Error inserting observation: {e}")
            raise
        return obs_id

    def _insert_embedding(self, obs_id: int, embedding: List[float]):
        """
        Inserts the embedding into observations_vec. Ensures dimension matches.
        """
        if len(embedding) != self.vector_dim:
            raise ValueError(
                f"Embedding length ({len(embedding)}) != vector_dim ({self.vector_dim})."
            )
        embedding_str = "[" + ", ".join(str(x) for x in embedding) + "]"
        insert_vec_sql = f"""
        INSERT INTO {self.observation_vector_table_name} (id, embedding)
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
        Deletes the observation by ID (including embedding). 
        Also deletes any associated file in the DB and from local storage.
        """
        # 1) Find if there's an associated file
        select_file_sql = f"""
        SELECT o.file_id, f.ref
        FROM {self.observation_table_name} o
        LEFT JOIN {self.files_table_name} f ON o.file_id = f.id
        WHERE o.id = ?
        """
        try:
            row = self.conn.execute(select_file_sql, (obs_id,)).fetchone()
            if not row:
                logger.info(f"No observation found with id={obs_id}. Nothing to delete.")
                return
        except sqlite3.Error as e:
            logger.error(f"Error fetching observation to delete: {e}")
            raise

        file_id = row["file_id"]
        file_ref = row["ref"]

        # 2) Delete from observations_vec
        sql_vec_del = f"DELETE FROM {self.observation_vector_table_name} WHERE id = ?"
        # 3) Delete from observations
        sql_obs_del = f"DELETE FROM {self.observation_table_name} WHERE id = ?"

        try:
            with self.conn:
                self.conn.execute(sql_vec_del, (obs_id,))
                self.conn.execute(sql_obs_del, (obs_id,))
        except sqlite3.Error as e:
            logger.error(f"Error deleting observation (id={obs_id}): {e}")
            raise

        # 4) If there's a file, remove it from DB and local storage
        if file_id is not None:
            try:
                if file_ref:
                    self.local_storage.delete_file(file_ref)
                sql_file_del = f"DELETE FROM {self.files_table_name} WHERE id = ?"
                with self.conn:
                    self.conn.execute(sql_file_del, (file_id,))
                logger.info(f"Deleted file id={file_id} ref={file_ref} from DB/storage.")
            except sqlite3.Error as e:
                logger.error(f"Error deleting file reference (id={file_id}): {e}")
                raise
            except Exception as e:
                logger.error(f"Error deleting physical file '{file_ref}': {e}")
                raise

        logger.info(f"Deleted observation with id={obs_id}.")

    #
    # Similarity Search
    #
    def select_similar(
        self,
        query_vector: List[float],
        top_k: int = 4,
        text_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Performs a similarity (k-NN) search using the `observations_vec` virtual table
        and returns up to `top_k` matches, joined with `observations` + `files`.

        :param query_vector: The query vector (length = self.vector_dim).
        :param top_k:        The number of top similar results to retrieve.
        :param text_only:    If True, only return rows that have no file_id.
        :return:             A list of dicts, each containing joined row + `distance`.
        """
        if len(query_vector) != self.vector_dim:
            raise ValueError(
                f"Query vector length ({len(query_vector)}) != vector_dim ({self.vector_dim})."
            )

        query_str = "[" + ", ".join(str(x) for x in query_vector) + "]"

        base_sql = f"""
        SELECT
            o.id AS observation_id,
            o.timestamp,
            o.content,
            f.id AS file_id,
            f.type AS file_type,
            f.ref AS file_ref,
            distance(ov.embedding, :query) AS distance
        FROM {self.observation_vector_table_name} ov
        JOIN {self.observation_table_name} o ON o.id = ov.id
        LEFT JOIN {self.files_table_name} f ON o.file_id = f.id
        WHERE ov.embedding MATCH :query
          AND k = :top_k
        """
        if text_only:
            base_sql += " AND o.file_id IS NULL"

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


#
# Example Usage
#
if __name__ == "__main__":
    from io import BytesIO
    from PIL import Image

    # 1) Create your embedding model
    embedding_helper = EmbeddingHelper()

    # 2) Initialize DB with that embedding model
    memory = EpisodicMemory(embedding_helper=embedding_helper, vector_dim=512)  # ensure this matches your model's embedding size

    # -- Insert text
    obs_text_id = memory.insert_text("Hello!")
    print(f"Inserted text observation ID={obs_text_id}")

    # -- Insert an image
    # Generate a red 100x100 PNG in memory
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img_buffer = BytesIO()
    img.save(img_buffer, format="PNG")
    img_bytes = img_buffer.getvalue()

    obs_img_id = memory.insert_image(img_bytes)
    print(f"Inserted image observation ID={obs_img_id}")

    # -- Insert an audio file
    # For a quick example, create 1 second of silent WAV data in memory (16-bit, 44100 Hz)
    # Normally you'd load real .wav bytes from disk or another source
    import wave
    silent_wav = BytesIO()
    with wave.open(silent_wav, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(16000)
        w.writeframes(b'\x00' * 16000 * 2)  # 1 second of silence
    audio_bytes = silent_wav.getvalue()

    obs_audio_id = memory.insert_audio(audio_bytes)
    print(f"Inserted audio observation ID={obs_audio_id}")

    # -- Perform a similarity search
    # In real usage, you'd embed some query and pass the vector in. Here, let's use the text embedding approach:
    query_emb_tensor = embedding_helper.create_text_embedding("hello world")
    query_vector = query_emb_tensor.tolist()

    results = memory.select_similar(query_vector, top_k=5, text_only=False)
    print("\nSimilarity search results:")
    for r in results:
        print(f"ObsID={r['observation_id']}, Distance={r['distance']:.4f}, FileType={r['file_type']}, Content={r['content']}")

    memory.close()
