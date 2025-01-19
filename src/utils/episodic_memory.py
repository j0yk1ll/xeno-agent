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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileType(str, Enum):
    IMAGE = "image"  # PNG
    AUDIO = "audio"  # WAV


class ObservationOutputType(str, Enum):
    """
    Represents the type of observation to filter on when doing similarity searches.
    TEXT:  Observations that have no file (i.e., text-only).
    IMAGE: Observations stored with FileType.IMAGE
    AUDIO: Observations stored with FileType.AUDIO
    """

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"


class EpisodicMemory:
    """
    A simple SQLite-based database class that stores:
      - Observations in a regular table (`observations`)
      - File metadata in a separate table (`files`)
      - Embeddings in a virtual sqlite-vec table (`observations_vec`).

    It requires `content` (text) for **all** observations (text, image, audio).

    You can:
      - Insert observations (text, image bytes, audio bytes).
      - Delete observations (and associated files).
      - Perform similarity (k-NN) searches using the sqlite-vec extension,
        with optional filtering by observation output type.
      - Use a single convenience method `select_similar_data` to automatically
        detect whether the query is text, an image (PIL.Image.Image), or audio (BytesIO).
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
          - `observations`
          - `files`
          - `observations_vec`

        Ensures `content` is NOT NULL for `observations`.
        """
        create_observations = f"""
        CREATE TABLE IF NOT EXISTS {self.observation_table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            content TEXT NOT NULL,
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
            embed_tensor = self.embedding_helper.create_text_embedding(
                text
            )  # shape [1, D]
        except Exception as e:
            raise Exception(f"An error occurred while embedding the text: {str(e)}")

        embedding = embed_tensor[0].tolist()  # convert to python list

        # Insert into observations (file_id=None for text)
        obs_id = self._insert_observation(content=text, file_id=None)

        # Insert embedding
        self._insert_embedding(obs_id, embedding)

        logger.info(f"Inserted text observation id={obs_id}, content='{text[:30]}'.")
        return obs_id

    def insert_image(self, image_bytes: bytes, content: str) -> int:
        """
        Inserts an observation with an image file. Also embeds the image internally.

        :param image_bytes: Raw bytes of the image (e.g., from a PNG file).
        :param content:     A textual description or other required text for the observation.
        """
        # Embed the image
        try:
            embed_tensor = self.embedding_helper.create_image_embedding(image_bytes)
        except Exception as e:
            raise Exception(f"An error occurred while embedding the image: {str(e)}")

        # Some embedding helpers return shape [D], others [1, D]. Adjust if needed:
        if len(embed_tensor.shape) == 2:
            embedding = embed_tensor[0].tolist()
        else:
            embedding = embed_tensor.tolist()

        # Store the file (type=IMAGE) -> get back file_id
        file_id, file_ref = self._insert_file(FileType.IMAGE, image_bytes)

        # Insert the observation
        obs_id = self._insert_observation(content=content, file_id=file_id)

        # Insert embedding
        self._insert_embedding(obs_id, embedding)

        logger.info(f"Inserted image observation id={obs_id}, file_ref={file_ref}.")
        return obs_id

    def insert_audio(self, audio_bytes: bytes, content: str) -> int:
        """
        Inserts an observation with an audio file. Also embeds the audio internally.

        :param audio_bytes: Raw bytes of the audio (e.g., WAV).
        :param content:     Required content text associated with this observation.
        """
        # Embed the audio
        try:
            embed_tensor = self.embedding_helper.create_audio_embedding(audio_bytes)
        except Exception as e:
            raise Exception(f"An error occurred while embedding audio: {str(e)}")

        if len(embed_tensor.shape) == 2:
            embedding = embed_tensor[0].tolist()
        else:
            embedding = embed_tensor.tolist()

        # Insert file (type=AUDIO)
        file_id, file_ref = self._insert_file(FileType.AUDIO, audio_bytes)

        # Insert observation row
        obs_id = self._insert_observation(content=content, file_id=file_id)

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

    def _insert_observation(self, content: str, file_id: Optional[int]) -> int:
        """
        Inserts into `observations` table. `content` must not be None.
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
                logger.info(
                    f"No observation found with id={obs_id}. Nothing to delete."
                )
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
                logger.info(
                    f"Deleted file id={file_id} ref={file_ref} from DB/storage."
                )
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
    def _select_similar(
        self,
        query_vector: List[float],
        top_k: int = 4,
        output_types: Optional[List[ObservationOutputType]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Performs a similarity (k-NN) search using the `observations_vec` virtual table
        and returns up to `top_k` matches, joined with `observations` + `files`.

        :param query_vector: The query vector (length = self.vector_dim).
        :param top_k:        The number of top similar results to retrieve.
        :param output_types: Which output observation types to include in the results
                             (e.g. [ObservationOutputType.TEXT, ObservationOutputType.IMAGE]).
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
        self, output_types: Optional[List[ObservationOutputType]]
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
        if ObservationOutputType.TEXT in output_types:
            # text means no file
            conditions.append("(o.file_id IS NULL)")
        if ObservationOutputType.IMAGE in output_types:
            conditions.append("(f.type = 'image')")
        if ObservationOutputType.AUDIO in output_types:
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
        output_types: Optional[List[ObservationOutputType]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convenience method to perform similarity search from one of:
          - A text string
          - A PIL image
          - A BytesIO object (assumed to be audio)

        Internally detects type, calls the appropriate embedding helper method,
        then calls select_similar with the resulting vector.
        """
        # 1) Determine input type
        if isinstance(data, str):
            # Treat as text
            try:
                embed_tensor = self.embedding_helper.create_text_embedding(data)
            except Exception as e:
                raise Exception(f"Error embedding query text: {e}")
            # shape [1, D] or [D]
            if len(embed_tensor.shape) == 2:
                query_vector = embed_tensor[0].tolist()
            else:
                query_vector = embed_tensor.tolist()

        elif isinstance(data, Image.Image):
            # Convert PIL image to raw bytes (PNG) then embed
            buffer = BytesIO()
            data.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
            try:
                embed_tensor = self.embedding_helper.create_image_embedding(image_bytes)
            except Exception as e:
                raise Exception(f"Error embedding query image: {e}")
            if len(embed_tensor.shape) == 2:
                query_vector = embed_tensor[0].tolist()
            else:
                query_vector = embed_tensor.tolist()

        elif isinstance(data, BytesIO):
            # Assume audio
            audio_bytes = data.getvalue()
            try:
                embed_tensor = self.embedding_helper.create_audio_embedding(audio_bytes)
            except Exception as e:
                raise Exception(f"Error embedding query audio: {e}")
            if len(embed_tensor.shape) == 2:
                query_vector = embed_tensor[0].tolist()
            else:
                query_vector = embed_tensor.tolist()

        else:
            raise ValueError(
                "Unsupported data type. Please pass str (text), PIL.Image.Image, or BytesIO (audio)."
            )

        # 2) Perform similarity search
        return self._select_similar(query_vector, top_k=top_k, output_types=output_types)


#
# Example Usage
#
if __name__ == "__main__":
    # Example usage of the updated EpisodicMemory class.
    import wave

    # 1) Create your embedding model
    embedding_helper = EmbeddingHelper()

    # 2) Initialize DB with that embedding model
    memory = EpisodicMemory(embedding_helper=embedding_helper, vector_dim=512)

    # -- Insert text
    obs_text_id = memory.insert_text("Hello from text!")
    print(f"Inserted text observation ID={obs_text_id}")

    # -- Insert an image (red 100x100 PNG in memory)
    red_img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img_buffer = BytesIO()
    red_img.save(img_buffer, format="PNG")
    img_bytes = img_buffer.getvalue()

    obs_img_id = memory.insert_image(img_bytes, content="A red square image")
    print(f"Inserted image observation ID={obs_img_id}")

    # -- Insert an audio file (1 second of silence, 16-bit, 16kHz)
    silent_wav = BytesIO()
    with wave.open(silent_wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(16000)
        w.writeframes(b"\x00" * 16000 * 2)  # 1 second of silence
    audio_bytes = silent_wav.getvalue()

    obs_audio_id = memory.insert_audio(audio_bytes, content="One second of silence")
    print(f"Inserted audio observation ID={obs_audio_id}")

    # -- Single convenience method for similarity (text example)
    results_text = memory.select_similar_data(
        "hello world",
        top_k=5,
        output_types=[ObservationOutputType.TEXT, ObservationOutputType.IMAGE],
    )
    print("\nSimilarity search (text query) results:")
    for r in results_text:
        print(
            f"ObsID={r['observation_id']}, Distance={r['distance']:.4f}, "
            f"FileType={r['file_type']}, Content={r['content']}"
        )

    # -- Single convenience method for similarity (image example)
    # We'll just re-use the BytesIO from earlier, but let's re-wrap it:
    results_image = memory.select_similar_data(
        Image.open(BytesIO(img_bytes)),
        top_k=5,
        output_types=[ObservationOutputType.IMAGE, ObservationOutputType.AUDIO],
    )
    print("\nSimilarity search (image query) results:")
    for r in results_image:
        print(
            f"ObsID={r['observation_id']}, Distance={r['distance']:.4f}, "
            f"FileType={r['file_type']}, Content={r['content']}"
        )

    # -- Single convenience method for similarity (audio example)
    # Pass a BytesIO object
    results_audio = memory.select_similar_data(
        BytesIO(audio_bytes),
        top_k=5,
        output_types=[ObservationOutputType.TEXT, ObservationOutputType.AUDIO],
    )
    print("\nSimilarity search (audio query) results:")
    for r in results_audio:
        print(
            f"ObsID={r['observation_id']}, Distance={r['distance']:.4f}, "
            f"FileType={r['file_type']}, Content={r['content']}"
        )

    # Cleanup
    memory.close()
