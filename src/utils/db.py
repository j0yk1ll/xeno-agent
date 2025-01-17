import threading
from pathlib import Path
from pony.orm import Database, db_session, commit, Required, Set
from db_models import entities

class DatabaseManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DatabaseManager, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.db = Database()
        self.entity_map = {}

        # Setup the .xeno directory in the user's home directory
        xeno_dir = Path.home() / ".xeno"
        xeno_dir.mkdir(parents=True, exist_ok=True)  # Ensure the folder exists

        # Define the path for the SQLite database file
        db_path = xeno_dir / "database.sqlite"

        # Bind the database to the SQLite file
        self.db.bind(provider='sqlite', filename=str(db_path), create_db=True)

        # Bind entities and generate mapping
        for entity in entities:
            self.entity_map[entity.__name__] = entity
            entity.bind(self.db)
        self.db.generate_mapping(create_tables=True)

    def _get_entity_class(self, entity_name):
        entity_class = self.entity_map.get(entity_name)
        if not entity_class:
            raise ValueError(f"Entity '{entity_name}' not found in the entity map.")
        return entity_class

    def _validate_include(self, entity_class, include):
        """
        Validate that each item in the include list is a valid relationship of the entity.
        """
        valid_relations = {rel for rel in entity_class._rel_names}
        invalid_relations = set(include) - valid_relations
        if invalid_relations:
            raise ValueError(
                f"Invalid include parameters: {', '.join(invalid_relations)}. "
                f"Valid relationships for '{entity_class.__name__}': {', '.join(valid_relations)}."
            )

    def _handle_to_dict(self, obj, include):
        """
        Helper method to handle the to_dict call with appropriate parameters.
        """
        if include:
            self._validate_include(obj.__class__, include)
            return obj.to_dict(
                include=include,
                related_objects=True,
                max_depth=2  # Adjust depth as needed
            )
        else:
            return obj.to_dict(
                include=None,
                related_objects=False
            )

    def _handle_to_dict_list(self, objects, include):
        """
        Helper method to handle a list of objects for to_dict.
        """
        if include:
            # Validate all included relationships for each object
            for obj in objects:
                self._validate_include(obj.__class__, include)
            return [
                obj.to_dict(
                    include=include,
                    related_objects=True,
                    max_depth=2  # Adjust depth as needed
                )
                for obj in objects
            ]
        else:
            return [
                obj.to_dict(
                    include=None,
                    related_objects=False
                )
                for obj in objects
            ]

    @db_session
    def insert_one(self, entity_name, data, include=None):
        """
        Insert a single record and optionally include related entities in the return value.

        :param entity_name: Name of the entity.
        :param data: Dictionary of data to insert.
        :param include: List of related entity names to include in the return value.
        :return: Dictionary representation of the inserted record with optional related entities.
        :raises ValueError: If entity_name is invalid or include contains invalid relationships.
        :raises Exception: For other unforeseen errors during insertion.
        """
        try:
            entity_class = self._get_entity_class(entity_name)
            processed_data = self._process_references(entity_class, data)
            obj = entity_class(**processed_data)
            commit()
            return self._handle_to_dict(obj, include)
        except ValueError as ve:
            # Re-raise known value errors
            raise ve
        except Exception as e:
            # Handle unexpected exceptions
            raise Exception(f"Failed to insert record into '{entity_name}': {str(e)}")

    @db_session
    def insert_many(self, entity_name, items, include=None):
        """
        Insert multiple records and optionally include related entities in the return values.

        :param entity_name: Name of the entity.
        :param items: List of dictionaries containing data to insert.
        :param include: List of related entity names to include in the return values.
        :return: List of dictionary representations of the inserted records with optional related entities.
        :raises ValueError: If entity_name is invalid or include contains invalid relationships.
        :raises Exception: For other unforeseen errors during insertion.
        """
        try:
            entity_class = self._get_entity_class(entity_name)
            created_objects = []
            for item in items:
                processed_item = self._process_references(entity_class, item)
                obj = entity_class(**processed_item)
                created_objects.append(obj)
            commit()
            return self._handle_to_dict_list(created_objects, include)
        except ValueError as ve:
            raise ve
        except Exception as e:
            raise Exception(f"Failed to insert multiple records into '{entity_name}': {str(e)}")

    def _process_references(self, entity_class, data):
        """
        Process reference fields in the data. If a field is a relationship,
        it can accept either an instance or an ID.

        :param entity_class: The Pony ORM entity class.
        :param data: The data dictionary to process.
        :return: Processed data dictionary with resolved references.
        :raises ValueError: If reference resolution fails.
        """
        processed_data = {}
        for key, value in data.items():
            attr = entity_class._get_attribute(key)
            if isinstance(attr, (Required, Set)):
                if hasattr(attr, 'entity') and value is not None:
                    related_entity = self._resolve_reference(attr, value)
                    processed_data[key] = related_entity
                else:
                    processed_data[key] = value
            else:
                processed_data[key] = value
        return processed_data

    def _resolve_reference(self, attr, value):
        """
        Resolve the reference for a relationship field.
        If value is an integer, assume it's an ID and fetch the related object.
        If it's an instance of the related entity, return it directly.

        :param attr: The Pony ORM attribute representing the relationship.
        :param value: The value to resolve (ID or instance).
        :return: Resolved related entity instance.
        :raises ValueError: If resolution fails.
        """
        related_entity_class = attr.entity
        if isinstance(value, related_entity_class):
            return value
        elif isinstance(value, int):
            related_entity = related_entity_class.get(id=value)
            if not related_entity:
                raise ValueError(f"Related entity with id {value} does not exist.")
            return related_entity
        else:
            raise ValueError(f"Invalid reference for field '{attr.name}': {value}")

    @db_session
    def update_one(self, entity_name, item_id, data, include=None):
        """
        Update a single record by its ID and optionally include related entities in the return value.

        :param entity_name: Name of the entity.
        :param item_id: ID of the record to update.
        :param data: Dictionary of data to update.
        :param include: List of related entity names to include in the return value.
        :return: Dictionary representation of the updated record with optional related entities.
        :raises ValueError: If entity_name is invalid, item_id does not exist, or include contains invalid relationships.
        :raises Exception: For other unforeseen errors during update.
        """
        try:
            entity_class = self._get_entity_class(entity_name)
            entity = entity_class.get(id=item_id)
            if not entity:
                raise ValueError(f"Entity '{entity_name}' with id {item_id} does not exist.")
            processed_data = self._process_references(entity_class, data)
            for key, value in processed_data.items():
                setattr(entity, key, value)
            commit()
            return self._handle_to_dict(entity, include)
        except ValueError as ve:
            raise ve
        except Exception as e:
            raise Exception(f"Failed to update record in '{entity_name}': {str(e)}")

    @db_session
    def update_many(self, entity_name, updates, include=None):
        """
        Update multiple records and optionally include related entities in the return values.

        :param entity_name: Name of the entity.
        :param updates: List of dictionaries containing updates. Each dictionary must include 'id'.
        :param include: List of related entity names to include in the return values.
        :return: List of dictionary representations of the updated records with optional related entities.
        :raises ValueError: If entity_name is invalid, 'id' is missing, or include contains invalid relationships.
        :raises Exception: For other unforeseen errors during updates.
        """
        try:
            entity_class = self._get_entity_class(entity_name)
            updated_objects = []
            for update in updates:
                item_id = update.get('id')
                if item_id is None:
                    raise ValueError("Update dictionary must include 'id'.")
                entity = entity_class.get(id=item_id)
                if entity:
                    processed_update = self._process_references(entity_class, update)
                    for key, value in processed_update.items():
                        if key != 'id':
                            setattr(entity, key, value)
                    updated_objects.append(entity)
                else:
                    raise ValueError(f"Entity '{entity_name}' with id {item_id} does not exist.")
            commit()
            return self._handle_to_dict_list(updated_objects, include)
        except ValueError as ve:
            raise ve
        except Exception as e:
            raise Exception(f"Failed to update multiple records in '{entity_name}': {str(e)}")

    @db_session
    def delete_one(self, entity_name, item_id):
        """
        Delete a single record by its ID.

        :param entity_name: Name of the entity.
        :param item_id: ID of the record to delete.
        :raises ValueError: If entity_name is invalid or item_id does not exist.
        :raises Exception: For other unforeseen errors during deletion.
        """
        try:
            entity_class = self._get_entity_class(entity_name)
            entity = entity_class.get(id=item_id)
            if entity:
                entity.delete()
                commit()
            else:
                raise ValueError(f"Entity '{entity_name}' with id {item_id} does not exist.")
        except ValueError as ve:
            raise ve
        except Exception as e:
            raise Exception(f"Failed to delete record from '{entity_name}': {str(e)}")

    @db_session
    def delete_many(self, entity_name, ids):
        """
        Delete multiple records by their IDs.

        :param entity_name: Name of the entity.
        :param ids: List of IDs of the records to delete.
        :raises ValueError: If entity_name is invalid or any item_id does not exist.
        :raises Exception: For other unforeseen errors during deletion.
        """
        try:
            entity_class = self._get_entity_class(entity_name)
            for item_id in ids:
                entity = entity_class.get(id=item_id)
                if entity:
                    entity.delete()
                else:
                    raise ValueError(f"Entity '{entity_name}' with id {item_id} does not exist.")
            commit()
        except ValueError as ve:
            raise ve
        except Exception as e:
            raise Exception(f"Failed to delete multiple records from '{entity_name}': {str(e)}")

    @db_session
    def select_one(self, entity_name, item_id, include=None):
        """
        Retrieve a single record by its ID, with optional related entities.

        :param entity_name: Name of the entity.
        :param item_id: ID of the record.
        :param include: List of related entity names to include.
        :return: Dictionary representation of the record.
        :raises ValueError: If entity_name is invalid or include contains invalid relationships.
        :raises Exception: For other unforeseen errors during retrieval.
        """
        try:
            entity_class = self._get_entity_class(entity_name)
            entity = entity_class.get(id=item_id)
            if entity:
                return self._handle_to_dict(entity, include)
            else:
                return None
        except ValueError as ve:
            raise ve
        except Exception as e:
            raise Exception(f"Failed to select record from '{entity_name}': {str(e)}")

    @db_session
    def select_many(self, entity_name, filters=None, include=None):
        """
        Retrieve multiple records with optional filters and related entities.

        :param entity_name: Name of the entity.
        :param filters: Dictionary of filter conditions (e.g., {'name': 'John'}).
        :param include: List of related entity names to include.
        :return: List of dictionary representations of matching records.
        :raises ValueError: If entity_name is invalid, filters contain invalid fields, or include contains invalid relationships.
        :raises Exception: For other unforeseen errors during retrieval.
        """
        try:
            entity_class = self._get_entity_class(entity_name)
            if not filters:
                query = entity_class.select()
            else:
                query = entity_class.select()
                for key, value in filters.items():
                    if key in entity_class.__dict__:
                        # Dynamically build filter expression
                        # Pony ORM's filter requires lambdas or expressions; using getattr for simplicity
                        query = query.filter(lambda e, k=key, v=value: getattr(e, k) == v)
                    else:
                        raise ValueError(f"Invalid filter field: {key} for entity '{entity_name}'.")
            results = query[:]
            return self._handle_to_dict_list(results, include)
        except ValueError as ve:
            raise ve
        except Exception as e:
            raise Exception(f"Failed to select multiple records from '{entity_name}': {str(e)}")

# Usage Example with Error Handling
if __name__ == "__main__":
    db_manager = DatabaseManager()
    
    try:
        # Insert a new File without including related entities
        new_file = db_manager.insert_one('File', {'path': 'path/to/file'})
        print("Inserted File:", new_file)
        
        # Insert a new Observation referencing the File by ID with include
        new_observation = db_manager.insert_one(
            'Observation',
            {'content': 'A new file', 'file': new_file['id']},
            include=['file']
        )
        print("Inserted Observation with included File:", new_observation)
        
        # Insert a new Observation referencing the File by object
        new_observation_obj = db_manager.insert_one(
            'Observation',
            {'content': 'Another file', 'file': new_file},
            include=['file']
        )
        print("Inserted Observation with included File (object reference):", new_observation_obj)
        
        # Insert multiple Observations with references
        observations = db_manager.insert_many(
            'Observation',
            [
                {'content': 'Observation 1', 'file': new_file['id']},
                {'content': 'Observation 2', 'file': new_file},
            ],
            include=['file']
        )
        print("Inserted Multiple Observations:", observations)
        
        # Select a single Observation with included File
        single = db_manager.select_one('Observation', new_observation['id'], include=['file'])
        print("Selected Single Observation:", single)
        
        # Select multiple Observations with included File
        multiple = db_manager.select_many('Observation', {'content': 'A new file'}, include=['file'])
        print("Selected Multiple Observations:", multiple)
        
    except ValueError as ve:
        print(f"Value Error: {ve}")
    except Exception as e:
        print(f"Error: {e}")
