import json
import os

import time
from typing import List, Dict , Any
from rediscluster import ClusterError, RedisCluster, RedisClusterException 

class RedisManager:
    def __init__(self, db, max_retries=5, backoff_factor=0.1):
        self.db_prefix = f"db{db}:"
        self.host = os.getenv('REDIS_SERVER', 'localhost')
        self.port = os.getenv('REDIS_PORT', 7001)
        print(self.host)
        startup_nodes = [
            {"host": self.host, "port": self.port}
        ]
        # Initialize RedisCluster client
        self.client = RedisCluster(
            startup_nodes=startup_nodes
        )
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
    def _get_key_with_db_prefix(self, key: str) -> str:
        """Concatenate the db prefix to the key."""
        return self.db_prefix + key

    def save_one(self, key: str, value: Dict[str, Any]):

        key_with_db = self._get_key_with_db_prefix(key)
        self.client.hset(key_with_db, mapping=value)

    def get_one(self, key: str):

        key_with_db = self._get_key_with_db_prefix(key)
        return self.client.hgetall(key_with_db)

    def save_many(self, keys: List[str], values: List[Dict[str, Any]]):
        with self.client.pipeline() as pipe:
            for key, value in zip(keys, values):
                key_with_db = self._get_key_with_db_prefix(key)
                # Serialize the dictionary to a JSON string
                serialized_value = {field: json.dumps(v) for field, v in value.items()}
                # Use HSET with serialized fields and values
                for field, val in serialized_value.items():
                    pipe.hset(key_with_db, field, val)
            pipe.execute()

    def get_many(self, keys: List[str]):
        with self.client.pipeline() as pipe:
            for key in keys:
                key_with_db = self._get_key_with_db_prefix(key)
                pipe.get(key_with_db)
            return pipe.execute()

    def update_by_field_one(self, key: str, field: str, value):
        key_with_db = self._get_key_with_db_prefix(key)
        lua_script = """
        local key = KEYS[1]
        local field = ARGV[1]
        local value = ARGV[2]
        
        redis.call('HSET', key, field, value)
        return redis.call('HGET', key, field)
        """
        script = self.client.register_script(lua_script)
        for attempt in range(self.max_retries):
            try:
                return script(keys=[key_with_db], args=[field, value])
            except RedisClusterException:
                time.sleep(self.backoff_factor * (2 ** attempt))  # Exponential backoff
        raise Exception(f"Failed to update field '{field}' in key '{key_with_db}' after {self.max_retries} attempts")

    def update_by_field_many(self, keys: List[str], field: str, values: List[str]):
        """Update a specific field in multiple hashes using a pipeline in Redis Cluster."""
        # Validate inputs
        if not all(isinstance(key, str) for key in keys):
            raise ValueError("All keys must be strings.")
        if not isinstance(field, str):
            raise ValueError("Field must be a string.")
        if not all(isinstance(value, str) for value in values):
            raise ValueError("All values must be strings.")
        if len(values) != len(keys):
            raise ValueError("The number of values must match the number of keys.")

        # Retry mechanism
        for attempt in range(self.max_retries):
            try:
                # Prepend the db prefix to the keys
                keys_with_db = [self._get_key_with_db_prefix(key) for key in keys]
                
                # Create a pipeline
                with self.client.pipeline() as pipe:
                    for key, value in zip(keys_with_db, values):
                        pipe.hset(key, field, value)
                    
                    # Execute the pipeline
                    pipe.execute()

                return  # Exit if the pipeline executes successfully
            except RedisClusterException as e:
                print(e)
                time.sleep(self.backoff_factor * (2 ** attempt))  # Exponential backoff
            except ClusterError as e:
                print(f"Redis error during update: {e}")
                raise
        # If retries are exhausted
        raise Exception(f"Failed to update field '{field}' in keys '{keys}' after {self.max_retries} attempts")


    def get_values_of_field_many(self, keys: List[str], field: str) -> List[str]:
        lua_script = """
        local result = {}
        for i, key in ipairs(KEYS) do
            local value = redis.call('HGET', key, ARGV[1])
            if value then
                result[i] = value
            else
                result[i] = ''  -- Return an empty string if the field doesn't exist
            end
        end
        return result
        """
        keys_with_db = [self._get_key_with_db_prefix(key) for key in keys]
        script = self.client.register_script(lua_script)
        results = script(keys=keys_with_db, args=[field])
        return [result.decode('utf-8') if result else '' for result in results]


