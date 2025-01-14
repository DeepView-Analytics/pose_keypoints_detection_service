import asyncio
import io
import json
import math
import os
import time
from PIL import Image
from typing import Tuple
from aiokafka import AIOKafkaConsumer

from v3.partitioneddetectionbatch import PartitionedDetectionBatch

from pose_keypoints_estimator_service.embeding_store.EmbedingStore import EmbeddingStore
from pose_keypoints_estimator_service.kafka_interaction.kafka_producer import KafkaProducerService
from pose_keypoints_estimator_service.model.RTMPose import RTMPose
from pose_keypoints_estimator_service.redis_manager.RedisManager import RedisManager



class KafkaConsumerService:
    def __init__(self, bootstrap_servers, topic , producer):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.consumer = None  
        self.embedder = RTMPose()
        self.producer = producer
        self.embedding_store = EmbeddingStore()
        self.frames_metadata_manager_client = RedisManager(db=1)
        self.persons_metadata_manager_client = RedisManager(db=2)
        self.frames_data_manager_client = RedisManager(db=0)



        
    async def crop(self, image_bytes, xmin, ymin, xmax, ymax):
        """Private async method to crop an image based on given coordinates."""
        loop = asyncio.get_event_loop()
        img = await loop.run_in_executor(None, Image.open, io.BytesIO(image_bytes))
        cropped_img = img.crop((xmin, ymin, xmax, ymax))
        return cropped_img
    
    async def crop_images(self, frame_data, bboxs):
        images = []
        for i, bbox in enumerate(bboxs):
            for person in bbox:
                image = await self.crop(frame_data[i], person.xmin, person.ymin, person.xmax, person.ymax)
                images.append(image)
        return images
    
    async def start(self):
        # Initialize the consumer
        self.consumer = AIOKafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            value_deserializer=lambda x:  json.loads(x.decode('utf-8'))
        )
        await self.consumer.start()

        # Start the producer
        await self.producer.start()

        try:
            await self.consume_messages()
        finally:
            await self.consumer.stop()
            await self.producer.close()

    async def consume_messages(self):
        async for message in self.consumer:
            print("There is a message")
            start_time = time.time()
            request = [PartitionedDetectionBatch(**r) for r in message.value]
            bboxs = []
            frames_completed = []
            frames_keys = []
            update_keys_list = []

            for detection in request:
                key = detection.frame_key
                if detection.partition_number == detection.total_partitions:
                    frames_completed.append(f"metadata:{key}")
                bbox = detection.person_bbox
                update_keys = detection.person_keys

                bboxs.append(bbox)
                frames_keys.append(key)
                update_keys_list.extend(update_keys)

            frame_data = self.frames_data_manager_client.get_many(frames_keys)
            images = await self.crop_images(frame_data, bboxs)

            start_time_inf = time.time()
            vectors , flags , face_bboxs = await self.embedder.run_inference(images,bboxs)
            end_time_inf = time.time()
            inferance_time = end_time_inf - start_time_inf
            print(f"Time taken to inferance : {inferance_time} seconds")
            # help me here im stuck
            #where is the cast of the object ?,
            embedding_keys = self.embedding_store.save_to_milvus(vectors)
            self.persons_metadata_manager_client.update_by_field_many(update_keys_list, "is_face_clear", flags)
            self.persons_metadata_manager_client.update_by_field_many(update_keys_list, "face_bbox", face_bboxs)

            self.persons_metadata_manager_client.update_by_field_many(update_keys_list, "keypoint_key", embedding_keys)
            if frames_completed:
                self.frames_metadata_manager_client.update_by_field_many(frames_completed, "keypoint_complete", ["True" for _ in frames_completed])
                results = self.frames_metadata_manager_client.get_values_of_field_many(frames_completed, "embeding_complete")
                print("done")
                print(f"-{frames_completed[0]}-")
                test = self.persons_metadata_manager_client.get_one(update_keys_list[0])
                print(test)
                end_time = time.time()
                time_elapsed = end_time - start_time
                print(f"Time taken to execute the block of code: {time_elapsed} seconds")
                for i, result in enumerate(results):
                    if result == "True" :
                        await self.producer.trigger_tracker(frames_completed[i])
