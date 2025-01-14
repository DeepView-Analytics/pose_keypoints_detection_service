
import logging

from aiokafka import AIOKafkaProducer


class KafkaProducerService:
    def __init__(self, bootstrap_servers):
        self.bootstrap_servers = bootstrap_servers
        self.producer = None 



    async def start(self):
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: v.encode('utf-8')
        )
        await self.producer.start()


    async def trigger_tracker(self, key):
        camera_id = key.split(":")[1]
        topic = f"tracker"

        try:
            future = await self.producer.send_and_wait(topic, key)
            print(f"Message sent successfully to topic {topic}")
        except Exception as e:
            logging.error(f"Failed to send message: {e}", exc_info=True)

    async def close(self):
        await self.producer.stop()