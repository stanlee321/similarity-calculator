import os
import uuid
import json
import tempfile
import datetime
from urllib.parse import unquote, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import threading

from libs.core import Core, display_similarity_results
from libs.queues import KafkaHandler
from typing import List
from minio import Minio


class Application:
    
    def __init__(self, 
                 brokers: List[str], 
                 server_ip: str, 
                 minio_access_key: str, 
                 minio_secret_key: str,
                 working_dir: str,
                 topic_input: str,
                 topic_output: str,
                 bucket_name: str,
                 croped_image_prefix: str,
                 threshold: float = 0.5,
                 top_n: int = 10,
                 max_workers: int = 4
                 ) -> None:
         
        self.kafka_handler = KafkaHandler(bootstrap_servers=brokers)
        self.server_ip = server_ip
        self.client_minio = Minio(f"{server_ip}:9000",
                                    access_key= minio_access_key,
                                    secret_key= minio_secret_key,
                                    secure=False)
        self.topic_input = topic_input
        self.topic_output = topic_output
        
        self.bucket_name = bucket_name
        self.working_dir = working_dir
        self.croped_image_prefix = croped_image_prefix
        self.threshold = threshold
        self.top_n = top_n
        self.max_workers = max_workers
        
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
        
    def generate_uuid(self):
        return str(uuid.uuid4())

    def extract_object_path(self, url: str) -> str:
        """Extract the object path from a Minio URL."""
        parsed = urlparse(url)
        # Remove leading slash and decode URL encoding
        path = unquote(parsed.path).lstrip('/')
        # Remove bucket name from path
        if path.startswith(f"{self.bucket_name}/"):
            path = path[len(self.bucket_name)+1:]
        return path

    def download_file(self, object_name: str, local_path: str) -> tuple[bool, str, str]:
        """Download a single file from Minio.
        Returns: (success, object_name, local_path)
        """
        try:
            self.client_minio.fget_object(
                self.bucket_name,
                object_name,
                local_path
            )
            return True, object_name, local_path
        except Exception as e:
            print(f"Error downloading {object_name}: {str(e)}")
            return False, object_name, ""

    def upload_file(self, object_name: str, local_path: str) -> tuple[bool, str, str]:
        """Upload a single file to Minio.
        Returns: (success, object_name, url)
        """
        try:
            self.client_minio.fput_object(
                self.bucket_name,
                object_name,
                local_path
            )
            return True, object_name, self.get_minio_url(object_name)
        except Exception as e:
            print(f"Error uploading {object_name}: {str(e)}")
            return False, object_name, ""

    def process_message(self, message):
        _message_input = message.value
        remote_url: str = _message_input['target_image_path']
        video_id: str = _message_input['video_id']
        
        remote_path = self.extract_object_path(remote_url)
        target_image_name = os.path.basename(remote_path)
        
        print(f"Processing target image: {remote_path}")
        
        temp_dir = f"{self.working_dir}/{video_id}/similarity-calculation-{self.generate_uuid()}"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        downloaded_files = []
        
        try:
            croped_images_prefix = f"{video_id}/images/{self.croped_image_prefix}/"
            print(f"Looking for images in: {croped_images_prefix}")
            
            # Get list of all objects to download
            objects_to_download = [
                obj for obj in self.client_minio.list_objects(
                    self.bucket_name,
                    prefix=croped_images_prefix,
                    recursive=True
                ) if obj.object_name.endswith('.jpg')
            ]
            
            # Parallel download of all images
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Prepare download tasks
                download_tasks = {
                    executor.submit(
                        self.download_file,
                        obj.object_name,
                        os.path.join(temp_dir, os.path.basename(obj.object_name))
                    ): obj.object_name
                    for obj in objects_to_download
                }
                
                # Process completed downloads
                for future in as_completed(download_tasks):
                    success, object_name, local_path = future.result()
                    if success:
                        downloaded_files.append(local_path)
                        print(f"Successfully downloaded: {object_name}")
            
            print(f"Total files downloaded: {len(downloaded_files)}")
            
            if downloaded_files:
                # Download target image
                target_image_path = os.path.join(temp_dir, target_image_name)
                success, _, _ = self.download_file(remote_path, target_image_path)
                if not success:
                    raise Exception("Failed to download target image")
                
                # Process images
                core = Core(image_paths=downloaded_files)
                similar_images, similarity_scores = core.find_pairs(
                    threshold=self.threshold,
                    target_image_path=target_image_path,
                    top_n=self.top_n,
                    return_json=False
                )
                
                # Prepare results for parallel upload
                similarity_base = f"{video_id}/similarity/{target_image_name}"
                
                # Generate visualization
                plot_filename = f"visualization_{target_image_name.replace('.jpg', '.png')}"
                plot_local_path = os.path.join(temp_dir, plot_filename)
                plot_object_name = f"{similarity_base}/{plot_filename}"
                
                display_similarity_results(
                    target_image_path=target_image_path,
                    similar_images=similar_images,
                    similarity_scores=similarity_scores,
                    save_path=plot_local_path,
                    threshold=self.threshold,
                    top_n=self.top_n
                )
                
                # Prepare upload tasks
                upload_tasks = []
                result_urls = {
                    "target_image": self.get_minio_url(f"{video_id}/images/{target_image_name}"),
                    "similar_images": [],
                    "visualization": None
                }
                
                # Parallel upload of results
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # Upload visualization
                    viz_future = executor.submit(self.upload_file, plot_object_name, plot_local_path)
                    
                    # Upload similar images
                    similar_futures = []
                    for idx, local_path in enumerate(similar_images):
                        filename = os.path.basename(local_path)
                        object_name = f"{similarity_base}/similar_{idx+1}_{filename}"
                        similar_futures.append((
                            executor.submit(self.upload_file, object_name, local_path),
                            idx
                        ))
                    
                    # Process visualization result
                    success, _, url = viz_future.result()
                    if success:
                        result_urls["visualization"] = url
                    
                    # Process similar images results
                    for future, idx in similar_futures:
                        success, object_name, url = future.result()
                        if success:
                            result_urls["similar_images"].append({
                                "url": url,
                                "object_name": object_name
                            })
                
                # Create and upload final JSON
                results_json = {
                    "target_image": {
                        "url": result_urls["target_image"],
                        "filename": target_image_name
                    },
                    "total_results": len(similar_images),
                    "similar_images": [
                        {
                            "url": img["url"],
                            "object_name": img["object_name"],
                            "filename": os.path.basename(img["object_name"]),
                            "similarity_score": float(score),
                            "rank": idx + 1
                        }
                        for idx, (img, score) in enumerate(zip(result_urls["similar_images"], similarity_scores))
                    ],
                    "visualization_url": result_urls["visualization"],
                    "metadata": {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "threshold": self.threshold,
                        "top_n": self.top_n,
                        "video_id": video_id
                    }
                }
                
                # Upload JSON
                json_filename = f"results_{target_image_name.replace('.jpg', '.json')}"
                json_local_path = os.path.join(temp_dir, json_filename)
                json_object_name = f"{similarity_base}/{json_filename}"
                
                with open(json_local_path, 'w') as f:
                    json.dump(results_json, f, indent=2)
                
                success, _, url = self.upload_file(json_object_name, json_local_path)
                if success:
                    results_json["results_json_url"] = url
                
                
                #
        finally:
            # Cleanup
            if os.path.exists(temp_dir):
                for file in os.listdir(temp_dir):
                    try:
                        os.remove(os.path.join(temp_dir, file))
                    except:
                        pass
                os.rmdir(temp_dir)

    def run(self, offset: str):
        print("Starting...")
        group_id = 'similarity-calculation-' + self.generate_uuid()

        consumer = self.kafka_handler.create_consumer(self.topic_input,
                                                      group_id=group_id,
                                                      auto_offset_reset=offset)

        for message in consumer:
            print("Message received: ", message.value)
            self.process_message(message)


    def get_minio_url(self, object_name: str) -> str:
        """Generate a URL for an object in Minio."""
        return f"http://{self.server_ip}:9000/{self.bucket_name}/{object_name}"