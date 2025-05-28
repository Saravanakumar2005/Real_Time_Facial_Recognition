import cv2
import numpy as np
import json
import sys
import pickle
from datetime import datetime
import logging
from sklearn.metrics.pairwise import cosine_similarity
import insightface
from insightface.app import FaceAnalysis
import pymongo
from bson import ObjectId
import threading
import time
import base64
import io
from PIL import Image
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceRecognitionRAG:
    def __init__(self):
        # MongoDB connection
        self.client = pymongo.MongoClient('mongodb://localhost:27017/')
        self.db = self.client['facialrecognition']
        self.faces_collection = self.db['faces']
        self.logs_collection = self.db['recognitionlogs']
        self.sessions_collection = self.db['sessions']
        
        self.recognition_threshold = 0.4
        self.similarity_threshold = 0.6
        
        # Initialize InsightFace
        self.init_insightface()
        
        # Load all faces into memory for RAG
        self.load_all_faces()
        
        # RAG components
        self.face_index = {}  # Face embedding index for fast retrieval
        self.face_metadata = {}  # Additional metadata for context
        
    def init_insightface(self):
        """Initialize InsightFace models"""
        try:
            self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("✅ InsightFace models loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize InsightFace: {e}")
            self.app = None
            
    def load_all_faces(self):
        """Load all faces from MongoDB for RAG retrieval"""
        try:
            faces = list(self.faces_collection.find())
            self.face_encodings = []
            self.face_names = []
            self.face_ids = []
            self.face_index = {}
            self.face_metadata = {}
            
            for face in faces:
                embedding = np.array(face['embedding'])
                face_id = str(face['_id'])
                
                self.face_encodings.append(embedding)
                self.face_names.append(face['name'])
                self.face_ids.append(face_id)
                
                # Build RAG index
                self.face_index[face_id] = {
                    'embedding': embedding,
                    'name': face['name'],
                    'created_at': face.get('created_at', datetime.now()),
                    'last_seen': face.get('last_seen', datetime.now())
                }
                
                # Store metadata for context generation
                self.face_metadata[face_id] = {
                    'recognition_count': face.get('recognition_count', 0),
                    'locations': face.get('locations', []),
                    'tags': face.get('tags', []),
                    'notes': face.get('notes', ''),
                    'quality_scores': face.get('quality_scores', [])
                }
            
            logger.info(f"✅ Loaded {len(self.face_encodings)} faces into RAG system")
            
        except Exception as e:
            logger.error(f"❌ Failed to load faces: {e}")
            
    def extract_face_features(self, image_input, input_type='path'):
        """Extract face features from image"""
        try:
            if self.app is None:
                return None
            
            if input_type == 'path':
                image = cv2.imread(image_input)
            elif input_type == 'base64':
                image_data = base64.b64decode(image_input.split(',')[1] if ',' in image_input else image_input)
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            elif input_type == 'array':
                image = image_input
            else:
                return None
            
            if image is None:
                return None
                
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = self.app.get(rgb_image)
            
            if not faces:
                return None
            
            best_face = max(faces, key=lambda x: x.det_score)
            embedding = best_face.embedding / np.linalg.norm(best_face.embedding)
            quality_score = min(best_face.det_score + 0.3, 1.0)
            
            return {
                'embedding': embedding.tolist(),
                'bbox': best_face.bbox.astype(int).tolist(),
                'quality_score': quality_score,
                'det_score': best_face.det_score,
                'normalized_embedding': embedding
            }
            
        except Exception as e:
            logger.error(f"Face feature extraction failed: {e}")
            return None
    
    def retrieve_similar_faces(self, query_embedding, top_k=5):
        """RAG Retrieve: Find similar faces from the database"""
        try:
            if len(self.face_encodings) == 0:
                return []
            
            similarities = cosine_similarity([query_embedding], self.face_encodings)[0]
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            similar_faces = []
            for idx in top_indices:
                if similarities[idx] > self.similarity_threshold:
                    face_id = self.face_ids[idx]
                    similar_faces.append({
                        'face_id': face_id,
                        'name': self.face_names[idx],
                        'similarity': float(similarities[idx]),
                        'metadata': self.face_metadata.get(face_id, {}),
                        'index': idx
                    })
            
            return similar_faces
            
        except Exception as e:
            logger.error(f"Similar face retrieval failed: {e}")
            return []
    
    def augment_context(self, similar_faces, query_info=None):
        """RAG Augment: Generate context from retrieved faces"""
        try:
            if not similar_faces:
                return {
                    'context': 'No similar faces found in database',
                    'recommendations': ['Consider adding this as a new person'],
                    'confidence': 0.0
                }
            
            context_parts = []
            total_confidence = 0
            
            for face in similar_faces:
                metadata = face['metadata']
                context_parts.append(f"Similar person: {face['name']} (similarity: {face['similarity']:.3f})")
                if metadata.get('recognition_count', 0) > 0:
                    context_parts.append(f"  - Recognized {metadata['recognition_count']} times")
                if metadata.get('locations'):
                    context_parts.append(f"  - Last seen locations: {', '.join(metadata['locations'][-3:])}")
                if metadata.get('tags'):
                    context_parts.append(f"  - Tags: {', '.join(metadata['tags'])}")
                total_confidence += face['similarity']
            
            avg_confidence = total_confidence / len(similar_faces)
            recommendations = []
            if avg_confidence > 0.8:
                recommendations.append("High confidence match - this is likely the same person")
            elif avg_confidence > 0.6:
                recommendations.append("Medium confidence - verify identity before proceeding")
            else:
                recommendations.append("Low confidence - this might be a new person")
            if len(similar_faces) > 1:
                recommendations.append("Multiple similar faces found - check for duplicates")
            
            return {
                'context': '\n'.join(context_parts),  
                'recommendations': recommendations,
                'confidence': avg_confidence,
                'similar_count': len(similar_faces)
            }
            
        except Exception as e:
            logger.error(f"Context augmentation failed: {e}")
            return {'context': 'Error generating context', 'recommendations': [], 'confidence': 0.0}
    
    def generate_response(self, query_face, similar_faces, augmented_context, action='recognize'):
        """RAG Generate: Create final response based on context"""
        try:
            response = {
                'timestamp': datetime.now().isoformat(),
                'action': action,
                'face_detected': query_face is not None,
                'context': augmented_context,
                'results': []
            }
            
            if query_face is None:
                response['message'] = "No face detected in the image"
                return response
            
            if action == 'recognize':
                if similar_faces and augmented_context['confidence'] > self.recognition_threshold:
                    best_match = similar_faces[0]
                    response['recognized'] = True
                    response['person'] = {
                        'id': best_match['face_id'],
                        'name': best_match['name'],
                        'confidence': best_match['similarity']
                    }
                    response['message'] = f"Recognized as {best_match['name']} with {best_match['similarity']:.1%} confidence"
                    self.update_recognition_stats(best_match['face_id'])
                else:
                    response['recognized'] = False
                    response['message'] = "Person not recognized in database"
                    
            elif action == 'register':
                response['registration_ready'] = True
                response['message'] = "Face extracted and ready for registration"
                
            elif action == 'search':
                response['search_results'] = similar_faces
                response['message'] = f"Found {len(similar_faces)} similar faces"
            
            response['technical'] = {
                'face_quality': query_face.get('quality_score', 0),
                'detection_score': query_face.get('det_score', 0),
                'bbox': query_face.get('bbox', []),
                'similar_faces_count': len(similar_faces)
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def update_recognition_stats(self, face_id):
        """Update recognition statistics"""
        try:
            self.faces_collection.update_one(
                {'_id': ObjectId(face_id)},
                {
                    '$inc': {'recognition_count': 1},
                    '$set': {'last_seen': datetime.now()}
                }
            )
            self.logs_collection.insert_one({
                'face_id': face_id,
                'action': 'recognized',
                'timestamp': datetime.now(),
                'confidence': self.face_index[face_id].get('last_confidence', 0)
            })
        except Exception as e:
            logger.error(f"Stats update failed: {e}")
    
    def register_new_face(self, image_input, person_name, input_type='path', metadata=None):
        """Register a new face in the RAG system"""
        try:
            face_data = self.extract_face_features(image_input, input_type)
            if not face_data:
                return {'success': False, 'error': 'No face detected'}
            
            similar_faces = self.retrieve_similar_faces(face_data['normalized_embedding'])
            if similar_faces and similar_faces[0]['similarity'] > 0.9:
                return {
                    'success': False, 
                    'error': 'Similar face already exists',
                    'similar_person': similar_faces[0]['name']
                }
            
            face_doc = {
                'name': person_name,
                'embedding': face_data['embedding'],
                'created_at': datetime.now(),
                'last_seen': datetime.now(),
                'recognition_count': 0,
                'quality_score': face_data['quality_score'],
                'det_score': face_data['det_score'],
                'bbox': face_data['bbox'],
                'metadata': metadata or {}
            }
            
            result = self.faces_collection.insert_one(face_doc)
            face_id = str(result.inserted_id)
            self.load_all_faces()
            
            return {
                'success': True,
                'face_id': face_id,
                'message': f'Successfully registered {person_name}'
            }
            
        except Exception as e:
            logger.error(f"Face registration failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def list_registered_people(self):
        """List all registered people"""
        try:
            faces = list(self.faces_collection.find())
            people = []
            for face in faces:
                people.append({
                    'name': face['name'],
                    'image_count': 1,  # Assuming one image per person
                    'recognition_count': face.get('recognition_count', 0),
                    'last_seen': face.get('last_seen', datetime.now()).isoformat()
                })
            return {'success': True, 'people': people}
        except Exception as e:
            logger.error(f"List registered people failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def delete_person(self, name):
        """Delete a person by name"""
        try:
            result = self.faces_collection.delete_many({'name': name})
            if result.deleted_count > 0:
                self.load_all_faces()
                return {'success': True, 'message': f'Successfully deleted {name}'}
            return {'success': False, 'error': f'No person found with name {name}'}
        except Exception as e:
            logger.error(f"Delete person failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def clear_all_data(self):
        """Clear all face data"""
        try:
            self.faces_collection.delete_many({})
            self.logs_collection.delete_many({})
            self.sessions_collection.delete_many({})
            self.load_all_faces()
            return {'success': True, 'message': 'All data cleared successfully'}
        except Exception as e:
            logger.error(f"Clear all data failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def process_rag_request(self, image_input, action='recognize', person_name=None, input_type='path'):
        """Main RAG processing pipeline"""
        try:
            query_face = self.extract_face_features(image_input, input_type)
            if not query_face and action != 'status':
                return {'error': 'No face detected', 'timestamp': datetime.now().isoformat()}
            
            similar_faces = []
            if query_face:
                similar_faces = self.retrieve_similar_faces(query_face['normalized_embedding'])
            
            augmented_context = self.augment_context(similar_faces, query_face)
            response = self.generate_response(query_face, similar_faces, augmented_context, action)
            
            if action == 'register' and person_name and query_face:
                register_result = self.register_new_face(image_input, person_name, input_type)
                response.update(register_result)
            
            return response
            
        except Exception as e:
            logger.error(f"RAG request processing failed: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def get_system_status(self):
        """Get RAG system status"""
        try:
            total_faces = len(self.face_encodings)
            total_logs = self.logs_collection.count_documents({})
            
            return {
                'status': 'active',
                'total_faces': total_faces,
                'total_recognitions': total_logs,
                'insightface_loaded': self.app is not None,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

# CLI interface for server.js integration
def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Invalid arguments. Usage: python script.py <action> [arguments...]"}))
        sys.exit(1)
    
    processor = FaceRecognitionRAG()
    action = sys.argv[1].lower()
    
    try:
        if action == 'recognize' and len(sys.argv) >= 3:
            image_path = sys.argv[2]
            result = processor.process_rag_request(image_path, 'recognize')
            print(json.dumps(result))
            
        elif action == 'register' and len(sys.argv) >= 4:
            image_path = sys.argv[2]
            person_name = sys.argv[3]
            result = processor.process_rag_request(image_path, 'register', person_name)
            print(json.dumps(result))
            
        elif action == 'search' and len(sys.argv) >= 3:
            image_path = sys.argv[2]
            result = processor.process_rag_request(image_path, 'search')
            print(json.dumps(result))
            
        elif action == 'status':
            result = processor.get_system_status()
            print(json.dumps(result))
            
        elif action == 'list-people':
            result = processor.list_registered_people()
            print(json.dumps(result))
            
        elif action == 'delete-person' and len(sys.argv) >= 3:
            name = sys.argv[2]
            result = processor.delete_person(name)
            print(json.dumps(result))
            
        elif action == 'clear-all':
            result = processor.clear_all_data()
            print(json.dumps(result))
            
        elif action == 'extract' and len(sys.argv) >= 3:
            image_path = sys.argv[2]
            person_name = sys.argv[3] if len(sys.argv) >= 4 else "unknown"
            face_data = processor.extract_face_features(image_path)
            if face_data:
                result = {
                    "embedding": face_data['embedding'],
                    "faceInfo": {"det_score": face_data['det_score']},
                    "qualityScore": face_data['quality_score'],
                    "faceBbox": face_data['bbox']
                }
                print(json.dumps(result))
            else:
                print(json.dumps({"error": "No face detected"}))
                
        else:
            print(json.dumps({
                "error": "Invalid action", 
                "available_actions": ["recognize", "register", "search", "status", "list-people", "delete-person", "clear-all", "extract"]
            }))
            
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()