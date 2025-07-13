import os
import json
import logging
from typing import List, Dict, Any
import mysql.connector
import numpy as np
import cv2
import base64
from flask import Flask, request, jsonify, render_template
import insightface
from sklearn.metrics.pairwise import cosine_similarity

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from api_key import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

class FaceRecognitionRAG:
    def __init__(self, openai_api_key: str):
        """Initialize the RAG system for face recognition database queries"""
        self.openai_api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
    
        self.db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': password,
            'database': 'facial_recognize',
            'autocommit': True,
            'auth_plugin': 'mysql_native_password'
        }
    
        try:
            self.embeddings = OpenAIEmbeddings()
            self.vectorstore = None
            self.qa_chain = None
            self.conversation_chain = None
            self.memory = ConversationBufferWindowMemory(
                k=5,
                memory_key="chat_history",
                return_messages=True
            )
            
       
            if self.test_db_connection():
                self.setup_rag_system()
            else:
                logger.error("Failed to connect to database")
        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}")
    
    def test_db_connection(self):
        try:
            connection = mysql.connector.connect(**self.db_config)
            connection.ping(reconnect=True)
            connection.close()
            logger.info("✅ Database connection test successful")
            return True
        except mysql.connector.Error as e:
            logger.error(f"❌ Database connection test failed: {e}")
            return False
        
    def get_db_connection(self):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                connection = mysql.connector.connect(**self.db_config)
                connection.ping(reconnect=True)
                return connection
            except mysql.connector.Error as e:
                logger.error(f"Database connection attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    return None
        return None
    
    def fetch_database_content(self) -> Dict[str, Any]:
        try:
            connection = self.get_db_connection()
            if not connection:
                logger.error("No database connection available")
                return {}
            
            cursor = connection.cursor()
            cursor.execute('''
                SELECT id, name, registration_timestamp, quality_score, face_bbox
                FROM registered_faces 
                ORDER BY registration_timestamp ASC
            ''')
            faces_data = cursor.fetchall()

            cursor.execute('''
                SELECT id, person_name, confidence, similarity_score, 
                       recognition_timestamp, status, face_quality
                FROM recognition_log 
                ORDER BY recognition_timestamp DESC
                LIMIT 100
            ''')
            recognition_logs = cursor.fetchall()
            cursor.execute('SELECT COUNT(*) FROM registered_faces')
            total_faces = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(DISTINCT name) FROM registered_faces')
            unique_people = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM recognition_log WHERE status = "SUCCESS"')
            successful_recognitions = cursor.fetchone()[0]
            
            connection.close()

            database_content = {
                'faces': faces_data,
                'recognition_logs': recognition_logs,
                'statistics': {
                    'total_faces': total_faces,
                    'unique_people': unique_people,
                    'successful_recognitions': successful_recognitions
                }
            }
            
            logger.info(f"✅ Fetched database content: {total_faces} faces, {len(recognition_logs)} logs")
            return database_content
            
        except Exception as e:
            logger.error(f"Error fetching database content: {e}")
            return {}
    
    def create_documents_from_data(self, database_content: Dict[str, Any]) -> List[Document]:
        documents = []
        
        try:
            if 'faces' in database_content:
                for i, (face_id, name, reg_timestamp, quality_score, face_bbox) in enumerate(database_content['faces']):
                    content = f"""
                    Face Registration Record:
                    - Face ID: {face_id}
                    - Person Name: {name}
                    - Registration Date: {reg_timestamp}
                    - Registration Time: {reg_timestamp.strftime('%Y-%m-%d %H:%M:%S') if reg_timestamp else 'Unknown'}
                    - Quality Score: {quality_score}
                    - Registration Order: {i + 1} (This is the {self.ordinal(i + 1)} registered person)
                    - Face Bounding Box: {face_bbox}
                    
                    Summary: {name} was registered as the {self.ordinal(i + 1)} person in the face recognition system on {reg_timestamp.strftime('%B %d, %Y at %H:%M:%S') if reg_timestamp else 'unknown date'} with a quality score of {quality_score}.
                    """
                    
                    documents.append(Document(
                        page_content=content,
                        metadata={
                            'type': 'face_registration',
                            'face_id': face_id,
                            'person_name': name,
                            'registration_timestamp': str(reg_timestamp),
                            'registration_order': i + 1,
                            'quality_score': quality_score
                        }
                    ))
            
            if 'recognition_logs' in database_content:
                for log_id, person_name, confidence, similarity_score, rec_timestamp, status, face_quality in database_content['recognition_logs']:
                    content = f"""
                    Face Recognition Log:
                    - Log ID: {log_id}
                    - Recognized Person: {person_name}
                    - Recognition Confidence: {confidence}%
                    - Similarity Score: {similarity_score}
                    - Recognition Time: {rec_timestamp.strftime('%Y-%m-%d %H:%M:%S') if rec_timestamp else 'Unknown'}
                    - Recognition Status: {status}
                    - Face Quality: {face_quality}
                    
                    Summary: {person_name} was recognized on {rec_timestamp.strftime('%B %d, %Y at %H:%M:%S') if rec_timestamp else 'unknown date'} with {confidence}% confidence and {status} status.
                    """
                    
                    documents.append(Document(
                        page_content=content,
                        metadata={
                            'type': 'recognition_log',
                            'log_id': log_id,
                            'person_name': person_name,
                            'confidence': confidence,
                            'recognition_timestamp': str(rec_timestamp),
                            'status': status
                        }
                    ))
            if 'statistics' in database_content:
                stats = database_content['statistics']
                content = f"""
                Face Recognition System Statistics:
                - Total Registered Faces: {stats['total_faces']}
                - Total Unique People: {stats['unique_people']}
                - Successful Recognitions: {stats['successful_recognitions']}
                - Success Rate: {(stats['successful_recognitions'] / max(stats['total_faces'], 1)) * 100:.2f}%
                
                Summary: The face recognition system currently has {stats['total_faces']} registered faces representing {stats['unique_people']} unique people, with {stats['successful_recognitions']} successful recognitions recorded.
                """
                
                documents.append(Document(
                    page_content=content,
                    metadata={
                        'type': 'system_statistics',
                        'total_faces': stats['total_faces'],
                        'unique_people': stats['unique_people'],
                        'successful_recognitions': stats['successful_recognitions']
                    }
                ))
            
            logger.info(f"Created {len(documents)} documents from database content")
            return documents
            
        except Exception as e:
            logger.error(f"Error creating documents: {e}")
            return []
    
    def ordinal(self, n: int) -> str:
        if 10 <= n % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        return f"{n}{suffix}"
    
    def setup_rag_system(self):
        try:
            database_content = self.fetch_database_content()
            if not database_content:
                logger.warning("No database content found")
                return
            
            documents = self.create_documents_from_data(database_content)
            if not documents:
                logger.warning("No documents created")
                return
            text_splitter = CharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separator="\n"
            )
            split_docs = text_splitter.split_documents(documents)
            
            self.vectorstore = FAISS.from_documents(
                split_docs,
                self.embeddings
            )
            
            # Enhanced prompt template with "can't answer" handling
            custom_prompt = PromptTemplate(
                template="""You are a face recognition system assistant. Use the provided context to answer questions about registered faces, recognition logs, and system statistics.

Context: {context}
Question: {question}

Instructions:
- If the context contains relevant information to answer the question, provide a clear and specific answer
- For "who" questions: return the person's name
- For "when" questions: return the date/time
- For "how many" questions: return the number
- For ordinal questions (1st, 2nd, 3rd person): return the name
- If the context does NOT contain enough information to answer the question, respond with: "Sorry, I can't answer this question based on the available face recognition data."
- If the question is not related to face recognition, registered faces, or recognition logs, respond with: "Sorry, I can only answer questions related to the face recognition system data."

Answer:""",
                input_variables=["context", "question"]
            )

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=OpenAI(temperature=0.1, max_tokens=500),
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5}
                ),
                chain_type_kwargs={"prompt": custom_prompt},
                return_source_documents=True
            )
            
            logger.info("✅ RAG system setup completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Error setting up RAG system: {e}")
    
    def refresh_database_content(self):
        """Refresh the vector store with latest database content"""
        try:
            logger.info("Refreshing database content...")
            self.setup_rag_system()
            logger.info("✅ Database content refreshed")
            return True
        except Exception as e:
            logger.error(f"❌ Error refreshing database content: {e}")
            return False
    
    def query(self, question: str, use_conversation: bool = True) -> Dict[str, Any]:
        """Query the RAG system with enhanced error handling"""
        try:
            if not self.vectorstore:
                return {
                    'answer': 'RAG system not initialized. Please check database connection and OpenAI API key.',
                    'source_documents': [],
                    'error': 'System not initialized'
                }
     
            result = self.qa_chain.invoke({"query": question})
            
            # Check if the answer indicates inability to respond
            answer = result['result'].strip()
            if not answer or len(answer) < 5:
                answer = "Sorry, I can't answer this question based on the available face recognition data."
            
            source_docs = []
            if 'source_documents' in result:
                for doc in result['source_documents']:
                    source_docs.append({
                        'content': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        'metadata': doc.metadata
                    })
            
            return {
                'answer': answer,
                'source_documents': source_docs
            }
            
        except Exception as e:
            logger.error(f"Error querying RAG system: {e}")
            return {
                'answer': 'Sorry, I encountered an error while processing your question. Please try again.',
                'source_documents': [],
                'error': str(e)
            }

class FaceRecognitionSystem:
    def __init__(self):
        
        self.db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': password,
            'database': 'facial_recognize',
            'autocommit': True,
            'auth_plugin': 'mysql_native_password'
        }
        
      
        try:
            self.app = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("✅ InsightFace model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading InsightFace model: {e}")
            self.app = None

        self.similarity_threshold = 0.4
        self.init_database()
    
    def test_db_connection(self):
        try:
            connection = mysql.connector.connect(**self.db_config)
            connection.ping(reconnect=True)
            connection.close()
            return True
        except mysql.connector.Error as e:
            logger.error(f"Database connection test failed: {e}")
            return False
        
    def get_db_connection(self):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                connection = mysql.connector.connect(**self.db_config)
                connection.ping(reconnect=True)
                return connection
            except mysql.connector.Error as e:
                logger.error(f"Database connection attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    return None
        return None
    
    def init_database(self):
        try:
            connection = self.get_db_connection()
            if not connection:
                logger.error("❌ Cannot initialize database - no connection")
                return
            
            cursor = connection.cursor()
            cursor.execute("CREATE DATABASE IF NOT EXISTS facial_recognize")
            cursor.execute("USE facial_recognize")

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS registered_faces (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    face_encoding LONGTEXT NOT NULL,
                    registration_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    quality_score FLOAT,
                    face_bbox VARCHAR(255)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recognition_log (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    person_name VARCHAR(255),
                    confidence FLOAT,
                    similarity_score FLOAT,
                    recognition_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status VARCHAR(50),
                    face_quality FLOAT
                )
            ''')
            
            connection.commit()
            connection.close()
            logger.info("✅ Database initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing database: {e}")
    
    def extract_face_encoding(self, image_array):
        try:
            if self.app is None:
                logger.error("InsightFace model not loaded")
                return None, None, None
                
            faces = self.app.get(image_array)
            if len(faces) == 0:
                return None, None, None
            
            face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            
            face_area = (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
            quality_score = min(1.0, face_area / (100 * 100))  
            return face.embedding, face.bbox, quality_score
            
        except Exception as e:
            logger.error(f"Error extracting face encoding: {e}")
            return None, None, None
    
    def draw_face_box(self, image_array, bbox, name="", confidence=0):
        try:
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            cv2.rectangle(image_array, (x1, y1), (x2, y2), (0, 255, 0), 3)

            if name:
                label = f"{name} ({confidence:.1f}% confidence)" if confidence > 0 else name
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = x1
                text_y = y1 - 10
                
                cv2.rectangle(
                    image_array,
                    (text_x, text_y - text_size[1] - 10),
                    (text_x + text_size[0] + 20, text_y + 10),
                    (0, 255, 0),
                    -1
                )
                
                cv2.putText(
                    image_array,
                    label,
                    (text_x + 10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA
                )
            
            return image_array
        except Exception as e:
            logger.error(f"Error drawing face box: {e}")
            return image_array
    
    def register_face(self, name: str, image_data: str):
        try:
            if not name or not image_data:
                return {'success': False, 'message': 'Name and image are required'}
            try:
                image_bytes = base64.b64decode(image_data.split(',')[1])
                image_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            except Exception as e:
                logger.error(f"Error decoding image: {e}")
                return {'success': False, 'message': 'Invalid image format'}
            
            face_encoding, face_bbox, quality_score = self.extract_face_encoding(image_array)
            
            if face_encoding is None:
                return {'success': False, 'message': 'No face detected in the image'}

            if face_bbox is not None:
                image_with_box = self.draw_face_box(image_array.copy(), face_bbox, name)
                _, buffer = cv2.imencode('.jpg', image_with_box)
                processed_image = base64.b64encode(buffer).decode('utf-8')
            else:
                processed_image = None

            encoding_str = json.dumps(face_encoding.tolist())
            bbox_str = json.dumps(face_bbox.tolist()) if face_bbox is not None else None
           
            connection = self.get_db_connection()
            if not connection:
                return {'success': False, 'message': 'Database connection failed'}
            
            cursor = connection.cursor()
            cursor.execute('''
                INSERT INTO registered_faces (name, face_encoding, quality_score, face_bbox)
                VALUES (%s, %s, %s, %s)
            ''', (name, encoding_str, quality_score, bbox_str))
            
            connection.commit()
            face_id = cursor.lastrowid
            connection.close()
            
            logger.info(f"✅ Face registered: {name} (ID: {face_id})")
            return {
                'success': True, 
                'message': f'Face registered successfully for {name}',
                'face_id': face_id,
                'quality_score': quality_score,
                'processed_image': processed_image,
                'bbox': face_bbox.tolist() if face_bbox is not None else None
            }
            
        except Exception as e:
            logger.error(f"❌ Error registering face: {e}")
            return {'success': False, 'message': f'Registration failed: {str(e)}'}
    
    def recognize_face(self, image_data: str):
        try:
            if not image_data:
                return {'success': False, 'message': 'Image is required'}
            try:
                image_bytes = base64.b64decode(image_data.split(',')[1])
                image_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            except Exception as e:
                logger.error(f"Error decoding image: {e}")
                return {'success': False, 'message': 'Invalid image format'}

            connection = self.get_db_connection()
            if not connection:
                return {'success': False, 'message': 'Database connection failed'}
            
            cursor = connection.cursor()
            cursor.execute('SELECT id, name, face_encoding FROM registered_faces')
            registered_faces = cursor.fetchall()
            
            if not registered_faces:
                connection.close()
                return {'success': False, 'message': 'No registered faces found'}
            faces = self.app.get(image_array)
            if len(faces) == 0:
                return {'success': False, 'message': 'No faces detected in the image'}
        
            processed_image = image_array.copy()
            recognized_faces = []
            
            for face in faces:
                face_encoding = face.embedding
                face_bbox = face.bbox
                quality_score = min(1.0, (face_bbox[2] - face_bbox[0]) * (face_bbox[3] - face_bbox[1]) / (100 * 100))
                
                best_match = None
                best_similarity = 0
                
                for face_id, name, encoding_str in registered_faces:
                    try:
                        stored_encoding = np.array(json.loads(encoding_str))
                        similarity = cosine_similarity([face_encoding], [stored_encoding])[0][0]
                        
                        if similarity > best_similarity and similarity > self.similarity_threshold:
                            best_similarity = similarity
                            best_match = {'id': face_id, 'name': name, 'similarity': similarity}
                    except Exception as e:
                        logger.error(f"Error comparing encodings: {e}")
                        continue
          
                if best_match:
                    person_name = best_match['name']
                    confidence = best_similarity * 100
                    status = "SUCCESS"
                    processed_image = self.draw_face_box(processed_image, face_bbox, person_name, confidence)
                    
                    cursor.execute('''
                        INSERT INTO recognition_log (person_name, confidence, similarity_score, status, face_quality)
                        VALUES (%s, %s, %s, %s, %s)
                    ''', (person_name, confidence, best_similarity, status, quality_score))
                    
                    recognized_faces.append({
                        'success': True,
                        'person_name': person_name,
                        'confidence': confidence,
                        'similarity_score': best_similarity,
                        'face_id': best_match['id'],
                        'bbox': face_bbox.tolist()
                    })
                    
                    logger.info(f"✅ Face recognized: {person_name} ({confidence:.2f}% confidence)")
                else:
                    processed_image = self.draw_face_box(processed_image, face_bbox, "Unknown", 0)
                    
                    cursor.execute('''
                        INSERT INTO recognition_log (person_name, confidence, similarity_score, status, face_quality)
                        VALUES (%s, %s, %s, %s, %s)
                    ''', ("Unknown", 0, 0, "UNKNOWN", quality_score))
                    
                    recognized_faces.append({
                        'success': False,
                        'message': 'Face not recognized',
                        'confidence': 0,
                        'bbox': face_bbox.tolist()
                    })
            

            _, buffer = cv2.imencode('.jpg', processed_image)
            processed_image_base64 = base64.b64encode(buffer).decode('utf-8')
    
            connection.commit()
            connection.close()
            
            return {
                'success': True,
                'faces': recognized_faces,
                'processed_image': processed_image_base64,
                'total_faces': len(faces)
            }
            
        except Exception as e:
            logger.error(f"❌ Error recognizing faces: {e}")
            return {'success': False, 'message': f'Recognition failed: {str(e)}'}
    
    def get_registered_faces(self):
        try:
            connection = self.get_db_connection()
            if not connection:
                return []
            
            cursor = connection.cursor()
            cursor.execute('''
                SELECT id, name, registration_timestamp, quality_score 
                FROM registered_faces 
                ORDER BY registration_timestamp DESC
            ''')
            faces = cursor.fetchall()
            connection.close()
            
            return [
                {
                    'id': face_id,
                    'name': name,
                    'registration_date': timestamp.strftime('%Y-%m-%d %H:%M:%S') if timestamp else None,
                    'quality_score': quality_score
                }
                for face_id, name, timestamp, quality_score in faces
            ]
            
        except Exception as e:
            logger.error(f"Error getting registered faces: {e}")
            return []

face_system = FaceRecognitionSystem()
rag_system = None

def initialize_rag_system(openai_api_key: str):
    global rag_system
    try:
        rag_system = FaceRecognitionRAG(openai_api_key)
        logger.info("✅ RAG system initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG system: {e}")
        return False

@app.route('/')
def index():
    return render_template('templates/index.html')

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        name = data.get('name')
        image = data.get('image')
        
        result = face_system.register_face(name, image)
        
        if result.get('success') and rag_system:
            rag_system.refresh_database_content()
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Registration endpoint error: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        data = request.get_json()
        image = data.get('image')
        
        result = face_system.recognize_face(image)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Recognition endpoint error: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/chat', methods=['POST'])
def chat():
    try:
        if not rag_system:
            return jsonify({
                'answer': 'RAG system not initialized. Please check your OpenAI API key and database connection.',
                'error': 'RAG system not available'
            })
        
        data = request.get_json()
        question = data.get('question', '')
        
        if not question.strip():
            return jsonify({
                'answer': 'Please ask a question about the face recognition system.',
                'error': 'Empty question'
            })
        
        result = rag_system.query(question)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return jsonify({
            'answer': 'Sorry, I encountered an error while processing your question. Please try again.',
            'error': str(e)
        })



@app.route('/refresh', methods=['POST'])
def refresh():
    """Refresh RAG system data"""
    try:
        if rag_system:
            success = rag_system.refresh_database_content()
            return jsonify({'success': success})
        else:
            return jsonify({'success': False, 'message': 'RAG system not initialized'})
    except Exception as e:
        logger.error(f"Refresh endpoint error: {e}")
        return jsonify({'success': False, 'message': str(e)})
    
@app.route('/status')
def status():
    try:
        registered_faces = face_system.get_registered_faces()
        return jsonify({
            'registered_faces': len(registered_faces),
            'faces': registered_faces
        })
    except Exception as e:
        logger.error(f"Status endpoint error: {e}")
        return jsonify({'registered_faces': 0, 'faces': []})
    
if __name__ == '__main__':
    print("🚀 Starting Face Recognition System...")
    print("=" * 50)
    
    if face_system.test_db_connection():
        print("✅ Database connection: OK")
    else:
        print("❌ Database connection: FAILED")
        print("Please check your MySQL server and database configuration")

    if initialize_rag_system(openai_api_key):
        print("✅ RAG system: OK")
    else:
        print("❌ RAG system: FAILED")
        print("Please check your OpenAI API key and database connection")
    
    print("=" * 50)
    print("🌐 Server starting on http://localhost:5000")
    print("📱 Open your browser and navigate to the URL above")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)