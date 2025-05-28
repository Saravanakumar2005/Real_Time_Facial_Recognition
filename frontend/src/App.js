import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import RAGChatBox from './RAGChatBox';
import './App.css';

const API_BASE_URL = 'http://localhost:3000/api';

function App() {
  const [currentMode, setCurrentMode] = useState('recognition');
  const [registrationName, setRegistrationName] = useState('');
  const [stats, setStats] = useState({});
  const [registeredPeople, setRegisteredPeople] = useState([]);
  const [isRegistering, setIsRegistering] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [statusType, setStatusType] = useState('info');
  const [ragSystemStatus, setRagSystemStatus] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const wsRef = useRef(null);

  useEffect(() => {
    initializeCamera();
    initializeWebSocket();
    fetchStats();
    fetchRegisteredPeople();
    checkRAGSystem();

    const statsInterval = setInterval(fetchStats, 5000);
    const ragStatusInterval = setInterval(checkRAGSystem, 10000);
    const recognitionInterval = setInterval(() => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN && currentMode === 'recognition' && !isRegistering) {
        captureAndSendWebSocket();
      }
    }, 1000);

    return () => {
      clearInterval(statsInterval);
      clearInterval(ragStatusInterval);
      clearInterval(recognitionInterval);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const initializeWebSocket = () => {
    try {
      wsRef.current = new WebSocket('ws://localhost:3002');
      console.log('Attempting WebSocket connection to ws://localhost:3002');

      wsRef.current.onopen = () => {
        setWsConnected(true);
        showMessage('Connected to WebSocket', 'success');
        console.log('WebSocket connected');
      };

      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'recognition_data') {
            if (data.data.recognized) {
              const person = data.data.person;
              const confidence = (person.confidence * 100).toFixed(2);
              showMessage(`Recognized: ${person.name} (${confidence}% confidence)`, 'success');
            } else {
              showMessage('Person not recognized', 'warning');
            }
          } else if (data.type === 'error') {
            showMessage(`WebSocket error: ${data.message}`, 'error');
          }
        } catch (error) {
          console.error('WebSocket message error:', error);
          showMessage('Error processing WebSocket message', 'error');
        }
      };

      wsRef.current.onclose = () => {
        setWsConnected(false);
        showMessage('WebSocket disconnected. Retrying...', 'error');
        console.log('WebSocket disconnected');
        setTimeout(() => {
          if (!wsRef.current || wsRef.current.readyState === WebSocket.CLOSED) {
            initializeWebSocket();
          }
        }, 5000); // Increased retry delay to 5s
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setWsConnected(false);
        showMessage('WebSocket connection failed. Ensure server is running.', 'error');
      };
    } catch (error) {
      console.error('WebSocket initialization error:', error);
      showMessage('Failed to initialize WebSocket.', 'error');
    }
  };

  const captureAndSendWebSocket = () => {
    try {
      const canvas = canvasRef.current;
      const video = videoRef.current;

      if (!canvas || !video || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        return;
      }

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0);

      const imageData = canvas.toDataURL('image/jpeg', 0.8);
      wsRef.current.send(JSON.stringify({
        type: 'recognize',
        image_data: imageData
      }));
    } catch (error) {
      console.error('WebSocket capture failed:', error);
    }
  };

  const checkRAGSystem = async (retries = 3, delay = 2000) => {
    for (let i = 0; i < retries; i++) {
      try {
        const response = await axios.get(`${API_BASE_URL}/status`);
        setRagSystemStatus(response.data.system);
        return;
      } catch (error) {
        console.error('RAG system check failed:', error);
        if (i < retries - 1) {
          await new Promise(resolve => setTimeout(resolve, delay));
          continue;
        }
        setRagSystemStatus(null);
        showMessage('RAG system offline. Check server.', 'error');
      }
    }
  };

  const initializeCamera = async (retries = 3, delay = 2000) => {
    for (let i = 0; i < retries; i++) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { width: 640, height: 480 } 
        });
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          console.log('Camera initialized successfully');
        }
        return;
      } catch (error) {
        console.error(`Camera initialization attempt ${i + 1} failed:`, error);
        if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
          showMessage('Camera in use or unavailable. Please close other apps using the camera.', 'error');
        } else if (error.name === 'NotAllowedError') {
          showMessage('Camera access denied. Please grant permission in your browser.', 'error');
        } else {
          showMessage('Camera initialization failed.', 'error');
        }
        if (i < retries - 1) {
          await new Promise(resolve => setTimeout(resolve, delay));
          continue;
        }
      }
    }
    console.error('All camera initialization attempts failed');
  };

  const fetchStats = async (retries = 3, delay = 2000) => {
    for (let i = 0; i < retries; i++) {
      try {
        const response = await axios.get(`${API_BASE_URL}/stats`);
        setStats(response.data);
        return;
      } catch (error) {
        console.error('Failed to fetch stats:', error);
        if (i < retries - 1) {
          await new Promise(resolve => setTimeout(resolve, delay));
          continue;
        }
        showMessage('Unable to fetch stats. Check server.', 'error');
      }
    }
  };

  const fetchRegisteredPeople = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/registered-people`);
      if (response.data.success) {
        setRegisteredPeople(response.data.people);
      }
    } catch (error) {
      console.error('Failed to fetch registered people:', error);
      showMessage('Failed to fetch registered people', 'error');
    }
  };

  const setMode = async (mode) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/set-mode`, {
        mode,
        name: mode === 'registration' ? registrationName : ''
      });
      
      if (response.data.success) {
        setCurrentMode(mode);
        showMessage(`Mode switched to ${mode}`, 'info');
      }
    } catch (error) {
      showMessage('Failed to switch mode', 'error');
    }
  };

  const handleRegistration = async () => {
    if (!registrationName.trim()) {
      showMessage('Please enter a person name', 'error');
      return;
    }

    if (isRegistering) {
      setIsRegistering(false);
      await setMode('recognition');
      return;
    }

    setIsRegistering(true);
    await setMode('registration');
    
    setTimeout(async () => {
      if (isRegistering) {
        await captureAndRegister();
      }
    }, 3000);
  };

  const captureAndRegister = async () => {
    try {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      
      if (!canvas || !video) {
        throw new Error('Canvas or video not available');
      }
      
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      if (video.videoWidth === 0 || video.videoHeight === 0) {
        throw new Error('Video dimensions are invalid');
      }

      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0);
      
      canvas.toBlob(async (blob) => {
        if (!(blob instanceof Blob)) {
          console.error('toBlob did not return a valid Blob:', blob);
          showMessage('Failed to capture image: Invalid blob', 'error');
          setIsRegistering(false);
          setIsLoading(false);
          await setMode('recognition');
          return;
        }
        
        const formData = new FormData();
        formData.append('image', blob, 'capture.jpg');
        formData.append('name', registrationName);
        
        try {
          setIsLoading(true);
          const response = await axios.post(`${API_BASE_URL}/register`, formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
          });
          
          if (response.data.success) {
            showMessage(response.data.message || 'Registration successful!', 'success');
            fetchStats();
            fetchRegisteredPeople();
            setRegistrationName('');
          } else {
            showMessage(response.data.error || 'Registration failed', 'error');
          }
        } catch (error) {
          showMessage('Registration failed', 'error');
          console.error('Registration error:', error);
        } finally {
          setIsRegistering(false);
          setIsLoading(false);
          await setMode('recognition');
        }
      }, 'image/jpeg', 0.8);
      
    } catch (error) {
      console.error('Capture failed:', error);
      showMessage('Failed to capture image', 'error');
      setIsRegistering(false);
      setIsLoading(false);
    }
  };

  const captureAndRecognize = async () => {
    try {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      
      if (!canvas || !video) {
        throw new Error('Canvas or video not available');
      }
      
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      if (video.videoWidth === 0 || video.videoHeight === 0) {
        throw new Error('Video dimensions are invalid');
      }

      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0);
      
      canvas.toBlob(async (blob) => {
        if (!(blob instanceof Blob)) {
          console.error('toBlob did not return a valid Blob:', blob);
          showMessage('Failed to capture image: Invalid blob', 'error');
          setIsLoading(false);
          return;
        }
        
        const formData = new FormData();
        formData.append('image', blob, 'capture.jpg');
        
        try {
          setIsLoading(true);
          showMessage('Recognizing faces...', 'info');
          
          const response = await axios.post(`${API_BASE_URL}/recognize`, formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
          });
          
          if (response.data.recognized) {
            const person = response.data.person;
            const confidence = (person.confidence * 100).toFixed(1);
            showMessage(`Recognized: ${person.name} (${confidence}% confidence)`, 'success');
          } else {
            showMessage('Person not recognized', 'warning');
          }
          
        } catch (error) {
          showMessage('Recognition failed', 'error');
          console.error('Recognition error:', error);
        } finally {
          setIsLoading(false);
        }
      }, 'image/jpeg', 0.8);
      
    } catch (error) {
      console.error('Capture and recognize failed:', error);
      showMessage('Failed to capture and recognize', 'error');
      setIsLoading(false);
    }
  };

  const deletePerson = async (name) => {
    if (!window.confirm(`Are you sure you want to delete all faces for ${name}?`)) {
      return;
    }
    
    try {
      setIsLoading(true);
      const response = await axios.delete(`${API_BASE_URL}/person/${encodeURIComponent(name)}`);
      
      if (response.data.success) {
        showMessage(`Successfully deleted ${name}`, 'success');
        fetchStats();
        fetchRegisteredPeople();
      } else {
        showMessage(response.data.error || 'Failed to delete person', 'error');
      }
    } catch (error) {
      showMessage('Failed to delete person', 'error');
      console.error('Delete error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const clearAllData = async () => {
    if (!window.confirm('Are you sure you want to delete ALL registered faces? This action cannot be undone.')) {
      return;
    }
    
    try {
      setIsLoading(true);
      const response = await axios.delete(`${API_BASE_URL}/clear-all`);
      
      if (response.data.success) {
        showMessage('All data cleared successfully', 'success');
        fetchStats();
        fetchRegisteredPeople();
      } else {
        showMessage(response.data.error || 'Failed to clear data', 'error');
      }
    } catch (error) {
      showMessage('Failed to clear data', 'error');
      console.error('Clear all error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const showMessage = (message, type = 'info') => {
    setStatusMessage(message);
    setStatusType(type);
    setTimeout(() => {
      setStatusMessage('');
    }, 5000);
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <header className="container mx-auto p-6">
        <h1 className="text-3xl font-bold text-center mb-6">Face Recognition System</h1>
        
        <div className="system-status flex justify-center gap-4 mb-4">
          <span className={`status-indicator ${wsConnected ? 'connected' : 'disconnected'}`}>
            WebSocket: {wsConnected ? '🟢 Connected' : '🔴 Disconnected'}
          </span>
          {ragSystemStatus && (
            <span className="status-indicator connected">
              RAG System: 🟢 Online
            </span>
          )}
        </div>

        {statusMessage && (
          <div className={`status-message ${statusType} p-4 rounded-lg mb-4 text-center text-white`}>
            {statusMessage}
          </div>
        )}

        {isLoading && (
          <div className="loading-indicator flex justify-center items-center gap-2 mb-4">
            <div className="spinner animate-spin h-6 w-6 border-4 border-blue-500 border-t-transparent rounded-full"></div>
            <span>Processing...</span>
          </div>
        )}

        <div className="video-container mx-auto mb-6 relative">
          <video 
            ref={videoRef} 
            autoPlay 
            playsInline 
            muted
            className={`rounded-lg shadow-lg ${currentMode === 'registration' ? 'border-4 border-red-500' : 'border-4 border-green-500'}`}
          />
          <canvas ref={canvasRef} className="hidden" />
          
          {isRegistering && (
            <div className="registration-overlay absolute inset-0 flex items-center justify-center bg-black bg-opacity-50 rounded-lg">
              <div className="countdown text-white text-xl font-semibold">
                Registering {registrationName}...
              </div>
            </div>
          )}
        </div>

        <div className="controls bg-white p-6 rounded-lg shadow-md mb-6">
          <h3 className="text-xl font-semibold mb-4">Current Mode: {currentMode}</h3>
          
          {currentMode === 'recognition' && (
            <div>
              <button 
                onClick={captureAndRecognize}
                disabled={isLoading}
                className="btn bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded disabled:opacity-50"
              >
                📸 Recognize Face
              </button>
            </div>
          )}
          
          <div className="registration-section flex gap-4 mt-4">
            <input
              type="text"
              placeholder="Enter person name"
              value={registrationName}
              onChange={(e) => setRegistrationName(e.target.value)}
              disabled={isRegistering || isLoading}
              className="name-input border border-gray-300 rounded px-3 py-2 flex-grow focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button 
              onClick={handleRegistration}
              disabled={isLoading}
              className={`btn ${isRegistering ? 'bg-red-500 hover:bg-red-600' : 'bg-green-500 hover:bg-green-600'} text-white px-4 py-2 rounded disabled:opacity-50`}
            >
              {isRegistering ? '⏹️ Stop Registration' : '📝 Register New Face'}
            </button>
          </div>
        </div>

        <div className="stats-section bg-white p-6 rounded-lg shadow-md mb-6">
          <h3 className="text-xl font-semibold mb-4">System Statistics</h3>
          <div className="stats-grid grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="stat-item p-4 bg-gray-50 rounded-lg text-center">
              <span className="stat-label block text-gray-600">Total Registered:</span>
              <span className="stat-value text-2xl font-bold">{stats.total_faces || 0}</span>
            </div>
            <div className="stat-item p-4 bg-gray-50 rounded-lg text-center">
              <span className="stat-label block text-gray-600">Total Images:</span>
              <span className="stat-value text-2xl font-bold">{stats.total_faces || 0}</span>
            </div>
            <div className="stat-item p-4 bg-gray-50 rounded-lg text-center">
              <span className="stat-label block text-gray-600">Recognition Count:</span>
              <span className="stat-value text-2xl font-bold">{stats.total_recognitions || 0}</span>
            </div>
          </div>
        </div>

        <div className="people-section bg-white p-6 rounded-lg shadow-md">
          <div className="section-header flex justify-between items-center mb-4">
            <h3 className="text-xl font-semibold">Registered People ({registeredPeople.length})</h3>
            {registeredPeople.length > 0 && (
              <button 
                onClick={clearAllData}
                disabled={isLoading}
                className="btn bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded disabled:opacity-50"
              >
                🗑️ Clear All
              </button>
            )}
          </div>
          
          {registeredPeople.length === 0 ? (
            <p className="no-data text-gray-500">No people registered yet</p>
          ) : (
            <div className="people-list space-y-2">
              {registeredPeople.map((person, index) => (
                <div key={index} className="person-item flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                  <div className="person-info">
                    <span className="person-name font-medium">{person.name}</span>
                    <span className="person-count text-gray-500 ml-2">
                      {person.image_count} image{person.image_count !== 1 ? 's' : ''}
                    </span>
                  </div>
                  <button
                    onClick={() => deletePerson(person.name)}
                    disabled={isLoading}
                    className="btn bg-red-500 hover:bg-red-600 text-white px-3 py-1 rounded disabled:opacity-50"
                  >
                    🗑️ Delete
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="rag-section">
          <RAGChatBox />
        </div>
      </header>
    </div>
  );
}

export default App;