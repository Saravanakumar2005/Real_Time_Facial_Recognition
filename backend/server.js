const express = require('express');
const multer = require('multer');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const cors = require('cors');
const WebSocket = require('ws');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors({ origin: 'http://localhost:3001' }));
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const uploadDir = './Uploads';
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir);
    }
    cb(null, uploadDir);
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + '-' + file.originalname);
  }
});

const upload = multer({ 
  storage: storage,
  limits: { fileSize: 10 * 1024 * 1024 },
  fileFilter: function (req, file, cb) {
    const allowedTypes = /jpeg|jpg|png|gif/;
    const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
    const mimetype = allowedTypes.test(file.mimetype);
    
    if (mimetype && extname) {
      return cb(null, true);
    } else {
      cb(new Error('Only image files are allowed'));
    }
  }
});

const PYTHON_SCRIPT = path.join(__dirname, 'face_recognition_backend.py');

function executePythonRAG(action, args = []) {
  return new Promise((resolve, reject) => {
    if (!fs.existsSync(PYTHON_SCRIPT)) {
      console.error(`Python script not found: ${PYTHON_SCRIPT}`);
      return reject(new Error('Python script not found'));
    }
    const pythonArgs = [PYTHON_SCRIPT, action, ...args];
    console.log(`Executing Python script: python ${pythonArgs.join(' ')}`);
    const pythonProcess = spawn('python', pythonArgs);
    
    let stdout = '';
    let stderr = '';
    
    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString();
    });
    
    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString();
    });
    
    pythonProcess.on('close', (code) => {
      console.log(`Python process exited with code ${code}`);
      if (code === 0) {
        try {
          const result = JSON.parse(stdout);
          resolve(result);
        } catch (error) {
          console.error('Failed to parse Python output:', stdout);
          resolve({ error: 'Failed to parse response', raw: stdout });
        }
      } else {
        console.error('Python process failed:', stderr);
        reject(new Error(`Python process failed: ${stderr}`));
      }
    });
    
    pythonProcess.on('error', (error) => {
      console.error('Python process error:', error);
      reject(error);
    });
  });
}

try {
  const wss = new WebSocket.Server({ port: 3002 }, () => {
    console.log('WebSocket server started on ws://localhost:3002');
  });

  wss.on('error', (error) => {
    console.error('WebSocket server error:', error);
  });

  wss.on('connection', (ws) => {
    console.log('WebSocket client connected');
    
    ws.on('message', async (message) => {
      try {
        const data = JSON.parse(message);
        if (data.type === 'recognize' && data.image_data) {
          const uploadDir = './Uploads';
          if (!fs.existsSync(uploadDir)) {
            fs.mkdirSync(uploadDir);
          }
          const buffer = Buffer.from(data.image_data.split(',')[1], 'base64');
          const tempPath = path.join(uploadDir, `temp_${Date.now()}.jpg`);
          fs.writeFileSync(tempPath, buffer);
          
          const result = await executePythonRAG('recognize', [tempPath]);
          fs.unlinkSync(tempPath);
          
          ws.send(JSON.stringify({
            type: 'recognition_data',
            data: result
          }));
        }
      } catch (error) {
        console.error('WebSocket message error:', error);
        ws.send(JSON.stringify({ type: 'error', message: error.message }));
      }
    });
    
    ws.on('close', () => {
      console.log('WebSocket client disconnected');
    });
  });
} catch (error) {
  console.error('Failed to start WebSocket server:', error);
}

app.post('/api/recognize', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image file provided' });
    }
    
    console.log(`Processing recognition for: ${req.file.filename}`);
    
    const result = await executePythonRAG('recognize', [req.file.path]);
    
    fs.unlinkSync(req.file.path);
    
    res.json({
      success: true,
      ...result,
      processing_time: new Date().toISOString()
    });
    
  } catch (error) {
    console.error('Recognition error:', error);
    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

app.post('/api/register', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image file provided' });
    }
    
    const personName = req.body.name;
    if (!personName) {
      fs.unlinkSync(req.file.path);
      return res.status(400).json({ error: 'Person name is required' });
    }
    
    console.log(`Registering new person: ${personName}`);
    
    const result = await executePythonRAG('register', [req.file.path, personName]);
    
    fs.unlinkSync(req.file.path);
    
    res.json({
      success: result.success || false,
      ...result,
      processing_time: new Date().toISOString()
    });
    
  } catch (error) {
    console.error('Registration error:', error);
    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

app.post('/api/search', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image file provided' });
    }
    
    console.log(`Searching for similar faces: ${req.file.filename}`);
    
    const result = await executePythonRAG('search', [req.file.path]);
    
    fs.unlinkSync(req.file.path);
    
    res.json({
      success: true,
      ...result,
      processing_time: new Date().toISOString()
    });
    
  } catch (error) {
    console.error('Search error:', error);
    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

app.get('/api/status', async (req, res) => {
  try {
    console.log('Checking system status...');
    
    const result = await executePythonRAG('status');
    
    res.json({
      success: true,
      system: result,
      server_time: new Date().toISOString()
    });
    
  } catch (error) {
    console.error('Status check error:', error);
    res.status(500).json({
      success: false,
      error: error.message,
      server_time: new Date().toISOString()
    });
  }
});

app.post('/api/recognize-base64', async (req, res) => {
  try {
    const { image_data, action = 'recognize', person_name } = req.body;
    
    if (!image_data) {
      return res.status(400).json({ error: 'No image data provided' });
    }
    
    const uploadDir = './Uploads';
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir);
    }
    
    const base64Data = image_data.replace(/^data:image\/\w+;base64,/, '');
    const buffer = Buffer.from(base64Data, 'base64');
    const tempPath = path.join(uploadDir, `temp_${Date.now()}.jpg`);
    
    fs.writeFileSync(tempPath, buffer);
    
    console.log(`Processing ${action} for base64 image`);
    
    let result;
    if (action === 'register' && person_name) {
      result = await executePythonRAG('register', [tempPath, person_name]);
    } else {
      result = await executePythonRAG(action, [tempPath]);
    }
    
    fs.unlinkSync(tempPath);
    
    res.json({
      success: true,
      ...result,
      processing_time: new Date().toISOString()
    });
    
  } catch (error) {
    console.error('Base64 processing error:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

app.post('/api/batch-process', upload.array('images', 10), async (req, res) => {
  try {
    if (!req.files || req.files.length === 0) {
      return res.status(400).json({ error: 'No image files provided' });
    }
    
    const action = req.body.action || 'recognize';
    console.log(`Batch processing ${req.files.length} images with action: ${action}`);
    
    const results = [];
    
    for (const file of req.files) {
      try {
        let result;
        if (action === 'register') {
          const personName = req.body[`name_${file.originalname}`] || `Person_${Date.now()}`;
          result = await executePythonRAG('register', [file.path, personName]);
        } else {
          result = await executePythonRAG(action, [file.path]);
        }
        
        results.push({
          filename: file.originalname,
          success: true,
          ...result
        });
        
      } catch (error) {
        results.push({
          filename: file.originalname,
          success: false,
          error: error.message
        });
      }
      
      if (fs.existsSync(file.path)) {
        fs.unlinkSync(file.path);
      }
    }
    
    res.json({
      success: true,
      processed: results.length,
      results: results,
      processing_time: new Date().toISOString()
    });
    
  } catch (error) {
    console.error('Batch processing error:', error);
    if (req.files) {
      req.files.forEach(file => {
        if (fs.existsSync(file.path)) {
          fs.unlinkSync(file.path);
        }
      });
    }
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

app.get('/api/stats', async (req, res) => {
  try {
    const result = await executePythonRAG('status');
    res.json({
      success: true,
      total_faces: result.total_faces || 0,
      total_images: result.total_faces || 0,
      total_recognitions: result.total_recognitions || 0
    });
  } catch (error) {
    console.error('Stats fetch error:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

app.get('/api/registered-people', async (req, res) => {
  try {
    const result = await executePythonRAG('list-people');
    res.json({
      success: true,
      people: result.people || []
    });
  } catch (error) {
    console.error('Registered people fetch error:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

app.post('/api/set-mode', async (req, res) => {
  try {
    const { mode, name } = req.body;
    if (!mode) {
      return res.status(400).json({ error: 'Mode is required' });
    }
    res.json({
      success: true,
      mode,
      message: `Mode set to ${mode}${name ? ` for ${name}` : ''}`
    });
  } catch (error) {
    console.error('Set mode error:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

app.delete('/api/person/:name', async (req, res) => {
  try {
    const { name } = req.params;
    const result = await executePythonRAG('delete-person', [name]);
    res.json({
      success: result.success || false,
      ...result
    });
  } catch (error) {
    console.error('Delete person error:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

app.delete('/api/clear-all', async (req, res) => {
  try {
    const result = await executePythonRAG('clear-all');
    res.json({
      success: result.success || false,
      ...result
    });
  } catch (error) {
    console.error('Clear all error:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    timestamp: new Date().toISOString(),
    service: 'Face Recognition RAG API'
  });
});

app.use((error, req, res, next) => {
  if (error instanceof multer.MulterError) {
    if (error.code === 'LIMIT_FILE_SIZE') {
      return res.status(400).json({ error: 'File too large' });
    }
  }
  
  console.error('Unhandled error:', error);
  res.status(500).json({ error: 'Internal server error' });
});

app.use((req, res) => {
  res.status(404).json({ error: 'Endpoint not found' });
});

app.listen(PORT, () => {
  console.log(`🚀 Face Recognition RAG Server running on port ${PORT}`);
  console.log(`📋 Available endpoints:`);
  console.log(`   POST /api/recognize     - Recognize faces in uploaded image`);
  console.log(`   POST /api/register      - Register new person with image`);
  console.log(`   POST /api/search        - Search for similar faces`);
  console.log(`   GET  /api/status        - Check system status`);
  console.log(`   POST /api/recognize-base64 - Process base64 images`);
  console.log(`   POST /api/batch-process - Process multiple images`);
  console.log(`   GET  /api/stats        - Get system statistics`);
  console.log(`   GET  /api/registered-people - Get list of registered people`);
  console.log(`   POST /api/set-mode     - Set recognition/registration mode`);
  console.log(`   DELETE /api/person/:name - Delete a person`);
  console.log(`   DELETE /api/clear-all  - Clear all data`);
  console.log(`   GET  /health           - Health check`);
});

module.exports = app;
