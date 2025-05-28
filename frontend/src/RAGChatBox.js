import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api';

const RAGChatBox = ({ onImageUpload, currentStats }) => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'system',
      content: 'Hello! I\'m your Face Recognition AI assistant. You can ask me about:',
      suggestions: [
        'Recognize a face',
        'Register a new person',
        'Search for similar faces',
        'System statistics',
        'Recent recognitions'
      ],
      timestamp: new Date()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const addMessage = (message) => {
    const newMessage = {
      id: Date.now(),
      timestamp: new Date(),
      ...message
    };
    setMessages(prev => [...prev, newMessage]);
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = inputMessage.trim();
    setInputMessage('');
    
    // Add user message
    addMessage({
      type: 'user',
      content: userMessage
    });

    setIsLoading(true);

    try {
      // Process the message with RAG backend
      const response = await processRAGQuery(userMessage);
      
      // Add AI response
      addMessage({
        type: 'assistant',
        content: response.content,
        data: response.data,
        actions: response.actions,
        suggestions: response.suggestions
      });

    } catch (error) {
      addMessage({
        type: 'error',
        content: 'Sorry, I encountered an error processing your request. Please try again.'
      });
    } finally {
      setIsLoading(false);
    }
  };

  const processRAGQuery = async (query) => {
    const lowerQuery = query.toLowerCase();

    // Intent detection and routing
    if (lowerQuery.includes('status') || lowerQuery.includes('stats') || lowerQuery.includes('system')) {
      return await handleStatusQuery();
    } else if (lowerQuery.includes('register') || lowerQuery.includes('add') || lowerQuery.includes('new person')) {
      return handleRegisterQuery(query);
    } else if (lowerQuery.includes('recognize') || lowerQuery.includes('identify') || lowerQuery.includes('who is')) {
      return handleRecognizeQuery();
    } else if (lowerQuery.includes('search') || lowerQuery.includes('find') || lowerQuery.includes('similar')) {
      return handleSearchQuery();
    } else if (lowerQuery.includes('help') || lowerQuery.includes('what can')) {
      return handleHelpQuery();
    } else if (lowerQuery.includes('delete') || lowerQuery.includes('remove')) {
      return handleDeleteQuery(query);
    } else {
      return handleGeneralQuery(query);
    }
  };

  const handleStatusQuery = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/status`);
      const systemData = response.data;

      return {
        content: `📊 **System Status Report**\n\n` +
                `🟢 **Status**: ${systemData.system?.status || 'Active'}\n` +
                `👥 **Total Faces**: ${systemData.system?.total_faces || 0}\n` +
                `🔄 **Total Recognitions**: ${systemData.system?.total_recognitions || 0}\n` +
                `🤖 **AI Model**: ${systemData.system?.insightface_loaded ? 'Loaded ✅' : 'Not Loaded ❌'}\n` +
                `⏰ **Last Updated**: ${new Date(systemData.system?.last_updated).toLocaleString()}\n\n` +
                `The system is ${systemData.system?.status === 'active' ? 'running smoothly' : 'experiencing issues'}.`,
        data: systemData
      };
    } catch (error) {
      return {
        content: 'Unable to fetch system status. Please check if the backend is running.',
        data: null
      };
    }
  };

  const handleRegisterQuery = (query) => {
    // Extract name if mentioned
    const nameMatch = query.match(/register|add|new\s+(?:person\s+)?(.+)|(.+)\s+(?:as\s+)?(?:new\s+)?person/i);
    const extractedName = nameMatch ? (nameMatch[1] || nameMatch[2])?.trim() : '';

    return {
      content: `📝 **Face Registration**\n\n` +
              `To register a new person${extractedName ? ` (${extractedName})` : ''}:\n\n` +
              `1. Click the "Upload Image" button below\n` +
              `2. Select a clear photo of the person\n` +
              `3. I'll process and register their face\n\n` +
              `For best results, use a high-quality image with good lighting and a clear view of the face.`,
      actions: [
        {
          type: 'upload',
          label: '📷 Upload Image to Register',
          action: 'register',
          name: extractedName
        }
      ]
    };
  };

  const handleRecognizeQuery = () => {
    return {
      content: `🔍 **Face Recognition**\n\n` +
              `To recognize a person in an image:\n\n` +
              `1. Upload an image using the button below\n` +
              `2. I'll analyze the faces and match them with registered people\n` +
              `3. You'll get detailed recognition results with confidence scores\n\n` +
              `I can identify multiple faces in a single image!`,
      actions: [
        {
          type: 'upload',
          label: '🔍 Upload Image to Recognize',
          action: 'recognize'
        }
      ]
    };
  };

  const handleSearchQuery = () => {
    return {
      content: `🔎 **Face Search**\n\n` +
              `To find similar faces in the database:\n\n` +
              `1. Upload a reference image\n` +
              `2. I'll search for similar faces in our database\n` +
              `3. You'll get a list of similar people with similarity scores\n\n` +
              `This is useful for finding potential duplicates or similar-looking people.`,
      actions: [
        {
          type: 'upload',
          label: '🔎 Upload Image to Search',
          action: 'search'
        }
      ]
    };
  };

  const handleDeleteQuery = (query) => {
    return {
      content: `🗑️ **Delete Person**\n\n` +
              `To delete a person from the database, I need to know who you want to remove.\n\n` +
              `Please specify the person's name, for example:\n` +
              `"Delete John Smith" or "Remove Sarah from database"\n\n` +
              `⚠️ **Warning**: This action cannot be undone and will remove all faces associated with this person.`,
    };
  };

  const handleHelpQuery = () => {
    return {
      content: `🤖 **Face Recognition AI Assistant**\n\n` +
              `I can help you with:\n\n` +
              `🔍 **Recognition**: "Recognize this person" or "Who is in this image?"\n` +
              `📝 **Registration**: "Register new person" or "Add John as new person"\n` +
              `🔎 **Search**: "Find similar faces" or "Search for lookalikes"\n` +
              `📊 **Status**: "System status" or "Show statistics"\n` +
              `🗑️ **Management**: "Delete person" or "Remove John Smith"\n\n` +
              `You can also upload images directly and I'll process them based on context!`,
      suggestions: [
        'Show system status',
        'Register a new person',
        'Recognize faces in image',
        'Search for similar faces'
      ]
    };
  };

  const handleGeneralQuery = async (query) => {
    // Try to extract useful information and provide contextual help
    return {
      content: `I understand you're asking about: "${query}"\n\n` +
              `I'm specialized in face recognition tasks. Here's what I can help with:\n\n` +
              `• **Face Recognition**: Upload images to identify people\n` +
              `• **Person Registration**: Add new people to the database\n` +
              `• **Face Search**: Find similar faces\n` +
              `• **System Information**: Get status and statistics\n\n` +
              `Try asking something like "recognize this person" or "show system status"`,
      suggestions: [
        'Help me recognize a face',
        'Register a new person',
        'Show system status',
        'What can you do?'
      ]
    };
  };

  const handleImageUpload = async (file, action = 'recognize', personName = '') => {
    const formData = new FormData();
    formData.append('image', file);
    
    if (personName) {
      formData.append('name', personName);
    }

    addMessage({
      type: 'user',
      content: `📷 Uploading image for ${action}...`,
      imageFile: file
    });

    setIsLoading(true);

    try {
      let endpoint = '';
      let successMessage = '';

      switch (action) {
        case 'register':
          endpoint = '/register';
          successMessage = `Successfully registered ${personName || 'new person'}!`;
          break;
        case 'recognize':
          endpoint = '/recognize';
          successMessage = 'Face recognition completed!';
          break;
        case 'search':
          endpoint = '/search';
          successMessage = 'Face search completed!';
          break;
        default:
          endpoint = '/recognize';
      }

      const response = await axios.post(`${API_BASE_URL}${endpoint}`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      if (response.data.success || response.data.recognized !== undefined) {
        addMessage({
          type: 'assistant',
          content: formatRAGResponse(response.data, action),
          data: response.data
        });
      } else {
        addMessage({
          type: 'error',
          content: response.data.error || 'Processing failed'
        });
      }

    } catch (error) {
      addMessage({
        type: 'error',
        content: `Failed to process image: ${error.response?.data?.error || error.message}`
      });
    } finally {
      setIsLoading(false);
    }
  };

  const formatRAGResponse = (data, action) => {
    let response = '';

    switch (action) {
      case 'recognize':
        if (data.recognized) {
          response = `🎯 **Recognition Successful!**\n\n` +
                    `👤 **Person**: ${data.person.name}\n` +
                    `🎯 **Confidence**: ${(data.person.confidence * 100).toFixed(1)}%\n` +
                    `📊 **Quality Score**: ${(data.technical.face_quality * 100).toFixed(1)}%\n\n`;
          
          if (data.context && data.context.context) {
            response += `📋 **Context Information**:\n${data.context.context}\n\n`;
          }
          
          if (data.context && data.context.recommendations) {
            response += `💡 **Recommendations**:\n`;
            data.context.recommendations.forEach(rec => {
              response += `• ${rec}\n`;
            });
          }
        } else {
          response = `❓ **Person Not Recognized**\n\n` +
                    `The face was detected but not found in our database.\n\n`;
          
          if (data.context && data.context.similar_count > 0) {
            response += `🔍 Found ${data.context.similar_count} similar faces with low confidence.\n\n`;
          }
          
          response += `💡 **Suggestions**:\n` +
                     `• Register this person if they're new\n` +
                     `• Check image quality and try again\n` +
                     `• Ensure the person is already registered`;
        }
        break;

      case 'register':
        if (data.success) {
          response = `✅ **Registration Successful!**\n\n` +
                    `👤 **Person**: ${data.message}\n` +
                    `🆔 **Face ID**: ${data.face_id}\n` +
                    `📊 **Quality Score**: ${(data.technical?.face_quality * 100 || 0).toFixed(1)}%\n\n` +
                    `The person has been added to the recognition database.`;
        } else {
          response = `❌ **Registration Failed**\n\n` +
                    `${data.error}\n\n`;
          
          if (data.similar_person) {
            response += `👥 **Similar Person Found**: ${data.similar_person}\n` +
                       `This might be a duplicate registration.`;
          }
        }
        break;

      case 'search':
        response = `🔎 **Face Search Results**\n\n`;
        
        if (data.search_results && data.search_results.length > 0) {
          response += `Found ${data.search_results.length} similar faces:\n\n`;
          
          data.search_results.forEach((result, idx) => {
            response += `${idx + 1}. **${result.name}**\n` +
                       `   Similarity: ${(result.similarity * 100).toFixed(1)}%\n` +
                       `   Recognitions: ${result.metadata.recognition_count || 0}\n\n`;
          });
        } else {
          response += `No similar faces found in the database.\n\n` +
                     `This might be a completely new person.`;
        }
        break;

      default:
        response = `✅ **Processing Complete**\n\n${data.message || 'Operation completed successfully.'}`;
    }

    return response;
  };

  const handleActionClick = (action) => {
    if (action.type === 'upload') {
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = 'image/*';
      input.onchange = (e) => {
        const file = e.target.files[0];
        if (file) {
          let personName = action.name || '';
          if (action.action === 'register' && !personName) {
            personName = prompt('Enter person name:') || '';
          }
          handleImageUpload(file, action.action, personName);
        }
      };
      input.click();
    }
  };

  const handleSuggestionClick = (suggestion) => {
    setInputMessage(suggestion);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const formatMessageContent = (content) => {
    return content.split('\n').map((line, idx) => {
      if (line.startsWith('**') && line.endsWith('**')) {
        return <div key={idx} className="font-bold text-blue-600 mt-2">{line.slice(2, -2)}</div>;
      } else if (line.startsWith('• ')) {
        return <div key={idx} className="ml-4 text-gray-700">{line}</div>;
      } else if (line.trim() === '') {
        return <br key={idx} />;
      } else {
        return <div key={idx} className="text-gray-800">{line}</div>;
      }
    });
  };

  return (
    <div className="fixed bottom-4 right-4 z-50">
      {/* Chat Toggle Button */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className={`mb-2 p-3 rounded-full shadow-lg transition-all duration-300 ${
          isExpanded 
            ? 'bg-red-500 hover:bg-red-600' 
            : 'bg-blue-500 hover:bg-blue-600'
        } text-white`}
      >
        {isExpanded ? '✕' : '🤖'}
      </button>

      {/* Chat Box */}
      {isExpanded && (
        <div className="bg-white rounded-lg shadow-2xl border border-gray-200 w-96 h-96 flex flex-col">
          {/* Header */}
          <div className="bg-gradient-to-r from-blue-500 to-purple-600 text-white p-3 rounded-t-lg">
            <h3 className="font-bold text-sm">🤖 Face Recognition AI</h3>
            <p className="text-xs opacity-90">Ask me anything about face recognition!</p>
          </div>

          {/* Messages Container */}
          <div className="flex-1 overflow-y-auto p-3 space-y-3 bg-gray-50">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-xs p-2 rounded-lg text-sm ${
                    message.type === 'user'
                      ? 'bg-blue-500 text-white'
                      : message.type === 'error'
                      ? 'bg-red-100 text-red-800 border border-red-200'
                      : message.type === 'system'
                      ? 'bg-purple-100 text-purple-800 border border-purple-200'
                      : 'bg-white text-gray-800 border border-gray-200'
                  }`}
                >
                  <div className="whitespace-pre-wrap">
                    {formatMessageContent(message.content)}
                  </div>

                  {/* Suggestions */}
                  {message.suggestions && (
                    <div className="mt-2 space-y-1">
                      {message.suggestions.map((suggestion, idx) => (
                        <button
                          key={idx}
                          onClick={() => handleSuggestionClick(suggestion)}
                          className="block w-full text-left px-2 py-1 text-xs bg-blue-50 hover:bg-blue-100 
                                   text-blue-700 rounded border border-blue-200 transition-colors"
                        >
                          {suggestion}
                        </button>
                      ))}
                    </div>
                  )}

                  {/* Action Buttons */}
                  {message.actions && (
                    <div className="mt-2 space-y-1">
                      {message.actions.map((action, idx) => (
                        <button
                          key={idx}
                          onClick={() => handleActionClick(action)}
                          className="block w-full px-3 py-2 text-xs bg-green-500 hover:bg-green-600 
                                   text-white rounded transition-colors"
                        >
                          {action.label}
                        </button>
                      ))}
                    </div>
                  )}

                  {/* Timestamp */}
                  <div className="text-xs opacity-70 mt-1">
                    {message.timestamp.toLocaleTimeString()}
                  </div>
                </div>
              </div>
            ))}

            {/* Loading Indicator */}
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-white text-gray-800 border border-gray-200 p-2 rounded-lg text-sm">
                  <div className="flex items-center space-x-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                    <span>Processing...</span>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="border-t border-gray-200 p-3">
            <div className="flex space-x-2">
              <input
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask me about face recognition..."
                className="flex-1 border border-gray-300 rounded-lg px-3 py-2 text-sm 
                         focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                disabled={isLoading}
              />
              <button
                onClick={handleSendMessage}
                disabled={isLoading || !inputMessage.trim()}
                className="bg-blue-500 hover:bg-blue-600 disabled:bg-gray-300 
                         text-white px-4 py-2 rounded-lg text-sm transition-colors"
              >
                📤
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default RAGChatBox;