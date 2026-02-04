const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

export const chatbotAPI = {
  // Send message to chatbot - UPDATED to use /query endpoint
  sendMessage: async (message) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/chatbot/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to send message');
      }
      
      return await response.json();
    } catch (error) {
      console.error('API Error:', error);
      throw error;
    }
  },

  // Get chat history (if implemented in backend)
  getChatHistory: async (userId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/chatbot/history${userId ? `?userId=${userId}` : ''}`);
      if (!response.ok) {
        throw new Error('Failed to fetch chat history');
      }
      return await response.json();
    } catch (error) {
      console.error('API Error:', error);
      throw error;
    }
  },

  // Health check to verify backend is running
  healthCheck: async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      if (!response.ok) {
        throw new Error('Health check failed');
      }
      return await response.json();
    } catch (error) {
      console.error('Health check error:', error);
      throw error;
    }
  },

  // Test chatbot endpoint
  testChatbot: async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/chatbot/test`);
      if (!response.ok) {
        throw new Error('Test failed');
      }
      return await response.json();
    } catch (error) {
      console.error('Test error:', error);
      throw error;
    }
  },

  // Get available intents
  getIntents: async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/chatbot/intents`);
      if (!response.ok) {
        throw new Error('Failed to fetch intents');
      }
      return await response.json();
    } catch (error) {
      console.error('API Error:', error);
      throw error;
    }
  },

  // Train chatbot with new data (admin feature)
  trainChatbot: async (trainingData) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/training/train`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(trainingData),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to train chatbot');
      }
      
      return await response.json();
    } catch (error) {
      console.error('Training Error:', error);
      throw error;
    }
  },

  // Upload training file
  uploadTrainingFile: async (file) => {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${API_BASE_URL}/api/training/upload`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to upload file');
      }
      
      return await response.json();
    } catch (error) {
      console.error('Upload Error:', error);
      throw error;
    }
  }
};

export default chatbotAPI;