import { AIResponse, ChatMessage } from '../types/chat';

interface ChatResponse {
  response: string;
  error?: string;
}

interface EmailRequest {
  to: string;
  subject: string;
  body: string;
}

export const aiChatService = {
  sendMessage: async (
    question: string,
    conversations: Array<{ role: string; content: string }>,
    option: string,
    imgurl: string
  ) => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/doctor/chat-ai`, {  // Updated endpoint
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question,
          conversationsNew: conversations,
          option,
          imgurl
        })
      });

      if (!response.ok) {
        const errorData = await response.text();
        console.error('Server response error:', {
          status: response.status,
          statusText: response.statusText,
          data: errorData
        });
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Network or parsing error:', error);
      throw error;
    }
  }
};