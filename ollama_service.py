"""
Ollama (Mistral) integration service for the resume analyzer application.
This module handles the integration with locally running Ollama to provide
intelligent responses to user queries about career development,
skill improvement, and job search advice.
"""

import logging
import requests
import json
from typing import Optional, Dict, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "mistral"  # or whatever model you're using with Ollama

def generate_ollama_response(query: str, context: Optional[Dict] = None) -> str:
    """
    Generate a response from Ollama based on the user's query.
    
    Args:
        query (str): The user's question
        context (dict, optional): Additional context like resume data, job info, etc.
        
    Returns:
        str: The response from Ollama
    """
    try:
        # Create a prompt with context if available
        system_prompt = """You are a helpful AI career assistant providing advice on job skills, 
        resume building, and career development. Be specific, concise, and actionable in your advice."""
        
        if context:
            # Add resume and job context if available
            skills_context = ""
            if 'skills' in context and context['skills']:
                skills_context = "User's skills: " + ", ".join(context['skills'])
            
            missing_skills_context = ""
            if 'missing_skills' in context and context['missing_skills']:
                missing_skills_context = "Skills the user needs to develop: " + ", ".join(context['missing_skills'])
            
            job_title_context = ""
            if 'job_title' in context and context['job_title']:
                job_title_context = f"Job user is interested in: {context['job_title']}"
            
            system_prompt += f"\n\nContext:\n{skills_context}\n{missing_skills_context}\n{job_title_context}"
            
            system_prompt += "\n\nProvide specific, actionable advice based on the user's profile and their target job."

        # Prepare the messages for the API call
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        # Make the API request to Ollama
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": MODEL_NAME,
                    "messages": messages,
                    "options": {
                        "temperature": 0.7,
                        "max_tokens": 300
                    }
                },
                timeout=30  # 30 seconds timeout
            )

            response.raise_for_status()
            response_data = response.json()

            # Ollama's response is in the 'message' field
            if 'message' in response_data and 'content' in response_data['message']:
                return response_data['message']['content']
            else:
                logger.error("Unexpected Ollama response format")
                return generate_fallback_response(query, context)
            
        except requests.exceptions.RequestException as api_error:
            logger.error(f"Ollama API call error: {str(api_error)}")
            return generate_fallback_response(query, context)
        
    except Exception as e:
        logger.error(f"Error generating Ollama response: {str(e)}")
        return "I'm sorry, I couldn't process your question. Please try again or ask a different question."

def generate_fallback_response(query: str, context: Optional[Dict] = None) -> str:
    """Generate a fallback response when Ollama is unavailable"""
    # (Keep the same fallback responses as in your original chatgpt_service.py)
    query_lower = query.lower()
    
    # Prepare skills information from context
    skills_text = ""
    if context and 'skills' in context and context['skills']:
        skills_text = ", ".join(context['skills'])
    
    missing_skills_text = ""
    if context and 'missing_skills' in context and context['missing_skills']:
        missing_skills_text = ", ".join(context['missing_skills'])
    
    job_title = ""
    if context and 'job_title' in context and context['job_title']:
        job_title = context['job_title']
    
    # Check for different types of questions and provide appropriate responses
    if any(keyword in query_lower for keyword in ['resume', 'cv', 'improve']):
        skills_advice = ""
        if skills_text:
            skills_advice = " Include these skills in your resume: " + skills_text
        return "To improve your resume, focus on quantifying your achievements and highlighting relevant skills for your target roles. Use action verbs and ensure your experience demonstrates your capabilities clearly." + skills_advice
    
    elif any(keyword in query_lower for keyword in ['skill', 'learn', 'develop']):
        if missing_skills_text:
            return "Based on your profile, I recommend focusing on developing these key skills: " + missing_skills_text + ". You can learn them through online courses on platforms like Coursera, Udemy, or through hands-on projects."
        else:
            return "To develop your skills, consider taking online courses, working on personal projects, contributing to open source, or obtaining relevant certifications in your field."
    
    elif any(keyword in query_lower for keyword in ['interview', 'prep', 'question']):
        skills_advice = ""
        if skills_text:
            skills_advice = " Focus on how you have applied these skills: " + skills_text
        return "Prepare for interviews by researching the company, practicing common questions, and preparing examples that demonstrate your skills and experience." + skills_advice
    
    elif any(keyword in query_lower for keyword in ['job', 'search', 'find', 'application']):
        return "For an effective job search, update your LinkedIn profile, set up job alerts on major platforms, network with professionals in your target field, and tailor each application to the specific role and company."
    
    elif any(keyword in query_lower for keyword in ['salary', 'negotiate', 'offer']):
        return "When negotiating salary, research industry standards, highlight your unique value, consider the total compensation package including benefits, and practice your negotiation approach beforehand."
    
    elif any(keyword in query_lower for keyword in ['career', 'path', 'switch', 'change']):
        return "For a successful career change, identify transferable skills, fill knowledge gaps with targeted learning, network with professionals in your desired field, and consider starting with hybrid roles that bridge your current and target careers."
    
    else:
        return "As a career assistant, I can help with resume optimization, job search strategies, skill development, interview preparation, and career planning. Could you specify which aspect you need help with?"

def is_service_available() -> bool:
    """Check if the Ollama service is available"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False