#!/usr/bin/env python3
"""
LandingAI Document Processing Script
Basic setup for using LandingAI's Agentic Document Extraction API
"""

import os
from dotenv import load_dotenv
from agentic_doc.parse import parse

# Load environment variables from .env file
load_dotenv()

def main():
    """Main function to demonstrate document parsing with LandingAI"""
    
    # Check if API key is set
    api_key = os.getenv('VISION_AGENT_API_KEY')
    if not api_key:
        print("Error: VISION_AGENT_API_KEY not found in environment variables")
        print("Please set your API key in the .env file or as an environment variable")
        return
    
    print("LandingAI Document Processing Setup Complete!")
    print("API Key loaded successfully")
    
    # Example usage (uncomment when you have a document to test)
    # result = parse("path/to/your/document.pdf")
    # print(result.markdown)
    
    # To save results to a directory:
    # result = parse("path/to/your/document.pdf", result_save_dir="./results")

if __name__ == "__main__":
    main()