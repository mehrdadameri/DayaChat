# DayaChat

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=Streamlit&logoColor=white)](https://streamlit.io)

DayaChat is a powerful multi-model chatbot application built with Streamlit that supports various Large Language Model providers including OpenAI GPT-4o, Google Gemini, and DeepSeek.

![DayaChat Interface](https://via.placeholder.com/800x400?text=DayaChat+Interface)

## Table of Contents

- [DayaChat](#dayachat)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
  - [API Keys Setup](#api-keys-setup)
    - [OpenAI (GPT-4o)](#openai-gpt-4o)
    - [Google Gemini](#google-gemini)
    - [DeepSeek](#deepseek)
  - [Data Storage](#data-storage)
  - [Development Notes](#development-notes)
  - [License](#license)

## Features

- **Multiple LLM Providers Support**:
  - OpenAI GPT-4o
  - Google Gemini
  - DeepSeek
- **Chat Management**:
  - Create new chat sessions
  - Delete existing chat histories
  - Persistent storage of conversations
- **User-Friendly Interface**:
  - Clean design with Streamlit
  - Responsive layout
  - Model selection dropdown

## Prerequisites

- Python 3.8 or higher
- Internet connection (for API calls)
- API keys for desired LLM providers

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/dayachat.git
   cd dayachat
   ```

2. **Set up a virtual environment** (recommended):

   ```bash
   # Create a virtual environment
   python -m venv venv

   # Activate the virtual environment
   # On Windows
   .\venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   Or install individually:

   ```bash
   pip install streamlit openai google-generativeai deepseek-ai
   ```

## Usage

1. **Start the application**:

   ```bash
   streamlit run app.py
   ```

2. **Access the web interface**:

   Open your web browser and navigate to http://localhost:8501

3. **Configure API keys**:

   Enter your API keys in the sidebar for the models you wish to use

4. **Select your preferred model** from the dropdown menu

5. **Start chatting!** Type your message in the input field and press Enter

## API Keys Setup

To use DayaChat with different LLM providers, you'll need to obtain API keys:

### OpenAI (GPT-4o)

1. Go to [OpenAI Platform](https://platform.openai.com/signup)
2. Create an account or sign in
3. Navigate to API keys section
4. Create a new API key
5. Copy the key and paste it in DayaChat's sidebar

### Google Gemini

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the key and paste it in DayaChat's sidebar

### DeepSeek

> **Note**: The DeepSeek integration is currently provided as a placeholder and is not yet fully implemented. When the project matures, this section will be updated with actual implementation details.

For future integration:

1. Go to [DeepSeek Platform](https://platform.deepseek.com/)
2. Create an account or sign in
3. Navigate to the API section
4. Generate a new API key
5. Copy the key and paste it in DayaChat's sidebar

## Data Storage

Chat histories are stored locally in JSON format within the `chat_history` directory. Each chat session is assigned a unique UUID identifier file.

## Development Notes

- The DeepSeek integration is currently provided as a placeholder. In a production environment, implement the actual API calls using DeepSeek's official SDK.
- Feel free to contribute to this project by submitting issues or pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
