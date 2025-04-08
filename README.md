# AI-Powered KYC Verification System

A modern Know Your Customer (KYC) verification system that uses artificial intelligence for face matching and document verification. This system provides a seamless way to verify user identity through document validation and facial recognition.

## Tech Stack

### Backend
- **FastAPI**: Modern, fast web framework for building APIs with Python
- **SQLAlchemy**: SQL toolkit and ORM
- **OpenCV**: Computer vision library for image processing
- **face-recognition**: Face detection and recognition library
- **PyTesseract**: OCR (Optical Character Recognition) tool
- **Uvicorn**: ASGI server for running the FastAPI application

### Frontend
- **React**: JavaScript library for building user interfaces
- **Vite**: Next generation frontend tooling
- **Tailwind CSS**: Utility-first CSS framework

### Database
- **SQLite**: Lightweight, file-based database

## Project Structure
```
├── src/
│   ├── api/
│   │   ├── kyc_processor.py    # KYC verification logic
│   │   └── main.py             # FastAPI application
│   ├── frontend/               # React frontend application
│   ├── models/
│   │   └── database.py         # Database models and configuration
│   └── static/                 # Static files and uploads
├── requirements.txt            # Python dependencies
├── start_backend.py            # Backend startup script
└── README.md                   # Project documentation
```

## Prerequisites

- Python 3.8 or higher
- Node.js 14 or higher
- npm or yarn package manager

## Installation

1. Clone the repository:
```bash
https://github.com/arpiitt/AI-KYC-System.git
```

2. Set up the Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: .\venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Install frontend dependencies:
```bash
cd src/frontend
npm install
```

## Running the Application

1. Start the backend server:
```bash
python start_backend.py
```
The backend API will be available at `http://localhost:8000`

2. Start the frontend development server:
```bash
cd src/frontend
npm run dev
```
The frontend application will be available at `http://localhost:5173`

## API Endpoints

- `POST /api/kyc/initiate`: Initiate KYC verification
  - Required fields:
    - `customer_id`: Unique identifier for the customer
    - `full_name`: Customer's full name
    - `id_document`: ID document image file (JPEG/PNG)
    - `selfie`: Selfie image file (JPEG/PNG)

## Features

- Face matching between ID document and selfie
- Document authenticity verification
- OCR for document data extraction
- Real-time verification status updates
- Secure file handling and storage
- Comprehensive error handling and validation

## Security Considerations

- All uploaded files are stored securely with unique identifiers
- Input validation and sanitization
- Rate limiting on API endpoints
- Secure database configuration with connection pooling

## Development

- The project uses SQLite for development. For production, consider using a more robust database like PostgreSQL
- The frontend is configured with Vite for optimal development experience
- Hot reloading is enabled for both frontend and backend during development
