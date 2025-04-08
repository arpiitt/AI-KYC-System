from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Dict
from datetime import datetime
import os

from .kyc_processor import KYCProcessor
from models.database import get_db, KYCRecord
from sqlalchemy.orm import Session

app = FastAPI(title="AI KYC System")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create upload directory if it doesn't exist
UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/api/kyc/initiate")
async def initiate_kyc(
    customer_id: str = Form(...),
    full_name: str = Form(...),
    id_document: UploadFile = File(...),
    selfie: UploadFile = File(...),
    db: Session = Depends(get_db)
) -> Dict:
    try:
        # Save uploaded files
        doc_path = os.path.join(UPLOAD_DIR, f"{customer_id}_doc.jpg")
        selfie_path = os.path.join(UPLOAD_DIR, f"{customer_id}_selfie.jpg")
        
        with open(doc_path, "wb") as f:
            f.write(await id_document.read())
        with open(selfie_path, "wb") as f:
            f.write(await selfie.read())
        
        # Process KYC verification
        processor = KYCProcessor()
        
        # Verify document authenticity
        is_doc_valid, doc_checks = processor.verify_document_authenticity(doc_path)
        if not is_doc_valid:
            raise HTTPException(status_code=400, detail="Invalid document provided")
        
        # Verify face match
        is_face_match, match_score = processor.verify_face_match(doc_path, selfie_path)
        if not is_face_match:
            raise HTTPException(status_code=400, detail="Face verification failed")
        
        # Extract document data
        doc_data = processor.extract_document_data(doc_path)
        if "error" in doc_data:
            raise HTTPException(status_code=400, detail=doc_data["error"])
        
        # Create KYC record in database
        kyc_record = KYCRecord(
            customer_id=customer_id,
            full_name=full_name,
            document_path=doc_path,
            selfie_path=selfie_path,
            verification_status="verified" if is_face_match else "failed",
            created_at=datetime.utcnow()
        )
        
        db.add(kyc_record)
        db.commit()
        db.refresh(kyc_record)
        
        return {
            "status": "success",
            "message": "KYC verification completed successfully",
            "customer_id": customer_id,
            "verification_status": kyc_record.verification_status,
            "face_match_score": float(match_score),
            "document_checks": doc_checks
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    return {"message": "AI KYC System API"}