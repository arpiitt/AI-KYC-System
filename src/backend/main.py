from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.api.kyc_processor import KYCProcessor
from src.models.database import SessionLocal, KYCRecord, get_db
from sqlalchemy.orm import Session
from datetime import datetime
import uvicorn

app = FastAPI(title="AI KYC System")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/initiate-kyc")
async def initiate_kyc(
    customer_id: str = Form(...),
    full_name: str = Form(...),
    id_document: UploadFile = File(...),
    selfie: UploadFile = File(...),
    db: Session = Depends(get_db)
) -> dict:
    """Process KYC verification request
    Returns a standardized JSON response with status and details
    """
    # Initialize response structure
    response = {
        "status": "error",
        "message": "",
        "details": "",
        "error_type": "validation_error"
    }
    # Validate file types
    allowed_types = ["image/jpeg", "image/png"]
    if id_document.content_type not in allowed_types or selfie.content_type not in allowed_types:
        response.update({
            "message": "Invalid file type",
            "details": "Only JPEG and PNG images are allowed.",
            "error_type": "validation_error"
        })
        return response

    id_doc_path = None
    selfie_path = None

    try:
        import os
        # Ensure static directory exists
        os.makedirs("src/static", exist_ok=True)

        # Save uploaded files with unique names
        import uuid
        id_doc_path = f"src/static/{uuid.uuid4()}_{id_document.filename}"
        selfie_path = f"src/static/{uuid.uuid4()}_{selfie.filename}"
        
        # Save files
        try:
            with open(id_doc_path, "wb") as f:
                content = await id_document.read()
                if not content:
                    raise ValueError("Empty ID document file")
                f.write(content)
            with open(selfie_path, "wb") as f:
                content = await selfie.read()
                if not content:
                    raise ValueError("Empty selfie file")
                f.write(content)
        except Exception as e:
            raise ValueError(f"File upload error: {str(e)}")

        # Initialize KYC processor
        processor = KYCProcessor()

        # Verify document authenticity
        try:
            is_doc_valid, doc_checks = processor.verify_document_authenticity(id_doc_path)
            if not is_doc_valid:
                response.update({
                    "message": "Document verification failed",
                    "details": "The provided document could not be verified. Please ensure you've submitted a clear, valid identification document.",
                    "error_type": "document_error",
                    "technical_details": doc_checks
                })
                return response
        except Exception as e:
            response.update({
                "message": "Document processing error",
                "details": "An error occurred while processing your document. Please try again with a clearer image.",
                "error_type": "processing_error",
                "technical_details": str(e)
            })
            return response

        # Verify face match
        try:
            is_face_match, match_score, verification_result = processor.verify_face_match(id_doc_path, selfie_path)
            if not is_face_match:
                response.update({
                    "error_type": verification_result.get("error_type", "face_verification_error"),
                    "message": verification_result.get("message", "Face verification failed"),
                    "details": verification_result.get("details"),
                    "match_score": match_score
                })
                return response
        except Exception as e:
            response.update({
                "message": "Face verification error",
                "details": "An error occurred during face verification. Please ensure both images are clear and try again.",
                "error_type": "face_verification_error",
                "technical_details": str(e)
            })
            return response

        # Extract document data
        doc_data = processor.extract_document_data(id_doc_path)
        if "error" in doc_data:
            response.update({
                "message": "Failed to extract document data",
                "details": doc_data["error"],
                "error_type": "data_extraction_error"
            })
            return response

        # Create KYC record in database
        try:
            kyc_record = KYCRecord(
                customer_id=customer_id,
                full_name=full_name,
                document_type="ID",
                document_number=doc_data.get("document_number", ""),
                date_of_birth=datetime.strptime(doc_data.get("date_of_birth", ""), "%Y-%m-%d") if doc_data.get("date_of_birth") else None,
                address=doc_data.get("address", ""),
                document_path=id_doc_path,
                selfie_path=selfie_path,
                verification_status="verified"
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
                "document_data": doc_data,
                "details": "Your KYC verification has been processed and approved."
            }
        except Exception as db_error:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {str(db_error)}")

    except ValueError as ve:
        response.update({
            "message": "Validation error",
            "details": str(ve),
            "error_type": "validation_error"
        })
        return response
    except Exception as e:
        error_message = str(e)
        error_type = "system_error"
        
        if "File upload error" in error_message:
            error_type = "upload_error"
        
        response.update({
            "error_type": error_type,
            "message": "An error occurred during KYC verification",
            "details": str(e)
        })
        return response
    finally:
        # Cleanup temporary files
        import os
        try:
            os.remove(id_doc_path)
            os.remove(selfie_path)
        except:
            pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)