import cv2
import numpy as np
from PIL import Image
import pytesseract
from typing import Tuple, Dict
import os

class KYCProcessor:
    def verify_face_match(self, id_doc_path: str, selfie_path: str) -> tuple:
        """Verify if faces are detected in both images and match"""
        try:
            # Load images
            id_image = cv2.imread(id_doc_path)
            selfie_image = cv2.imread(selfie_path)
            
            # Check if images were loaded successfully
            if id_image is None or id_image.size == 0:
                return False, 0.0, {
                    "error_type": "upload_error",
                    "message": "Failed to load ID document image",
                    "details": "The ID document image file appears to be corrupted or in an unsupported format. Please ensure you're uploading a valid JPEG or PNG image."
                }
            if selfie_image is None or selfie_image.size == 0:
                return False, 0.0, {
                    "error_type": "upload_error",
                    "message": "Failed to load selfie image",
                    "details": "The selfie image file appears to be corrupted or in an unsupported format. Please ensure you're uploading a valid JPEG or PNG image."
                }
            
            # Check image dimensions
            min_dimension = 300
            if id_image.shape[0] < min_dimension or id_image.shape[1] < min_dimension:
                return False, 0.0, {
                    "error_type": "image_quality_error",
                    "message": "ID document image too small",
                    "details": f"The ID document image must be at least {min_dimension}x{min_dimension} pixels."
                }
            if selfie_image.shape[0] < min_dimension or selfie_image.shape[1] < min_dimension:
                return False, 0.0, {
                    "error_type": "image_quality_error",
                    "message": "Selfie image too small",
                    "details": f"The selfie image must be at least {min_dimension}x{min_dimension} pixels."
                }
            
            # Convert to grayscale and check image quality
            try:
                id_gray = cv2.cvtColor(id_image, cv2.COLOR_BGR2GRAY)
                selfie_gray = cv2.cvtColor(selfie_image, cv2.COLOR_BGR2GRAY)
                
                # Check image brightness
                id_brightness = cv2.mean(id_gray)[0]
                selfie_brightness = cv2.mean(selfie_gray)[0]
                
                if id_brightness < 40:
                    return False, 0.0, {
                        "error_type": "image_quality_error",
                        "message": "ID document image too dark",
                        "details": "Please provide a well-lit photo of your ID document."
                    }
                if selfie_brightness < 40:
                    return False, 0.0, {
                        "error_type": "image_quality_error",
                        "message": "Selfie image too dark",
                        "details": "Please take a selfie in better lighting conditions."
                    }
                
                # Check image blur
                id_laplacian = cv2.Laplacian(id_gray, cv2.CV_64F).var()
                selfie_laplacian = cv2.Laplacian(selfie_gray, cv2.CV_64F).var()
                
                if id_laplacian < 100:
                    return False, 0.0, {
                        "error_type": "image_quality_error",
                        "message": "ID document image too blurry",
                        "details": "Please provide a clearer, focused photo of your ID document."
                    }
                if selfie_laplacian < 100:
                    return False, 0.0, {
                        "error_type": "image_quality_error",
                        "message": "Selfie image too blurry",
                        "details": "Please take a clearer selfie, ensuring your face is in focus."
                    }
            except Exception as e:
                return False, 0.0, {
                    "error_type": "upload_error",
                    "message": "Error processing images",
                    "details": "The uploaded images could not be processed. Please ensure both images are valid color images in JPEG or PNG format."
                }
            
            # Load face cascade classifier
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if face_cascade.empty():
                return False, 0.0, {
                    "error_type": "system_error",
                    "message": "Face detection system unavailable",
                    "details": "The face detection system is temporarily unavailable. Please try again in a few minutes."
                }
            
            # Detect faces with more lenient parameters
            id_faces = face_cascade.detectMultiScale(id_gray, scaleFactor=1.2, minNeighbors=3, minSize=(80, 80))
            selfie_faces = face_cascade.detectMultiScale(selfie_gray, scaleFactor=1.2, minNeighbors=3, minSize=(80, 80))
            
            # Check if faces were detected in both images
            if len(id_faces) == 0:
                return False, 0.0, {
                    "error_type": "face_verification_error",
                    "message": "No face detected in ID document",
                    "details": "We couldn't detect a face in your ID document. Please ensure the document shows a clear photo of your face."
                }
            
            if len(selfie_faces) == 0:
                return False, 0.0, {
                    "error_type": "face_verification_error",
                    "message": "No face detected in selfie",
                    "details": "We couldn't detect a face in your selfie. Please take a front-facing photo."
                }
            
            # Extract largest face from each image
            id_face = max(id_faces, key=lambda f: f[2] * f[3])
            selfie_face = max(selfie_faces, key=lambda f: f[2] * f[3])
            
            # Extract face regions
            id_face_img = id_gray[id_face[1]:id_face[1]+id_face[3], id_face[0]:id_face[0]+id_face[2]]
            selfie_face_img = selfie_gray[selfie_face[1]:selfie_face[1]+selfie_face[3], selfie_face[0]:selfie_face[0]+selfie_face[2]]
            
            # Resize faces to same size for comparison
            face_size = (128, 128)
            id_face_resized = cv2.resize(id_face_img, face_size)
            selfie_face_resized = cv2.resize(selfie_face_img, face_size)
            
            # Calculate similarity score using normalized correlation coefficient
            similarity = cv2.matchTemplate(id_face_resized, selfie_face_resized, cv2.TM_CCORR_NORMED)[0][0]
            similarity_score = float(similarity * 100)
            
            # Lower threshold for face matching (60% similarity)
            if similarity_score >= 60.0:
                return True, similarity_score, {
                    "error_type": None,
                    "message": "Face verification successful",
                    "details": "Your face has been successfully verified against your ID document."
                }
            else:
                return False, similarity_score, {
                    "error_type": "face_verification_error",
                    "message": "Face verification failed",
                    "details": "The face in your selfie doesn't match the face in your ID document closely enough. Please try taking another selfie with better lighting and a clear view of your face."
                }
            
        except ValueError as ve:
            return False, 0.0, {
                "error_type": "face_verification_error",
                "message": "Face verification failed",
                "details": str(ve)
            }
        except Exception as e:
            return False, 0.0, {
                "error_type": "system_error",
                "message": "System error during face verification",
                "details": "An unexpected error occurred during face verification. Please try again later."
            }


    @staticmethod
    def verify_document_authenticity(document_path: str) -> Tuple[bool, Dict]:
        try:
            # Validate file path
            if not os.path.exists(document_path):
                raise FileNotFoundError("Document file not found")

            # Load the image
            image = cv2.imread(document_path)
            if image is None:
                raise ValueError("Could not load document image")
            
            # Enhanced document checks with detailed feedback
            checks = {
                "resolution_check": {
                    "passed": image.shape[0] >= 1000 and image.shape[1] >= 1000,
                    "message": "Image resolution must be at least 1000x1000 pixels"
                },
                "color_check": {
                    "passed": len(image.shape) == 3,
                    "message": "Image must be in color format"
                },
                "size_check": {
                    "passed": os.path.getsize(document_path) < 10 * 1024 * 1024,
                    "message": "File size must be less than 10MB"
                }
            }
            
            # Collect failed checks
            failed_checks = []
            for check_name, check_data in checks.items():
                if not check_data["passed"]:
                    failed_checks.append(f"{check_name}: {check_data['message']}")
            
            # Overall verification result
            is_valid = all(check["passed"] for check in checks.values())
            
            if not is_valid:
                raise ValueError(f"Document verification failed: {'; '.join(failed_checks)}")
            
            return is_valid, {"checks": {k: v["passed"] for k, v in checks.items()}}
        except Exception as e:
            raise Exception(f"Document verification error: {str(e)}")


    @staticmethod
    def extract_document_data(document_path: str) -> Dict:
        try:
            # Validate file path
            if not os.path.exists(document_path):
                raise FileNotFoundError("Document file not found")

            # Open the image
            image = Image.open(document_path)
            
            # Extract text using OCR
            text = pytesseract.image_to_string(image)
            if not text.strip():
                raise ValueError("No text could be extracted from the document")
            
            # Process the extracted text
            lines = text.split('\n')
            data = {
                "raw_text": text,
                "extracted_lines": [line for line in lines if line.strip()]
            }
            
            return data
        except Exception as e:
            raise Exception(f"Document data extraction error: {str(e)}")