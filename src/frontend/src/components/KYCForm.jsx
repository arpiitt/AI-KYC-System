import React, { useState } from 'react';
import {
  Box,
  Button,
  Container,
  TextField,
  Typography,
  Paper,
  CircularProgress,
} from '@mui/material';
import KYCResponse from './KYCResponse';

const KYCForm = () => {
  const [formData, setFormData] = useState({
    customer_id: '',
    full_name: '',
    id_document: null,
    selfie: null,
  });
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const validateImage = (file) => {
    if (!file) return { valid: false, error: 'No file selected' };
    
    // Check file type
    if (!file.type.match('image.*')) {
      return { valid: false, error: 'File must be an image (JPEG or PNG)' };
    }
    
    // Check file size (100KB to 5MB)
    const minSize = 100 * 1024; // 100KB
    const maxSize = 5 * 1024 * 1024; // 5MB
    if (file.size < minSize || file.size > maxSize) {
      return { 
        valid: false, 
        error: `File size must be between 100KB and 5MB (current size: ${(file.size / 1024).toFixed(1)}KB)`
      };
    }
    
    return { valid: true };
  };

  const handleFileChange = (e) => {
    const { name, files } = e.target;
    const file = files[0];
    
    const validation = validateImage(file);
    if (!validation.valid) {
      setResponse({
        status: 'error',
        error_type: 'upload_error',
        message: 'Invalid file',
        details: validation.error
      });
      e.target.value = ''; // Reset file input
      return;
    }
    
    setFormData({ ...formData, [name]: file });
    if (response?.error_type === 'upload_error') {
      setResponse(null); // Clear previous upload error
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Validate all required fields
    if (!formData.customer_id.trim() || !formData.full_name.trim()) {
      setResponse({
        status: 'error',
        error_type: 'validation_error',
        message: 'Missing required fields',
        details: 'Please fill in all required fields'
      });
      return;
    }
    
    // Validate files
    if (!formData.id_document || !formData.selfie) {
      setResponse({
        status: 'error',
        error_type: 'upload_error',
        message: 'Missing required files',
        details: 'Please upload both ID document and selfie images'
      });
      return;
    }
    
    setLoading(true);
    setResponse(null);

    const submitData = new FormData();
    Object.keys(formData).forEach((key) => {
      submitData.append(key, formData[key]);
    });

    try {
      // Add timeout to the fetch request to avoid long waiting times
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 seconds timeout
      
      try {
        const response = await fetch('/api/initiate-kyc', {
          method: 'POST',
          body: submitData,
          headers: {
            'Accept': 'application/json',
          },
          signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        let result;
        const text = await response.text();
        
        if (!text || text.trim() === '') {
          console.error('Empty response received from server');
          throw new Error('Empty response received from server');
        }
        
        try {
          result = JSON.parse(text);
        } catch (parseError) {
          console.error('Response parsing error:', parseError, 'Response text:', text);
          throw new Error('Server returned an invalid response format. Please try again later.');
        }

        if (!result || typeof result !== 'object') {
          console.error('Invalid response structure:', result);
          throw new Error('Invalid response structure received from server');
        }
      
        // Handle specific API error responses
        if (!response.ok) {
          throw new Error(result?.message || 'Failed to process KYC verification');
        }

        setResponse(result);
      } catch (fetchError) {
        if (fetchError.name === 'AbortError') {
          throw new Error('Request timed out. The server is taking too long to respond.');
        }
        throw fetchError;
      }
    } catch (error) {
      setResponse({
        status: 'error',
        error_type: 'system_error',
        message: error.message || 'An error occurred while processing your request',
        details: 'Please check your input and try again. If the problem persists, contact support.'
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="sm">
      <Paper elevation={3} sx={{ p: 4, mt: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          KYC Verification
        </Typography>

        {!response && (

        <Box component="form" onSubmit={handleSubmit} noValidate>
          <TextField
            margin="normal"
            required
            fullWidth
            id="customer_id"
            label="Customer ID"
            name="customer_id"
            value={formData.customer_id}
            onChange={handleInputChange}
          />

          <TextField
            margin="normal"
            required
            fullWidth
            id="full_name"
            label="Full Name"
            name="full_name"
            value={formData.full_name}
            onChange={handleInputChange}
          />

          <Box sx={{ mt: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              ID Document
            </Typography>
            <input
              accept="image/*"
              type="file"
              name="id_document"
              onChange={handleFileChange}
              required
            />
          </Box>

          <Box sx={{ mt: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              Selfie
            </Typography>
            <input
              accept="image/*"
              type="file"
              name="selfie"
              onChange={handleFileChange}
              required
            />
          </Box>

          <Button
            type="submit"
            fullWidth
            variant="contained"
            sx={{ mt: 3, mb: 2 }}
            disabled={loading}
          >
            {loading ? <CircularProgress size={24} /> : 'Submit KYC'}
          </Button>
        </Box>
        )}
        {response && (
          <KYCResponse
            response={response}
            onReset={() => {
              setResponse(null);
              setFormData({
                customer_id: '',
                full_name: '',
                id_document: null,
                selfie: null,
              });
            }}
          />
        )}
      </Paper>
    </Container>
  );
};

export default KYCForm;