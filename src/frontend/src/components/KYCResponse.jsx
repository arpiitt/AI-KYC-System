import React from 'react';
import {
  Box,
  Paper,
  Typography,
  Alert,
  List,
  ListItem,
  ListItemText,
  Button,
  Divider,
} from '@mui/material';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';

const KYCResponse = ({ response, onReset }) => {
  if (!response) return null;

  const isSuccess = response.status === 'success';

  const renderSuccessContent = () => (
    <Box sx={{ textAlign: 'center', py: 2 }}>
      <CheckCircleOutlineIcon color="success" sx={{ fontSize: 64, mb: 2 }} />
      <Typography variant="h5" gutterBottom>
        KYC Verification Successful
      </Typography>
      <Typography color="text.secondary" paragraph>
        {response.message}
      </Typography>
      <List sx={{ width: '100%', bgcolor: 'background.paper' }}>
        <ListItem>
          <ListItemText
            primary="Customer ID"
            secondary={response.customer_id}
          />
        </ListItem>
        <Divider />
        <ListItem>
          <ListItemText
            primary="Face Match Score"
            secondary={`${(response.face_match_score * 100).toFixed(2)}%`}
          />
        </ListItem>
      </List>
    </Box>
  );

  const getErrorIcon = () => {
    return <ErrorOutlineIcon color="error" sx={{ fontSize: 64, mb: 2 }} />;
  };

  const getErrorTitle = () => {
    if (response.error_type === 'face_verification_error') {
      return 'Face Verification Failed';
    } else if (response.error_type === 'document_verification_error') {
      return 'Document Verification Failed';
    } else if (response.error_type === 'upload_error') {
      return 'File Upload Failed';
    }
    return 'KYC Verification Failed';
  };

  const getErrorGuidance = () => {
    // Get specific guidance based on error type
    const baseGuidance = [
      'Check all input fields are filled correctly',
      'Ensure your files meet the requirements',
      'Try the process again'
    ];

    // If there's a specific error message and details from the backend, show those first
    if (response.message && response.details) {
      return [
        response.message,
        response.details,
        ...baseGuidance
      ];
    }

    switch (response.error_type) {
      case 'face_verification_error':
        return [
          'Ensure your selfie is well-lit and clear',
          'Look directly at the camera',
          'Remove any accessories (glasses, masks, etc.)',
          'Make sure your face is fully visible',
          'Avoid multiple faces in the image',
          'Use a neutral background',
          ...baseGuidance
        ];
      case 'document_verification_error':
        return [
          'Use a clear, high-quality image of your ID',
          'Ensure all corners and edges are visible',
          'Avoid glare or shadows on the document',
          'Make sure the document is not expired',
          'Place the document on a dark, solid background',
          'Ensure the document text is clearly readable',
          ...baseGuidance
        ];
      case 'upload_error':
        return [
          'Ensure your files are in JPEG or PNG format',
          'File size should be between 100KB and 5MB',
          'Image resolution should be at least 640x480 pixels',
          'Check that the image is not corrupted',
          'Check your internet connection',
          'Try uploading again',
          ...baseGuidance
        ];
      case 'validation_error':
        return [
          'Please check all required fields',
          'Ensure your ID document is valid',
          'Make sure your selfie meets the requirements',
          ...baseGuidance
        ];
      case 'system_error':
        return [
          'A system error occurred',
          'Please try again in a few minutes',
          'If the problem persists, contact support',
          ...baseGuidance
        ];
      default:
        return baseGuidance;
    }
  };

  const renderErrorContent = () => (
    <Box sx={{ textAlign: 'center', py: 2 }}>
      {getErrorIcon()}
      <Typography variant="h5" gutterBottom>
        {getErrorTitle()}
      </Typography>
      <Alert severity="error" sx={{ mb: 2, textAlign: 'left' }}>
        <Typography variant="body1" sx={{ fontWeight: 'bold', mb: 1 }}>
          {response.message}
        </Typography>
        {response.details && (
          <Typography variant="body2" sx={{ mt: 1 }}>
            {response.details}
          </Typography>
        )}
      </Alert>
      <Box sx={{ mt: 3, textAlign: 'left' }}>
        <Typography variant="h6" gutterBottom>
          How to fix this:
        </Typography>
        <List>
          {getErrorGuidance().map((guidance, index) => (
            <ListItem key={index}>
              <ListItemText primary={`${index + 1}. ${guidance}`} />
            </ListItem>
          ))}
        </List>
      </Box>
      {response.details && (
        <Box sx={{ mt: 2, textAlign: 'left' }}>
          <Typography variant="subtitle1" gutterBottom>
            Error Details:
          </Typography>
          <Paper variant="outlined" sx={{ p: 2, bgcolor: '#fafafa' }}>
            {typeof response.details === 'string' ? (
              <Typography color="error">{response.details}</Typography>
            ) : (
              <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
                {JSON.stringify(response.details, null, 2)}
              </pre>
            )}
          </Paper>
        </Box>
      )}
      {response.match_score && (
        <Box sx={{ mt: 2 }}>
          <Typography variant="body2" color="error">
            Face Match Score: {(response.match_score * 100).toFixed(2)}%
            {response.match_score < 0.5 && ' - Low Confidence Match'}
          </Typography>
        </Box>
      )}
    </Box>
  );

  return (
    <Paper elevation={3} sx={{ p: 3, mt: 3 }}>
      {isSuccess ? renderSuccessContent() : renderErrorContent()}
      <Box sx={{ mt: 3, textAlign: 'center' }}>
        <Button
          variant="contained"
          color="primary"
          onClick={onReset}
        >
          Start New Verification
        </Button>
      </Box>
    </Paper>
  );
};

export default KYCResponse;