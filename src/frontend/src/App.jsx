import React from 'react';
import { CssBaseline, ThemeProvider, createTheme } from '@mui/material';
import KYCForm from './components/KYCForm';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    background: {
      default: '#f5f5f5',
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <KYCForm />
    </ThemeProvider>
  );
}

export default App;