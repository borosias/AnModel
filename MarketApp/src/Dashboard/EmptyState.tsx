import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Button,
  alpha,
  useTheme,
} from '@mui/material';
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';

interface EmptyStateProps {
  selectedService: string;
  modelFeatures: string[];
  onResetToDefault: () => void;
  setError: (error: string) => void;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  selectedService,
  modelFeatures,
  onResetToDefault,
  setError,
}) => {
  const theme = useTheme();

  const handleStartAnalysis = () => {
    if (!selectedService) {
      setError("Select a model first!");
    } else if (modelFeatures.length === 0) {
      setError("Load model features!");
    } else {
      onResetToDefault();
    }
  };

  return (
    <Card
      elevation={0}
      sx={{
        borderRadius: 3,
        border: `2px dashed ${alpha(theme.palette.primary.main, 0.2)}`,
        background: alpha(theme.palette.primary.main, 0.02),
        height: 'calc(100vh - 200px)',
        display: 'flex',
        alignItems: 'center',
        color: theme.palette.primary.contrastText,
        justifyContent: 'center',
      }}
    >
      <CardContent sx={{ textAlign: 'center', py: 8 }}>
        <AutoFixHighIcon
          sx={{
            fontSize: 80,
            color: alpha(theme.palette.primary.main, 0.2),
            mb: 2,
          }}
        />
        <Typography variant="h5" color="text.contrastText" gutterBottom sx={{ fontWeight: 600 }}>
          No data for analysis
        </Typography>
        <Typography variant="body1" color="text.contrastText" paragraph>
          Select a model, enter data, and start the first analysis
        </Typography>
        <Button
          variant="contained"
          size="large"
          startIcon={<TrendingUpIcon />}
          sx={{
            mt: 2,
            borderRadius: 2,
            px: 4,
            py: 1.5,
            fontWeight: 700,
            background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
          }}
          onClick={handleStartAnalysis}
        >
          Start Analysis
        </Button>
      </CardContent>
    </Card>
  );
};