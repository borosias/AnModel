import React from 'react';
import {
  TextField,
  Stack,
  InputAdornment,
  Badge,
  Alert,
} from '@mui/material';
import { TRANSLATIONS } from '../constants';

interface ManualInputProps {
  modelFeatures: string[];
  inputData: Record<string, any>;
  onInputChange: (field: string, value: string) => void;
}

export const ManualInput: React.FC<ManualInputProps> = ({
  modelFeatures,
  inputData,
  onInputChange,
}) => {
  if (modelFeatures.length === 0) {
    return (
      <Alert severity="info" sx={{ borderRadius: 2 }}>
        Оберіть модель для відображення полів вводу
      </Alert>
    );
  }

  return (
    <Stack spacing={2}>
      {modelFeatures.map((feature, index) => (
        <TextField
          key={feature}
          fullWidth
          label={TRANSLATIONS[feature] || feature}
          type="number"
          value={inputData[feature] || 0}
          onChange={(e) => onInputChange(feature, e.target.value)}
          size="medium"
          sx={{
            '& .MuiOutlinedInput-root': { borderRadius: 2 }
          }}
          InputProps={{
            inputProps: {
              step: "any",
              min: 0,
              style: { textAlign: 'right' }
            },
            startAdornment: (
              <InputAdornment position="start">
                <Badge
                  badgeContent={index + 1}
                  color="primary"
                  sx={{
                    '& .MuiBadge-badge': {
                      fontSize: '0.7rem',
                      height: 20,
                      minWidth: 20,
                    }
                  }}
                />
              </InputAdornment>
            ),
          }}
        />
      ))}
    </Stack>
  );
};