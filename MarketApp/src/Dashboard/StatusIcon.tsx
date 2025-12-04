import React from 'react';
import { Chip } from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import WarningIcon from '@mui/icons-material/Warning';
import {TRANSLATIONS} from "../constants";

interface StatusIconProps {
  status: string;
  size?: 'small' | 'medium';
}

export const StatusIcon: React.FC<StatusIconProps> = ({ status, size = 'small' }) => {
  const getIcon = () => {
    switch (status) {
      case "loaded": return <CheckCircleIcon color="success" fontSize={size} />;
      case "error": return <ErrorIcon color="error" fontSize={size} />;
      default: return <WarningIcon color="warning" fontSize={size} />;
    }
  };

  return (
    <>
      {getIcon()}
      <Chip
        label={TRANSLATIONS[status] || status}
        size={size}
        variant="filled"
        color={
          status === "loaded" ? "success" :
          status === "error" ? "error" : "warning"
        }
        sx={{ ml: 0.5 }}
      />
    </>
  );
};