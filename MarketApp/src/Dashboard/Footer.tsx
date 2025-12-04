import React from 'react';
import {
  Box,
  Container,
  Stack,
  Typography,
  alpha,
  useTheme,
} from '@mui/material';

interface FooterProps {
  servicesCount?: number;
}

export const Footer: React.FC<FooterProps> = ({ servicesCount = 0 }) => {
  const theme = useTheme();

  return (
    <Box sx={{
      py: 2,
      px: 3,
      bgcolor: alpha(theme.palette.primary.main, 0.05),
      borderTop: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
      mt: 3,
    }}>
      <Container maxWidth="xl">
        <Stack direction="row" justifyContent="space-between" alignItems="center">
          <Typography variant="caption" color="text.secondary">
            © {new Date().getFullYear()} Marketing Predictions AI v2.0
          </Typography>
          <Stack direction="row" spacing={2}>
            <Typography variant="caption" color="text.secondary">
              API: {import.meta.env.VITE_API_BASE || "http://localhost:8000"}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Моделей: {servicesCount}
            </Typography>
          </Stack>
        </Stack>
      </Container>
    </Box>
  );
};