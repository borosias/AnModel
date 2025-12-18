import React from 'react';
import {
  Card,
  CardContent,
  Stack,
  Avatar,
  Box,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
    type SelectChangeEvent,
  Paper,
  alpha,
  useTheme,
} from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import InfoIcon from '@mui/icons-material/Info';
import type {ServiceDetail} from '../types';
import { TRANSLATIONS } from '../constants';
import { StatusIcon } from './StatusIcon';

interface ServiceOption {
  name: string;
  detail: ServiceDetail;
}

interface ModelSelectionCardProps {
  services: ServiceOption[];
  selectedService: string;
  onServiceChange: (event: SelectChangeEvent) => void;
  servicesLoading: boolean;
  modelFeatures: string[];
}

export const ModelSelectionCard: React.FC<ModelSelectionCardProps> = ({
  services,
  selectedService,
  onServiceChange,
  servicesLoading,
  modelFeatures,
}) => {
  const theme = useTheme();
  // If there is only one available service, display static model info instead of a selectable dropdown.
  if (services.length <= 1) {
    const only = services[0];
    return (
      <Card
        elevation={0}
        sx={{
          borderRadius: 3,
          border: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
          background: "white",
          overflow: "hidden",
          position: "relative",
          '&:before': {
            content: '""',
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            height: 4,
            background: `linear-gradient(90deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
          },
        }}
      >
        <CardContent>
          <Stack spacing={2.5}>
            <Stack direction="row" alignItems="center" spacing={1.5}>
              <Avatar
                sx={{
                  bgcolor: alpha(theme.palette.primary.main, 0.1),
                  color: theme.palette.primary.main,
                }}
              >
                <TrendingUpIcon />
              </Avatar>
              <Box>
                <Typography variant="h6" sx={{ fontWeight: 700, color: "text.primary" }}>
                  Модель AI
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Використовується єдина доступна модель
                </Typography>
              </Box>
            </Stack>
            <Stack direction="row" alignItems="center" spacing={1}>
              <Typography variant="body1" fontWeight={600}>
                {only?.name}
              </Typography>
              {only?.detail && (
                <StatusIcon status={only.detail.status} />
              )}
              <Typography variant="caption" color="text.secondary">
                {only?.detail?.features?.length || 0} характеристик
              </Typography>
            </Stack>
            {only && modelFeatures.length > 0 && (
              <Paper
                variant="outlined"
                sx={{
                  p: 2,
                  borderRadius: 2,
                  bgcolor: alpha(theme.palette.info.light, 0.05),
                  borderColor: alpha(theme.palette.info.main, 0.2),
                }}
              >
                <Stack direction="row" spacing={1} alignItems="center" mb={1}>
                  <InfoIcon color="info" fontSize="small" />
                  <Typography variant="subtitle2" color="info.main">
                    Активна модель: {only.name}
                  </Typography>
                </Stack>
                <Typography variant="body2" color="text.secondary">
                  <strong>Характеристики:</strong> {modelFeatures.map(f => TRANSLATIONS[f] || f).join(", ")}
                </Typography>
              </Paper>
            )}
          </Stack>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card
      elevation={0}
      sx={{
        borderRadius: 3,
        border: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
        background: "white",
        overflow: "hidden",
        position: "relative",
        '&:before': {
          content: '""',
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          height: 4,
          background: `linear-gradient(90deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
        },
      }}
    >
      <CardContent>
        <Stack spacing={2.5}>
          <Stack direction="row" alignItems="center" spacing={1.5}>
            <Avatar
              sx={{
                bgcolor: alpha(theme.palette.primary.main, 0.1),
                color: theme.palette.primary.main,
              }}
            >
              <TrendingUpIcon />
            </Avatar>
            <Box>
              <Typography variant="h6" sx={{ fontWeight: 700, color: "text.primary" }}>
                Вибір моделі AI
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Оберіть алгоритм для аналізу даних
              </Typography>
            </Box>
          </Stack>

          <FormControl fullWidth size="medium">
            <InputLabel sx={{ fontWeight: 600 }}>Оберіть модель</InputLabel>
            <Select
              value={selectedService}
              onChange={onServiceChange}
              label="Оберіть модель"
              disabled={servicesLoading}
              sx={{
                borderRadius: 2,
                '& .MuiSelect-select': { py: 1.5 },
              }}
            >
              {services.map((option) => (
                <MenuItem key={option.name} value={option.name}>
                  <Stack spacing={1} width="100%">
                    <Stack direction="row" justifyContent="space-between" alignItems="center">
                      <Typography variant="body1" fontWeight={600}>
                        {option.name}
                      </Typography>
                      <Stack direction="row" spacing={1} alignItems="center">
                        <StatusIcon status={option.detail.status} />
                      </Stack>
                    </Stack>
                    <Typography variant="caption" color="text.secondary">
                      {option.detail.features?.length || 0} характеристик
                    </Typography>
                  </Stack>
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {selectedService && modelFeatures.length > 0 && (
            <Paper
              variant="outlined"
              sx={{
                p: 2,
                borderRadius: 2,
                bgcolor: alpha(theme.palette.info.light, 0.05),
                borderColor: alpha(theme.palette.info.main, 0.2),
              }}
            >
              <Stack direction="row" spacing={1} alignItems="center" mb={1}>
                <InfoIcon color="info" fontSize="small" />
                <Typography variant="subtitle2" color="info.main">
                  Активна модель: {selectedService}
                </Typography>
              </Stack>
              <Typography variant="body2" color="text.secondary">
                <strong>Характеристики:</strong> {modelFeatures.map(f => TRANSLATIONS[f] || f).join(", ")}
              </Typography>
            </Paper>
          )}
        </Stack>
      </CardContent>
    </Card>
  );
};