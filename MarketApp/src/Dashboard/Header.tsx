import React from 'react';
import {
  AppBar,
  Toolbar,
  Stack,
  Avatar,
  Box,
  Typography,
  Chip,
  IconButton,
  Tooltip,
  alpha,
  useTheme,
  CircularProgress,
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import type {HealthResponse} from '../types';

interface HeaderProps {
  health: HealthResponse | undefined;
  healthLoading: boolean;
  onRefresh: () => void;
}

export const Header: React.FC<HeaderProps> = ({ health, healthLoading, onRefresh }) => {
  const theme = useTheme();

  return (
    <AppBar
      position="static"
      elevation={0}
      sx={{
        background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.primary.dark} 100%)`,
        borderBottom: `1px solid ${alpha(theme.palette.common.white, 0.1)}`,
      }}
    >
      <Toolbar>
        <Stack direction="row" alignItems="center" spacing={2}>
          <Avatar
            sx={{
              bgcolor: "white",
              color: theme.palette.primary.main,
              width: 40,
              height: 40,
            }}
          >
            <AnalyticsIcon />
          </Avatar>
          <Box>
            <Typography variant="h6" sx={{ fontWeight: 800, color: "white" }}>
              Marketing Predictions AI
            </Typography>
            <Typography variant="caption" sx={{ color: alpha(theme.palette.common.white, 0.8) }}>
              Система прогнозування маркетингових показників
            </Typography>
          </Box>
        </Stack>

        <Box sx={{ flexGrow: 1 }} />

        <Stack direction="row" spacing={2} alignItems="center">
          <Tooltip title="Статус API">
            <Chip
              label={healthLoading ? "Перевірка..." : health?.status === "ok" ? "API активний" : "API помилка"}
              size="medium"
              color={health?.status === "ok" ? "success" : "error"}
              sx={{
                fontWeight: 600,
                color: "white",
                bgcolor: health?.status === "ok" ? alpha(theme.palette.success.main, 0.9) : alpha(theme.palette.error.main, 0.9),
              }}
              icon={healthLoading ? <CircularProgress size={16} color="inherit" /> : undefined}
            />
          </Tooltip>

          <Tooltip title="Оновити статус">
            <IconButton
              size="medium"
              onClick={onRefresh}
              sx={{
                color: "white",
                bgcolor: alpha(theme.palette.common.white, 0.1),
                '&:hover': { bgcolor: alpha(theme.palette.common.white, 0.2) },
              }}
            >
              <RefreshIcon />
            </IconButton>
          </Tooltip>
        </Stack>
      </Toolbar>
    </AppBar>
  );
};