import React from 'react';
import {
  Card,
  CardContent,
  Stack,
  Avatar,
  Typography,
  Chip,
  Box,
  LinearProgress,
  alpha,
  useTheme,
} from '@mui/material';
import AssessmentIcon from '@mui/icons-material/Assessment';

interface StatisticsCardProps {
  historyLength: number;
  lastPredictionDate?: Date;
  selectedService: string;
}

export const StatisticsCard: React.FC<StatisticsCardProps> = ({
  historyLength,
  lastPredictionDate,
  selectedService,
}) => {
  const theme = useTheme();

  return (
    <Card
      elevation={0}
      sx={{
        borderRadius: 3,
        border: `1px solid ${alpha(theme.palette.success.main, 0.1)}`,
        background: "white",
        position: "relative",
        '&:before': {
          content: '""',
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          height: 4,
          background: `linear-gradient(90deg, ${theme.palette.success.main}, ${theme.palette.info.main})`,
        }
      }}
    >
      <CardContent>
        <Stack spacing={2}>
          <Stack direction="row" alignItems="center" spacing={1.5}>
            <Avatar
              sx={{
                bgcolor: alpha(theme.palette.success.main, 0.1),
                color: theme.palette.success.main,
              }}
            >
              <AssessmentIcon />
            </Avatar>
            <Typography variant="h6" sx={{ fontWeight: 700 }}>
              Статистика
            </Typography>
          </Stack>

          <Stack spacing={2}>
            <Stack direction="row" justifyContent="space-between" alignItems="center">
              <Typography variant="body2" color="text.secondary">
                Усього прогнозів:
              </Typography>
              <Chip
                label={historyLength}
                color="primary"
                size="small"
                sx={{ fontWeight: 700 }}
              />
            </Stack>

            <Stack direction="row" justifyContent="space-between" alignItems="center">
              <Typography variant="body2" color="text.secondary">
                Останній прогноз:
              </Typography>
              <Typography variant="body2" fontWeight={600}>
                {lastPredictionDate?.toLocaleDateString('uk-UA') || "—"}
              </Typography>
            </Stack>

            <Stack direction="row" justifyContent="space-between" alignItems="center">
              <Typography variant="body2" color="text.secondary">
                Активна модель:
              </Typography>
              <Typography variant="body2" fontWeight={600} color="primary">
                {selectedService || "Не обрана"}
              </Typography>
            </Stack>

            {historyLength > 0 && (
              <Box>
                <Typography variant="body2" color="text.secondary" mb={0.5}>
                  Прогрес аналізу:
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={Math.min((historyLength / 20) * 100, 100)}
                  sx={{
                    height: 8,
                    borderRadius: 4,
                    bgcolor: alpha(theme.palette.primary.main, 0.1),
                    '& .MuiLinearProgress-bar': {
                      borderRadius: 4,
                      background: `linear-gradient(90deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                    }
                  }}
                />
              </Box>
            )}
          </Stack>
        </Stack>
      </CardContent>
    </Card>
  );
};