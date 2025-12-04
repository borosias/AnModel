import React from 'react';
import {
  Card,
  CardContent,
  Stack,
  Avatar,
  Typography,
  Tabs,
  Tab,
  Box,
  alpha,
  useTheme,
} from '@mui/material';
import TimelineIcon from '@mui/icons-material/Timeline';
import EqualizerIcon from '@mui/icons-material/Equalizer';
import AreaChartIcon from '@mui/icons-material/AreaChart';
import PieChartIcon from '@mui/icons-material/PieChart';
import RadarIcon from '@mui/icons-material/Radar';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  AreaChart,
  Area,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  Legend,
  Cell,
  PieChart as RechartsPieChart,
  Pie,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from 'recharts';
import { TRANSLATIONS } from '../constants';

interface GraphTab {
  label: string;
  icon: React.ReactElement;
}

interface VisualizationCardProps {
  activeGraphTab: number;
  setActiveGraphTab: (tab: number) => void;
  predictionData: any[];
  metricKeys: string[];
  pieChartData: any[];
  radarChartData: any[];
  chartColors: string[];
}

export const VisualizationCard: React.FC<VisualizationCardProps> = ({
  activeGraphTab,
  setActiveGraphTab,
  predictionData,
  metricKeys,
  pieChartData,
  radarChartData,
  chartColors,
}) => {
  const theme = useTheme();

  const graphTabs: GraphTab[] = [
    { label: "Лінійний графік", icon: <TimelineIcon /> },
    { label: "Стовпчикова діаграма", icon: <EqualizerIcon /> },
    { label: "Обласний графік", icon: <AreaChartIcon /> },
    { label: "Кругова діаграма", icon: <PieChartIcon /> },
    { label: "Радарна діаграма", icon: <RadarIcon /> },
  ];

  const renderChart = () => {
    switch (activeGraphTab) {
      case 0:
        return (
          <LineChart data={predictionData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis
              dataKey="id"
              label={{ value: 'Прогнози', position: 'insideBottom', offset: -5 }}
              stroke="#666"
            />
            <YAxis stroke="#666" />
            <RechartsTooltip
              contentStyle={{
                borderRadius: 8,
                border: 'none',
                boxShadow: theme.shadows[3],
              }}
              formatter={(value: number, name: string) => [
                typeof value === 'number' ? value.toFixed(3) : value,
                TRANSLATIONS[name] || name,
              ]}
            />
            <Legend />
            {metricKeys.slice(0, 3).map((key, index) => (
              <Line
                key={key}
                type="monotone"
                dataKey={key}
                name={TRANSLATIONS[key] || key}
                stroke={chartColors[index % chartColors.length]}
                strokeWidth={3}
                dot={{ r: 4 }}
                activeDot={{ r: 8, strokeWidth: 2 }}
              />
            ))}
          </LineChart>
        );

      case 1:
        return (
          <BarChart data={predictionData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="id" stroke="#666" />
            <YAxis stroke="#666" />
            <RechartsTooltip
              contentStyle={{
                borderRadius: 8,
                border: 'none',
                boxShadow: theme.shadows[3],
              }}
              formatter={(value: number, name: string) => [
                typeof value === 'number' ? value.toFixed(3) : value,
                TRANSLATIONS[name] || name,
              ]}
            />
            <Legend />
            {metricKeys.slice(0, 3).map((key, index) => (
              <Bar
                key={key}
                dataKey={key}
                name={TRANSLATIONS[key] || key}
                fill={chartColors[index % chartColors.length]}
                radius={[4, 4, 0, 0]}
              />
            ))}
          </BarChart>
        );

      case 2:
        return (
          <AreaChart data={predictionData}>
            <defs>
              {metricKeys.slice(0, 3).map((key, index) => (
                <linearGradient key={key} id={`color${key}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={chartColors[index % chartColors.length]} stopOpacity={0.8}/>
                  <stop offset="95%" stopColor={chartColors[index % chartColors.length]} stopOpacity={0}/>
                </linearGradient>
              ))}
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="id" stroke="#666" />
            <YAxis stroke="#666" />
            <RechartsTooltip
              contentStyle={{
                borderRadius: 8,
                border: 'none',
                boxShadow: theme.shadows[3],
              }}
              formatter={(value: number, name: string) => [
                typeof value === 'number' ? value.toFixed(3) : value,
                TRANSLATIONS[name] || name,
              ]}
            />
            <Legend />
            {metricKeys.slice(0, 3).map((key, index) => (
              <Area
                key={key}
                type="monotone"
                dataKey={key}
                name={TRANSLATIONS[key] || key}
                stroke={chartColors[index % chartColors.length]}
                fillOpacity={1}
                fill={`url(#color${key})`}
              />
            ))}
          </AreaChart>
        );

      case 3:
        return pieChartData.length > 0 ? (
          <RechartsPieChart>
            <Pie
              data={pieChartData}
              cx="50%"
              cy="50%"
              labelLine={true}
              label={({ name, percent }) => `${name}: ${((percent || 0) * 100).toFixed(1)}%`}
              outerRadius={120}
              fill="#8884d8"
              dataKey="value"
            >
              {pieChartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.fill || chartColors[index % chartColors.length]} />
              ))}
            </Pie>
            <RechartsTooltip
              formatter={(value: number, name: string) => [
                typeof value === 'number' ? value.toFixed(3) : value,
                name,
              ]}
              contentStyle={{
                borderRadius: 8,
                border: 'none',
                boxShadow: theme.shadows[3],
              }}
            />
            <Legend />
          </RechartsPieChart>
        ) : (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
            <Typography color="text.secondary">Немає даних для кругової діаграми</Typography>
          </Box>
        );

      case 4:
        return radarChartData.length > 0 ? (
          <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarChartData}>
            <PolarGrid />
            <PolarAngleAxis dataKey="subject" />
            <PolarRadiusAxis />
            <Radar
              name="Останній прогноз"
              dataKey="value"
              stroke={theme.palette.primary.main}
              fill={theme.palette.primary.main}
              fillOpacity={0.6}
            />
            <RechartsTooltip
              contentStyle={{
                borderRadius: 8,
                border: 'none',
                boxShadow: theme.shadows[3],
              }}
            />
            <Legend />
          </RadarChart>
        ) : (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
            <Typography color="text.secondary">Немає даних для радарної діаграми</Typography>
          </Box>
        );

      default:
        return null;
    }
  };

  return (
    <Card
      elevation={0}
      sx={{
        borderRadius: 3,
        border: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
        background: "white",
      }}
    >
      <CardContent>
        <Stack spacing={2}>
          <Stack direction="row" justifyContent="space-between" alignItems="center">
            <Stack direction="row" alignItems="center" spacing={1.5}>
              <Avatar
                sx={{
                  bgcolor: alpha(theme.palette.primary.main, 0.1),
                  color: theme.palette.primary.main,
                }}
              >
                <TimelineIcon />
              </Avatar>
              <Typography variant="h6" sx={{ fontWeight: 700 }}>
                Візуалізація результатів
              </Typography>
            </Stack>

            <Tabs
              value={activeGraphTab}
              onChange={(_, newValue) => setActiveGraphTab(newValue)}
              variant="scrollable"
              scrollButtons="auto"
              sx={{
                '& .MuiTab-root': {
                  textTransform: 'none',
                  fontWeight: 600,
                  minHeight: 48,
                  minWidth: 100,
                }
              }}
            >
              {graphTabs.map((tab, index) => (
                <Tab
                  key={index}
                  label={tab.label}
                  icon={tab.icon}
                  iconPosition="start"
                />
              ))}
            </Tabs>
          </Stack>

          <Box sx={{ height: 350, mt: 1 }}>
            <ResponsiveContainer width="100%" height="100%">
              {renderChart()}
            </ResponsiveContainer>
          </Box>
        </Stack>
      </CardContent>
    </Card>
  );
};