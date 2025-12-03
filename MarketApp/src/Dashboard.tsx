import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  AppBar,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Container,
  FormControl,
  Grid,
  IconButton,
  InputAdornment,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  type SelectChangeEvent,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Toolbar,
  Typography,
  Alert,
  useTheme,
} from "@mui/material";
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
  PieChart,
  Pie,
} from "recharts";
import RefreshIcon from "@mui/icons-material/Refresh";
import AnalyticsIcon from "@mui/icons-material/Analytics";
import TrendingUpIcon from "@mui/icons-material/TrendingUp";
import PeopleIcon from "@mui/icons-material/People";
import CalendarTodayIcon from "@mui/icons-material/CalendarToday";
import ShoppingCartIcon from "@mui/icons-material/ShoppingCart";
import EventIcon from "@mui/icons-material/Event";
import HistoryIcon from "@mui/icons-material/History";
import DeleteIcon from "@mui/icons-material/Delete";

// Типы данных
interface ServiceDetail {
  status: string;
  features: string[];
  description?: string;
}

interface ServicesResponse {
  services: string[];
  details: Record<string, ServiceDetail>;
}

interface HealthResponse {
  status: string;
  timestamp?: string;
}

interface PredictionRecord {
  total_events: number;
  total_purchases: number;
  days_since_last: number;
}

interface PredictionResult {
  predictions: any[];
  model?: string;
  timestamp?: string;
}

interface HistoryItem {
  id: string;
  timestamp: Date;
  service: string;
  input: PredictionRecord;
  output: any[];
  model: string;
}

const API_BASE = import.meta.env.VITE_API_BASE || "/api";

// Словарь перевода технических терминов на украинский
const TRANSLATIONS: Record<string, string> = {
  // Основные термины
  total_events: "Всього подій",
  total_purchases: "Всього покупок",
  days_since_last: "Днів з останньої події",
  purchase_proba: "Ймовірність покупки",
  will_purchase_pred: "Прогноз покупки",
  days_to_next_pred: "Дні до наступної покупки",
  next_purchase_amount_pred: "Сума наступної покупки",
  churn_probability: "Ймовірність відтоку",
  lifetime_value: "Життєва цінність",
  engagement_score: "Рівень залучення",

  // Статусы
  loaded: "Завантажено",
  error: "Помилка",
  loading: "Завантаження",

  // Другие термины
  predictions: "Прогнози",
  records: "Записи",
  model: "Модель",
  features: "Ознаки",
};

// Функция для получения переведенного текста
const t = (key: string): string => TRANSLATIONS[key] || key;

function Dashboard() {
  const theme = useTheme();
  const [selectedService, setSelectedService] = useState<string>("");
  const [inputData, setInputData] = useState<PredictionRecord>({
    total_events: 100,
    total_purchases: 3,
    days_since_last: 5,
  });
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [error, setError] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);

  // Цвета для графиков
  const chartColors = [
    theme.palette.primary.main,
    theme.palette.secondary.main,
    theme.palette.success.main,
    theme.palette.warning.main,
    theme.palette.error.main,
    theme.palette.info.main,
  ];

  // Запрос состояния сервиса
  const { data: health, isLoading: healthLoading } = useQuery<HealthResponse>({
    queryKey: ["health"],
    queryFn: async () => {
      const response = await fetch(`${API_BASE}/health`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    },
    retry: 2,
    refetchInterval: 30000,
  });

  // Запрос списка сервисов
  const {
    data: services,
    isLoading: servicesLoading,
    refetch: refetchServices,
  } = useQuery<ServicesResponse>({
    queryKey: ["services"],
    queryFn: async () => {
      const response = await fetch(`${API_BASE}/services`);
      if (!response.ok) throw new Error("Не вдалося отримати список сервісів");
      return response.json();
    },
  });

  // Загрузка истории из localStorage при монтировании
  useEffect(() => {
    const savedHistory = localStorage.getItem("predictionHistory");
    if (savedHistory) {
      try {
        const parsed = JSON.parse(savedHistory);
        setHistory(parsed.map((item: any) => ({
          ...item,
          timestamp: new Date(item.timestamp),
        })));
      } catch (e) {
        console.error("Помилка завантаження історії:", e);
      }
    }
  }, []);

  // Сохранение истории в localStorage
  useEffect(() => {
    if (history.length > 0) {
      localStorage.setItem("predictionHistory", JSON.stringify(history));
    }
  }, [history]);

  // Обработчик изменения сервиса
  const handleServiceChange = (event: SelectChangeEvent) => {
    setSelectedService(event.target.value);
  };

  // Обновление поля ввода
  const handleInputChange = (field: keyof PredictionRecord, value: string) => {
    const numValue = parseFloat(value);
    setInputData({
      ...inputData,
      [field]: isNaN(numValue) ? 0 : numValue,
    });
  };

  // Сброс значений к дефолтным
  const handleResetToDefault = () => {
    setInputData({
      total_events: 100,
      total_purchases: 3,
      days_since_last: 5,
    });
  };

  // Отправка данных на предсказание
  const handlePredict = async () => {
    setError("");
    setIsLoading(true);

    if (!selectedService) {
      setError("Будь ласка, спочатку виберіть модель");
      setIsLoading(false);
      return;
    }

    // Валидация данных
    if (
      isNaN(inputData.total_events) ||
      isNaN(inputData.total_purchases) ||
      isNaN(inputData.days_since_last)
    ) {
      setError("Будь ласка, введіть коректні числа у всі поля");
      setIsLoading(false);
      return;
    }

    try {
      const url = `${API_BASE}/predict?service=${selectedService}`;
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ records: [inputData] }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        setError(
          errorData.detail ||
            errorData.error ||
            `Помилка сервера: ${response.status}`
        );
        setIsLoading(false);
        return;
      }

      const result: PredictionResult = await response.json();

      // Добавляем результат в историю
      const historyItem: HistoryItem = {
        id: Date.now().toString(),
        timestamp: new Date(),
        service: selectedService,
        input: { ...inputData },
        output: result.predictions || [],
        model: result.model || selectedService,
      };

      setHistory([historyItem, ...history.slice(0, 9)]); // Храним последние 10 записей
    } catch (error: any) {
      setError(`Мережева помилка: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  // Очистка истории
  const handleClearHistory = () => {
    setHistory([]);
    localStorage.removeItem("predictionHistory");
  };

  // Получение данных для графиков
  const getPredictionData = () => {
    return history.map((item, index) => {
      const output = item.output[0] || {};
      return {
        id: `#${history.length - index}`,
        time: item.timestamp.toLocaleTimeString([], {
          hour: '2-digit',
          minute: '2-digit'
        }),
        date: item.timestamp.toLocaleDateString('uk-UA'),
        ...output,
        total_events: item.input.total_events,
        total_purchases: item.input.total_purchases,
        days_since_last: item.input.days_since_last,
      };
    }).reverse();
  };

  // Получение ключей для графиков
  const getMetricKeys = () => {
    if (history.length === 0) return [];

    const keys = new Set<string>();
    history.forEach(item => {
      const output = item.output[0] || {};
      Object.keys(output).forEach(key => {
        if (typeof output[key] === 'number') {
          keys.add(key);
        }
      });
    });

    return Array.from(keys);
  };

  // Группировка метрик по типам для разных графиков
  const getMetricsByType = () => {
    const allKeys = getMetricKeys();
    const probabilityKeys = allKeys.filter(k =>
      k.includes('proba') || k.includes('probability') || k.includes('score')
    );
    const valueKeys = allKeys.filter(k =>
      k.includes('amount') || k.includes('value') || k.includes('price')
    );
    const timeKeys = allKeys.filter(k =>
      k.includes('days') || k.includes('time') || k.includes('hours')
    );
    const predictionKeys = allKeys.filter(k =>
      k.includes('pred') || k.includes('will') || k.includes('expected')
    );

    return {
      probabilities: probabilityKeys,
      values: valueKeys,
      time: timeKeys,
      predictions: predictionKeys,
    };
  };

  // Подготовка данных для Pie Chart (последний прогноз)
  const getPieChartData = () => {
    if (history.length === 0) return [];

    const lastPrediction = history[0]?.output[0] || {};
    return Object.entries(lastPrediction)
      .filter(([_, value]) => typeof value === 'number')
      .slice(0, 5) // Берем первые 5 метрик
      .map(([key, value], index) => ({
        name: t(key),
        value: value as number,
        color: chartColors[index % chartColors.length],
      }));
  };

  const serviceOptions = services
    ? services.services.map(name => ({
        name,
        detail: services.details[name],
      }))
    : [];

  const predictionData = getPredictionData();
  const metricsByType = getMetricsByType();
  const pieChartData = getPieChartData();
  const hasData = history.length > 0;

  return (
    <Box sx={{ minHeight: "100vh", width: "100vw", bgcolor: "grey.50" }}>
      <AppBar position="static" elevation={0}>
        <Toolbar>
          <Stack direction="row" alignItems="center" spacing={2}>
            <AnalyticsIcon />
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              Система прогнозування маркетингу
            </Typography>
          </Stack>
          <Box sx={{ flexGrow: 1 }} />
          <Stack direction="row" spacing={2} alignItems="center">
            <Chip
              label={healthLoading ? "Перевірка..." : t(health?.status || "недоступно")}
              size="small"
              color={health?.status === "ok" ? "success" : "error"}
              variant="outlined"
            />
            <IconButton
              size="small"
              onClick={() => refetchServices()}
              color="inherit"
            >
              <RefreshIcon />
            </IconButton>
          </Stack>
        </Toolbar>
      </AppBar>

      <Container  sx={{ py: 3 , width: '100%'}}>
        <Grid container spacing={3}>
          {/* Левая панель - Выбор модели и ввод данных */}
          <Grid item xs={12} md={4}>
            <Stack spacing={3}>
              {/* Выбор модели */}
              <Card elevation={2}>
                <CardContent>
                  <Stack spacing={2}>
                    <Stack direction="row" alignItems="center" spacing={1}>
                      <TrendingUpIcon color="primary" />
                      <Typography variant="h6" sx={{ fontWeight: 600 }}>
                        Вибір моделі
                      </Typography>
                    </Stack>

                    <FormControl fullWidth size="medium">
                      <InputLabel>Оберіть модель</InputLabel>
                      <Select
                        value={selectedService}
                        onChange={handleServiceChange}
                        label="Оберіть модель"
                        disabled={servicesLoading}
                      >
                        {serviceOptions.map((option) => (
                          <MenuItem key={option.name} value={option.name}>
                            <Stack spacing={0.5} width="100%">
                              <Typography variant="body2" fontWeight={500}>
                                {option.name}
                              </Typography>
                              <Stack direction="row" spacing={1} alignItems="center">
                                <Chip
                                  label={t(option.detail.status)}
                                  size="small"
                                  color={option.detail.status === "loaded" ? "success" : "warning"}
                                />
                                <Typography variant="caption" color="text.secondary">
                                  {option.detail.features?.length} ознак
                                </Typography>
                              </Stack>
                            </Stack>
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  </Stack>
                </CardContent>
              </Card>

              {/* Ввод данных */}
              <Card elevation={2}>
                <CardContent>
                  <Stack spacing={3}>
                    <Stack direction="row" alignItems="center" spacing={1}>
                      <PeopleIcon color="primary" />
                      <Typography variant="h6" sx={{ fontWeight: 600 }}>
                        Дані клієнта
                      </Typography>
                    </Stack>

                    <Stack spacing={2}>
                      <TextField
                        fullWidth
                        label={t("total_events")}
                        type="number"
                        value={inputData.total_events}
                        onChange={(e) => handleInputChange("total_events", e.target.value)}
                        InputProps={{
                          startAdornment: (
                            <InputAdornment position="start">
                              <EventIcon color="action" />
                            </InputAdornment>
                          ),
                          inputProps: { min: 0, step: 1 },
                        }}
                        size="medium"
                      />

                      <TextField
                        fullWidth
                        label={t("total_purchases")}
                        type="number"
                        value={inputData.total_purchases}
                        onChange={(e) => handleInputChange("total_purchases", e.target.value)}
                        InputProps={{
                          startAdornment: (
                            <InputAdornment position="start">
                              <ShoppingCartIcon color="action" />
                            </InputAdornment>
                          ),
                          inputProps: { min: 0, step: 1 },
                        }}
                        size="medium"
                      />

                      <TextField
                        fullWidth
                        label={t("days_since_last")}
                        type="number"
                        value={inputData.days_since_last}
                        onChange={(e) => handleInputChange("days_since_last", e.target.value)}
                        InputProps={{
                          startAdornment: (
                            <InputAdornment position="start">
                              <CalendarTodayIcon color="action" />
                            </InputAdornment>
                          ),
                          inputProps: { min: 0, step: 1 },
                        }}
                        size="medium"
                      />
                    </Stack>

                    <Stack spacing={1}>
                      <Button
                        variant="contained"
                        onClick={handlePredict}
                        disabled={!selectedService || isLoading}
                        fullWidth
                        size="large"
                      >
                        {isLoading ? "Обробка..." : "Зробити прогноз"}
                      </Button>

                      <Button
                        variant="outlined"
                        onClick={handleResetToDefault}
                        fullWidth
                      >
                        За замовчуванням
                      </Button>
                    </Stack>

                    {error && (
                      <Alert severity="error" sx={{ mt: 1 }}>
                        {error}
                      </Alert>
                    )}
                  </Stack>
                </CardContent>
              </Card>

              {/* Статистика */}
              <Card elevation={2}>
                <CardContent>
                  <Stack spacing={2}>
                    <Typography variant="h6" sx={{ fontWeight: 600 }}>
                      Статистика
                    </Typography>

                    <Stack spacing={1}>
                      <Stack direction="row" justifyContent="space-between">
                        <Typography variant="body2" color="text.secondary">
                          Усього прогнозів:
                        </Typography>
                        <Typography variant="body2" fontWeight={500}>
                          {history.length}
                        </Typography>
                      </Stack>

                      <Stack direction="row" justifyContent="space-between">
                        <Typography variant="body2" color="text.secondary">
                          Останній прогноз:
                        </Typography>
                        <Typography variant="body2" fontWeight={500}>
                          {history[0]?.timestamp.toLocaleDateString('uk-UA') || "—"}
                        </Typography>
                      </Stack>

                      <Stack direction="row" justifyContent="space-between">
                        <Typography variant="body2" color="text.secondary">
                          Активна модель:
                        </Typography>
                        <Typography variant="body2" fontWeight={500}>
                          {selectedService || "Не обрана"}
                        </Typography>
                      </Stack>
                    </Stack>
                  </Stack>
                </CardContent>
              </Card>
            </Stack>
          </Grid>

          {/* Правая панель - Графики и история */}
          <Grid item xs={12} md={8}>
            <Stack spacing={3}>
              {/* 4 графика в сетке 2x2 */}
              {hasData && (
                <Grid container spacing={2}>
                  {/* График 1: Вероятности */}
                  {metricsByType.probabilities.length > 0 && (
                    <Grid item xs={12} md={6}>
                      <Card elevation={2} sx={{ height: '100%' }}>
                        <CardContent>
                          <Typography variant="subtitle1" fontWeight={600} gutterBottom>
                            Ймовірності
                          </Typography>
                          <Box sx={{ height: 200 }}>
                            <ResponsiveContainer width="100%" height="100%">
                              <LineChart data={predictionData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                                <XAxis
                                  dataKey="id"
                                  stroke="#666"
                                  fontSize={12}
                                />
                                <YAxis
                                  stroke="#666"
                                  fontSize={12}
                                />
                                <RechartsTooltip />
                                <Legend />
                                {metricsByType?.probabilities?.map((key, index) => (
                                  <Line
                                    key={key}
                                    type="monotone"
                                    dataKey={key}
                                    name={t(key)}
                                    stroke={chartColors[index % chartColors.length]}
                                    strokeWidth={2}
                                    dot={{ r: 3 }}
                                  />
                                ))}
                              </LineChart>
                            </ResponsiveContainer>
                          </Box>
                        </CardContent>
                      </Card>
                    </Grid>
                  )}

                  {/* График 2: Временные показатели */}
                  {metricsByType.time.length > 0 && (
                    <Grid item xs={12} md={6}>
                      <Card elevation={2} sx={{ height: '100%' }}>
                        <CardContent>
                          <Typography variant="subtitle1" fontWeight={600} gutterBottom>
                            Часові показники
                          </Typography>
                          <Box sx={{ height: 200 }}>
                            <ResponsiveContainer width="100%" height="100%">
                              <BarChart data={predictionData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                                <XAxis dataKey="id" stroke="#666" fontSize={12} />
                                <YAxis stroke="#666" fontSize={12} />
                                <RechartsTooltip />
                                <Legend />
                                {metricsByType.time.map((key, index) => (
                                  <Bar
                                    key={key}
                                    dataKey={key}
                                    name={t(key)}
                                    fill={chartColors[index % chartColors.length]}
                                  />
                                ))}
                              </BarChart>
                            </ResponsiveContainer>
                          </Box>
                        </CardContent>
                      </Card>
                    </Grid>
                  )}

                  {/* График 3: Финансовые показатели */}
                  {metricsByType.values.length > 0 && (
                    <Grid item xs={12} md={6}>
                      <Card elevation={2} sx={{ height: '100%' }}>
                        <CardContent>
                          <Typography variant="subtitle1" fontWeight={600} gutterBottom>
                            Фінансові показники
                          </Typography>
                          <Box sx={{ height: 200 }}>
                            <ResponsiveContainer width="100%" height="100%">
                              <AreaChart data={predictionData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                                <XAxis dataKey="id" stroke="#666" fontSize={12} />
                                <YAxis stroke="#666" fontSize={12} />
                                <RechartsTooltip />
                                <Legend />
                                {metricsByType.values.map((key, index) => (
                                  <Area
                                    key={key}
                                    type="monotone"
                                    dataKey={key}
                                    name={t(key)}
                                    stroke={chartColors[index % chartColors.length]}
                                    fill={chartColors[index % chartColors.length]}
                                    fillOpacity={0.3}
                                  />
                                ))}
                              </AreaChart>
                            </ResponsiveContainer>
                          </Box>
                        </CardContent>
                      </Card>
                    </Grid>
                  )}

                  {/* График 4: Pie Chart - распределение метрик */}
                  {pieChartData.length > 0 && (
                    <Grid item xs={12} md={6}>
                      <Card elevation={2} sx={{ height: '100%' }}>
                        <CardContent>
                          <Typography variant="subtitle1" fontWeight={600} gutterBottom>
                            Розподіл показників
                          </Typography>
                          <Box sx={{ height: 200 }}>
                            <ResponsiveContainer width="100%" height="100%">
                              <PieChart>
                                <Pie
                                  data={pieChartData}
                                  cx="50%"
                                  cy="50%"
                                  labelLine={false}
                                  label={({ name, percent }) => `${name}: ${((percent || 0)  * 100).toFixed(0)}%`}
                                  outerRadius={70}
                                  fill="#8884d8"
                                  dataKey="value"
                                >
                                  {pieChartData.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={entry.color} />
                                  ))}
                                </Pie>
                                <RechartsTooltip formatter={(value) => [value, "Значення"]} />
                              </PieChart>
                            </ResponsiveContainer>
                          </Box>
                        </CardContent>
                      </Card>
                    </Grid>
                  )}
                </Grid>
              )}

              {/* История прогнозов */}
              <Card elevation={2}>
                <CardContent>
                  <Stack spacing={2}>
                    <Stack direction="row" justifyContent="space-between" alignItems="center">
                      <Stack direction="row" alignItems="center" spacing={1}>
                        <HistoryIcon color="action" />
                        <Typography variant="h6" sx={{ fontWeight: 600 }}>
                          Історія прогнозів
                        </Typography>
                        <Chip
                          label={`${history.length} записів`}
                          size="small"
                          color="primary"
                          variant="outlined"
                        />
                      </Stack>

                      {history.length > 0 && (
                        <Button
                          variant="text"
                          size="small"
                          onClick={handleClearHistory}
                          startIcon={<DeleteIcon/>}
                        >
                          Очистити історію
                        </Button>
                      )}
                    </Stack>

                    {history.length === 0 ? (
                      <Paper
                        variant="outlined"
                        sx={{
                          p: 6,
                          textAlign: "center",
                          bgcolor: "grey.50",
                        }}
                      >
                        <Typography color="text.secondary" gutterBottom>
                          Історія прогнозів порожня
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Зробіть перший прогноз, щоб побачити історію тут
                        </Typography>
                      </Paper>
                    ) : (
                      <TableContainer component={Paper} variant="outlined">
                        <Table size="small">
                          <TableHead sx={{ bgcolor: "grey.50" }}>
                            <TableRow>
                              <TableCell sx={{ fontWeight: 600 }}>Дата</TableCell>
                              <TableCell sx={{ fontWeight: 600 }}>Час</TableCell>
                              <TableCell sx={{ fontWeight: 600 }}>Модель</TableCell>
                              <TableCell sx={{ fontWeight: 600 }}>Вхідні дані</TableCell>
                              <TableCell sx={{ fontWeight: 600 }}>Результати</TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {history.map((item) => (
                              <TableRow
                                key={item.id}
                                sx={{
                                  '&:hover': { bgcolor: 'action.hover' },
                                }}
                              >
                                <TableCell>
                                  <Typography variant="body2">
                                    {item.timestamp.toLocaleDateString('uk-UA')}
                                  </Typography>
                                </TableCell>
                                <TableCell>
                                  <Typography variant="body2">
                                    {item.timestamp.toLocaleTimeString('uk-UA', {
                                      hour: '2-digit',
                                      minute: '2-digit'
                                    })}
                                  </Typography>
                                </TableCell>
                                <TableCell>
                                  <Chip
                                    label={item.model}
                                    size="small"
                                    variant="outlined"
                                  />
                                </TableCell>
                                <TableCell>
                                  <Stack spacing={0.5}>
                                    <Typography variant="caption">
                                      <strong>{t('total_events')}:</strong> {item.input.total_events}
                                    </Typography>
                                    <Typography variant="caption">
                                      <strong>{t('total_purchases')}:</strong> {item.input.total_purchases}
                                    </Typography>
                                    <Typography variant="caption">
                                      <strong>{t('days_since_last')}:</strong> {item.input.days_since_last}
                                    </Typography>
                                  </Stack>
                                </TableCell>
                                <TableCell>
                                  {item.output[0] && (
                                    <Stack spacing={0.5}>
                                      {Object.entries(item.output[0]).slice(0, 3).map(([key, value]) => (
                                        <Typography key={key} variant="caption">
                                          <strong>{t(key)}:</strong>{" "}
                                          {typeof value === 'number'
                                            ? value.toFixed(4)
                                            : String(value)}
                                        </Typography>
                                      ))}
                                      {Object.entries(item.output[0]).length > 3 && (
                                        <Typography variant="caption" color="text.secondary">
                                          + ще {Object.entries(item.output[0]).length - 3} показників
                                        </Typography>
                                      )}
                                    </Stack>
                                  )}
                                </TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </TableContainer>
                    )}
                  </Stack>
                </CardContent>
              </Card>
            </Stack>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
}

export default Dashboard;