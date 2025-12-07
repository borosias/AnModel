import {useState, useEffect} from "react";
import {useQuery} from "@tanstack/react-query";
import {
    Box,
    Container,
    Grid,
    Stack,
    alpha,
    useTheme, Typography,
} from "@mui/material";

import {apiClient} from "../api/client";
import {usePredictionHistory} from "../hooks/usePredictionHistory";
import {
    getPredictionData,
    getMetricKeys,
} from "../utils/chartUtils";
import type {HealthResponse, ServicesResponse, User, InputMode, Features} from "../types";

import {Header} from "./Header.tsx";
import {ModelSelectionCard} from "./ModelSelectionCard";
import {DataInputCard} from "./DataInputCard";
import {StatisticsCard} from "./StatisticsCard";
import {VisualizationCard} from "./VisualizationCard";
import {HistoryCard} from "./HistoryCard";
import {EmptyState} from "./EmptyState";

export default function Dashboard() {
    const theme = useTheme();
    const [selectedService, setSelectedService] = useState<string>("");
    const [modelFeatures, setModelFeatures] = useState<string[]>([]);
    const [inputMode, setInputMode] = useState<InputMode>('manual');
    const [inputData, setInputData] = useState<Record<string, any>>({});
    const {history, addToHistory, clearHistory} = usePredictionHistory();
    const [error, setError] = useState<string>("");
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [userId, setUserId] = useState<string>("");
    const [activeGraphTab, setActiveGraphTab] = useState<number>(0);

    const chartColors = [
        theme.palette.primary.main,
        theme.palette.secondary.main,
        theme.palette.success.main,
        theme.palette.warning.main,
        theme.palette.error.main,
        theme.palette.info.main,
    ];

    const {data: health, isLoading: healthLoading, refetch: refetchHealth} = useQuery<HealthResponse>({
        queryKey: ["health"],
        queryFn: apiClient.health,
        retry: 2,
        refetchInterval: 30000,
    });
    const {data: usersData, isLoading: usersLoading} = useQuery<{ users: User[] }, Error>({
        queryKey: ["users"],
        queryFn: () => apiClient.getUsers(),
    });

    const [users, setUsers] = useState<User[]>([]);

    useEffect(() => {
        if (usersData?.users && !usersLoading) {
            setUsers(usersData.users);
        }
    }, [usersData]);


    const {
        data: services,
        isLoading: servicesLoading,
        refetch: refetchServices,
    } = useQuery<ServicesResponse>({
        queryKey: ["services"],
        queryFn: apiClient.services,
    });

    useEffect(() => {
        if (selectedService) {
            fetchModelInfo(selectedService);
        }
    }, [selectedService]);

    const normalizePredictions = (result: any) => {
        if (!result || !Array.isArray(result.predictions)) return result;
        return {
            ...result,
            predictions: result.predictions.map((p: any) => ({
                ...p,
                days_to_next_pred:
                    p.days_to_next_pred === 999 ? 0 : p.days_to_next_pred,
            })),
        };
    };

    const fetchModelInfo = async (serviceName: string) => {
        try {
            const modelInfo = await apiClient.modelInfo(serviceName);
            setModelFeatures(modelInfo.features || []);

            const initialData: Record<string, any> = {};
            modelInfo.features?.forEach(feature => {
                initialData[feature] = modelInfo.feature_medians?.[feature] || 0;
            });
            setInputData(initialData);
        } catch (error) {
            console.error("Помилка завантаження інформації про модель:", error);
            const defaultFeatures = ["total_events", "total_purchases", "days_since_last_event"];
            setModelFeatures(defaultFeatures);

            const initialData: Record<string, any> = {};
            defaultFeatures.forEach(feature => {
                initialData[feature] = 0;
            });
            setInputData(initialData);
        }
    };

    const handleLoadUserData = async (user: User) => {
        try {
            const modelInput = modelFeatures.reduce<Record<string, any>>((acc, feature) => {
                const value = user.features?.[feature as keyof Features];

                if (
                    feature === "last_event_type" ||
                    feature === "last_item" ||
                    feature === "last_region" ||
                    feature === "snapshot_date"
                ) {
                    acc[feature] = value ?? ""; // Строковые поля: если нет значения, ставим пустую строку
                } else {
                    acc[feature] = typeof value === "number" ? value : 0; // Числовые поля: если нет значения, ставим 0
                }

                return acc;
            }, {});

            setInputData(modelInput);
            setError("");
            setIsLoading(true);

            if (!selectedService) {
                setError("Будь ласка, спочатку виберіть модель");
                setIsLoading(false);
                return;
            }

            const invalidFields = modelFeatures.filter(feature => {
                const value = modelInput[feature];
                if (
                    feature === "last_event_type" ||
                    feature === "last_item" ||
                    feature === "last_region" ||
                    feature === "snapshot_date"
                ) {
                    return value === null || value === undefined; // Строковые поля валидны даже если пустые
                } else {
                    return value === undefined || isNaN(Number(value)); // Числовые поля проверяем через Number
                }
            });

            if (invalidFields.length > 0) {
                setError(`Будь ласка, введіть коректні значення для полів: ${invalidFields.join(", ")}`);
                setIsLoading(false);
                return;
            }

            const rawResult = await apiClient.predict(selectedService, [inputData]);
            const result = normalizePredictions(rawResult);

            addToHistory({
                id: Date.now().toString(),
                timestamp: new Date(),
                service: selectedService,
                input: {...modelInput},
                output: result.predictions || [],
                model: result.model || selectedService,
                user_id: user.user_id || undefined,
            });
        } catch (error: any) {
            setError(`Помилка завантаження даних: ${error.message}`);
        } finally {
            setIsLoading(false);
        }
    };


    const handlePredict = async () => {
        setError("");
        setIsLoading(true);

        if (!selectedService) {
            setError("Будь ласка, спочатку виберіть модель");
            setIsLoading(false);
            return;
        }

        // Корректная валидация: строковые last_* поля не считаем числами
        const invalidFields = modelFeatures.filter(feature => {
            const value = inputData[feature];

            if (
                feature === "last_event_type" ||
                feature === "last_item" ||
                feature === "last_region" ||
                feature === "snapshot_date"
            ) {
                // Для строковых полей достаточно, чтобы значение вообще было (может быть пустой строкой)
                return value === null || value === undefined;
            }

            // Для числовых полей проверяем, что можно привести к числу
            return value === undefined || isNaN(Number(value));
        });

        if (invalidFields.length > 0) {
            setError(`Будь ласка, введіть коректні значення для полів: ${invalidFields.join(", ")}`);
            setIsLoading(false);
            return;
        }

        try {
            const rawResult = await apiClient.predict(selectedService, [inputData]);
            const result = normalizePredictions(rawResult);

            addToHistory({
                id: Date.now().toString(),
                timestamp: new Date(),
                service: selectedService,
                input: {...inputData},
                output: result.predictions || [],
                model: result.model || selectedService,
                user_id: userId || undefined,
            });
        } catch (error: any) {
            setError(`Мережева помилка: ${error.message}`);
        } finally {
            setIsLoading(false);
        }
    };

    const handleResetToDefault = () => {
        const defaultData: Record<string, any> = {};
        modelFeatures.forEach(feature => {
            if (feature === "total_events") defaultData[feature] = 100;
            else if (feature === "total_purchases") defaultData[feature] = 3;
            else if (feature === "days_since_last_event") defaultData[feature] = 5;
            else if (feature === "avg_spend_per_purchase_30d") defaultData[feature] = 1200;
            else defaultData[feature] = 0;
        });
        setInputData(defaultData);
    };

    const serviceOptions = services
        ? services.services.map(name => ({
            name,
            detail: services.details[name],
        }))
        : [];

    const predictionData = getPredictionData(history);
    const metricKeys = getMetricKeys(history);
    const hasData = history.length > 0;

    return (
        <Box sx={{
            minHeight: "100vh",
            width: "100vw",
            bgcolor: "grey.50",
            background: `linear-gradient(135deg, ${alpha(theme.palette.primary.light, 0.05)} 0%, ${alpha(theme.palette.secondary.light, 0.05)} 100%)`,
        }}>
            <Header
                health={health}
                healthLoading={healthLoading}
                onRefresh={() => {
                    refetchHealth();
                    refetchServices();
                }}
            />

            <Container sx={{py: 3, width: "100%"}}>
                <Grid container spacing={3}>
                    {/* @ts-ignore */}
                    <Grid item xs={12} md={12} sx={{width: "100%"}}>
                        <Stack spacing={3}>
                            <ModelSelectionCard
                                services={serviceOptions}
                                selectedService={selectedService}
                                onServiceChange={(e) => setSelectedService(e.target.value)}
                                servicesLoading={servicesLoading}
                                modelFeatures={modelFeatures}
                            />

                            <DataInputCard
                                inputMode={inputMode}
                                setInputMode={setInputMode}
                                modelFeatures={modelFeatures}
                                inputData={inputData}
                                onInputChange={(field: any, value: string) =>
                                    setInputData(prev => {
                                        const isLast = String(field).toLowerCase().startsWith('last');
                                        return {
                                            ...prev,
                                            [field]: isLast ? (value || '') : (parseFloat(value) || 0),
                                        };
                                    })
                                }
                                onResetToDefault={handleResetToDefault}
                                users={users}
                                usersLoading={usersLoading}
                                onSelectUserData={handleLoadUserData}
                                userId={userId}
                                setUserId={setUserId}
                                selectedService={selectedService}
                                isLoading={isLoading}
                                onPredict={handlePredict}
                                error={error}
                                setError={setError}
                            />

                            <StatisticsCard
                                historyLength={history.length}
                                lastPredictionDate={history[0]?.timestamp}
                                selectedService={selectedService}
                            />
                        </Stack>
                    </Grid>
                    {/* @ts-ignore */}
                    <Grid item xs={12} md={12} sx={{m: "0 auto"}}>
                        <Stack spacing={3}>
                            {hasData ? (
                                <>
                                    <VisualizationCard
                                        activeGraphTab={activeGraphTab}
                                        setActiveGraphTab={setActiveGraphTab}
                                        predictionData={predictionData}
                                        metricKeys={metricKeys}
                                        chartColors={chartColors}
                                    />

                                    <HistoryCard
                                        history={history}
                                        onClearHistory={clearHistory}
                                    />
                                </>
                            ) : (
                                <EmptyState
                                    selectedService={selectedService}
                                    modelFeatures={modelFeatures}
                                    onResetToDefault={handleResetToDefault}
                                    setError={setError}
                                />
                            )}
                        </Stack>
                    </Grid>
                </Grid>
            </Container>

            <Box sx={{
                display: "flex",
                py: 2,
                px: 3,
                bgcolor: alpha(theme.palette.primary.main, 0.05),
                borderTop: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                mt: 3,
                justifyContent: "center",
            }}>
                <Container maxWidth="lg" sx={{m: "0 auto"}}>
                    <Stack direction="row" justifyContent="space-between" alignItems="center">
                        <Typography variant="caption" color="text.contrastText">
                            © {new Date().getFullYear()} Marketing Predictions AI v2.0
                        </Typography>
                        <Stack direction="row" spacing={2}>
                            <Typography variant="caption" color="text.contrastText">
                                API: {import.meta.env.VITE_API_BASE || "http://localhost:8000"}
                            </Typography>
                            <Typography variant="caption" color="text.contrastText">
                                Моделей: {services?.services?.length || 0}
                            </Typography>
                        </Stack>
                    </Stack>
                </Container>
            </Box>
        </Box>
    );
}