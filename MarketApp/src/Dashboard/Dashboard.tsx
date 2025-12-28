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

    // Automatically select the only available service. When there is exactly one model
    // available from the backend, this effect assigns it to the selectedService state.
    useEffect(() => {
        if (!servicesLoading && services && services.services?.length === 1 && !selectedService) {
            const firstService = services.services[0];
            setSelectedService(firstService);
        }
    }, [services, servicesLoading, selectedService]);

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
            console.error("Error loading model info:", error);
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
            // 1. Form a correct data object from the user
            const modelInput = modelFeatures.reduce<Record<string, any>>((acc, feature) => {
                const value = user.features?.[feature as keyof Features];

                if (
                    feature === "last_event_type" ||
                    feature === "last_item" ||
                    feature === "last_region" ||
                    feature === "snapshot_date"
                ) {
                    acc[feature] = value ?? "";
                } else {
                    acc[feature] = typeof value === "number" ? value : 0;
                }

                return acc;
            }, {});

            // 2. Update UI (asynchronously)
            setInputData(modelInput);
            setUserId(user.user_id); // Important: update user ID
            setError("");
            setIsLoading(true);

            if (!selectedService) {
                setError("Please select a model first");
                setIsLoading(false);
                return;
            }

            // 3. Validation of the local variable modelInput (not state)
            const invalidFields = modelFeatures.filter(feature => {
                const value = modelInput[feature];
                if (
                    feature === "last_event_type" ||
                    feature === "last_item" ||
                    feature === "last_region" ||
                    feature === "snapshot_date"
                ) {
                    return value === null || value === undefined;
                } else {
                    return value === undefined || isNaN(Number(value));
                }
            });

            if (invalidFields.length > 0) {
                setError(`Please enter correct values for fields: ${invalidFields.join(", ")}`);
                setIsLoading(false);
                return;
            }

            // 4. PREDICT: Use modelInput, as inputData hasn't updated yet!
        } catch (error: any) {
            setError(`Error loading data: ${error.message}`);
        } finally {
            setIsLoading(false);
        }
    };

    const handlePredict = async () => {
        setError("");
        setIsLoading(true);

        if (!selectedService) {
            setError("Please select a model first");
            setIsLoading(false);
            return;
        }

        // Correct validation: string last_* fields are not considered numbers
        const invalidFields = modelFeatures.filter(feature => {
            const value = inputData[feature];

            if (
                feature === "last_event_type" ||
                feature === "last_item" ||
                feature === "last_region" ||
                feature === "snapshot_date"
            ) {
                // For string fields, it's enough that the value exists (can be an empty string)
                return value === null || value === undefined;
            }

            // For numeric fields, check if it can be converted to a number
            return value === undefined || isNaN(Number(value));
        });

        if (invalidFields.length > 0) {
            setError(`Please enter correct values for fields: ${invalidFields.join(", ")}`);
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
            setError(`Network error: ${error.message}`);
        } finally {
            setIsLoading(false);
        }
    };

    const handleResetToDefault = () => {
        const defaultData: Record<string, any> = {};

        modelFeatures.forEach(feature => {
            switch (feature) {
                // Basic
                case "total_events":
                    defaultData[feature] = 150;
                    break;
                case "total_purchases":
                    defaultData[feature] = 5;
                    break;
                case "total_spent":
                    defaultData[feature] = 5000;
                    break;
                case "days_since_last":
                    defaultData[feature] = 1;
                    break; // Active user
                case "days_since_first":
                    defaultData[feature] = 60;
                    break;

                // Rolling (Important for the new model)
                case "events_last_7d":
                    defaultData[feature] = 20;
                    break;
                case "events_last_30d":
                    defaultData[feature] = 50;
                    break;
                case "purchases_last_30d":
                    defaultData[feature] = 1;
                    break;
                case "spent_last_30d":
                    defaultData[feature] = 1000;
                    break;

                // Trends
                case "trend_popularity_mean":
                    defaultData[feature] = 50;
                    break;

                // String
                case "last_event_type":
                    defaultData[feature] = "click";
                    break;
                case "last_region":
                    defaultData[feature] = "UA-30";
                    break;

                default:
                    defaultData[feature] = 0;
            }
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
                                Models: {services?.services?.length || 0}
                            </Typography>
                        </Stack>
                    </Stack>
                </Container>
            </Box>
        </Box>
    );
}