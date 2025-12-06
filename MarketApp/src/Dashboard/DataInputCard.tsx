import React from 'react';
import {
    Card,
    CardContent,
    Stack,
    Avatar,
    Box,
    Typography,
    Tabs,
    Tab,
    Divider,
    Button,
    Alert,
    Fade,
    alpha,
    useTheme,
} from '@mui/material';
import PeopleIcon from '@mui/icons-material/People';
import KeyboardIcon from '@mui/icons-material/Keyboard';
import DatabaseIcon from '@mui/icons-material/Storage';
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh';
import CalculateIcon from '@mui/icons-material/Calculate';
import CircularProgress from '@mui/material/CircularProgress';
import type {User, InputMode} from '../types';
import {ManualInput} from './ManualInput';
import {UserSearch} from './UserSearch';

interface DataInputCardProps {
    inputMode: InputMode;
    setInputMode: (mode: InputMode) => void;
    modelFeatures: string[];
    inputData: Record<string, any>;
    onInputChange: (field: string, value: string) => void;
    onResetToDefault: () => void;
    users: User[];
    usersLoading: boolean;
    onSelectUserData: (user: User) => void;
    userId: string;
    setUserId: (id: string) => void;
    selectedService: string;
    isLoading: boolean;
    onPredict: () => void;
    error: string;
    setError: (error: string) => void;
}

export const DataInputCard: React.FC<DataInputCardProps> = ({
                                                                inputMode,
                                                                setInputMode,
                                                                modelFeatures,
                                                                inputData,
                                                                onInputChange,
                                                                onResetToDefault,
                                                                users,
                                                                userId,
                                                                setUserId,
                                                                selectedService,
                                                                usersLoading,
                                                                onSelectUserData,
                                                                isLoading,
                                                                onPredict,
                                                                error,
                                                                setError,
                                                            }) => {
    const theme = useTheme();

    return (
        <Card
            elevation={0}
            sx={{
                borderRadius: 3,
                border: `1px solid ${alpha(theme.palette.secondary.main, 0.1)}`,
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
                    background: `linear-gradient(90deg, ${theme.palette.secondary.main}, ${theme.palette.success.main})`,
                }
            }}
        >
            <CardContent>
                <Stack spacing={3}>
                    <Stack direction="row" alignItems="center" spacing={1.5}>
                        <Avatar
                            sx={{
                                bgcolor: alpha(theme.palette.secondary.main, 0.1),
                                color: theme.palette.secondary.main,
                            }}
                        >
                            <PeopleIcon/>
                        </Avatar>
                        <Box>
                            <Typography variant="h6" sx={{fontWeight: 700}}>
                                Введення даних
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                                Введіть дані для аналізу
                            </Typography>
                        </Box>
                    </Stack>

                    <Tabs
                        value={inputMode}
                        onChange={(_, newValue) => setInputMode(newValue)}
                        variant="fullWidth"
                        sx={{
                            mb: 2,
                            '& .MuiTab-root': {
                                borderRadius: 2,
                                py: 1.5,
                                fontWeight: 600,
                                textTransform: 'none',
                                '&.Mui-selected': {
                                    bgcolor: inputMode === 'manual'
                                        ? alpha(theme.palette.primary.main, 0.1)
                                        : alpha(theme.palette.secondary.main, 0.1),
                                    color: inputMode === 'manual'
                                        ? theme.palette.primary.main
                                        : theme.palette.secondary.main,
                                }
                            }
                        }}
                    >
                        <Tab
                            icon={<KeyboardIcon/>}
                            label="Ручний ввід"
                            value="manual"
                            iconPosition="start"
                        />
                        <Tab
                            icon={<DatabaseIcon/>}
                            label="Пошук користувача"
                            value="db"
                            iconPosition="start"
                        />
                    </Tabs>

                    {inputMode === 'db' ? (
                        <UserSearch
                            users={users}
                            usersLoading={usersLoading}
                            onSelectUserData={onSelectUserData}
                            userId={userId}
                            setUserId={setUserId}
                        />
                    ) : (
                        <ManualInput
                            modelFeatures={modelFeatures}
                            inputData={inputData}
                            onInputChange={onInputChange}
                        />
                    )}

                    <Divider sx={{my: 1}}/>

                    <Stack spacing={1.5}>
                        <Button
                            variant="contained"
                            onClick={onPredict}
                            disabled={!selectedService || isLoading || modelFeatures.length === 0}
                            fullWidth
                            size="large"
                            startIcon={<AutoFixHighIcon/>}
                            sx={{
                                borderRadius: 2,
                                py: 1.5,
                                fontWeight: 700,
                                fontSize: '1rem',
                                background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.primary.dark})`,
                                '&:hover': {
                                    transform: 'translateY(-2px)',
                                    boxShadow: theme.shadows[4],
                                },
                                transition: 'all 0.2s',
                            }}
                        >
                            {isLoading ? (
                                <>
                                    <CircularProgress size={24} color="inherit" sx={{mr: 1}}/>
                                    Аналіз...
                                </>
                            ) : (
                                "Запустити аналіз"
                            )}
                        </Button>

                        <Button
                            variant="outlined"
                            onClick={onResetToDefault}
                            fullWidth
                            disabled={modelFeatures.length === 0}
                            startIcon={<CalculateIcon/>}
                            sx={{
                                borderRadius: 2,
                                py: 1.5,
                                fontWeight: 600,
                            }}
                        >
                            Значення за замовчуванням
                        </Button>
                    </Stack>

                    {error && (
                        <Fade in={!!error}>
                            <Alert
                                severity="error"
                                sx={{borderRadius: 2}}
                                onClose={() => setError("")}
                            >
                                {error}
                            </Alert>
                        </Fade>
                    )}
                </Stack>
            </CardContent>
        </Card>
    );
};