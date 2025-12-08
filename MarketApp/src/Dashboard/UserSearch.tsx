import React, {useMemo, useState} from 'react';
import {
    TextField,
    Stack,
    Typography,
    Box,
    Paper,
    Alert,
    Button,
    CircularProgress,
    Chip,
    InputAdornment,
    Fade, Divider,
} from '@mui/material';
import PersonSearchIcon from '@mui/icons-material/PersonSearch';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import type {User} from '../types';

interface UserSearchProps {
    users: User[];
    onSelectUserData: (user: User) => void;
    userId: string;
    setUserId: (id: string) => void;
    // Опциональные пропсы для обратной совместимости
    searchQuery?: string;
    setSearchQuery?: (query: string) => void;
    onSearchUsers?: () => void;
    usersLoading?: boolean;
}

export const UserSearch: React.FC<UserSearchProps> = ({
                                                          users,
                                                          onSelectUserData,
                                                          userId,
                                                          setUserId,
                                                          // Опциональные пропсы
                                                          searchQuery: externalSearchQuery,
                                                          setSearchQuery: externalSetSearchQuery,
                                                          onSearchUsers,
                                                          usersLoading = false,
                                                      }) => {
    // Локальное состояние для поискового запроса
    const [localSearchQuery, setLocalSearchQuery] = useState('');

    // Используем внешний или локальный searchQuery
    const searchQuery = externalSearchQuery !== undefined ? externalSearchQuery : localSearchQuery;
    const setSearchQuery = externalSetSearchQuery || setLocalSearchQuery;

    // Фильтруем пользователей локально на основе searchQuery
    const filteredUsers = useMemo(() => {
        const query = localSearchQuery.toLowerCase().trim();
        return query
            ? users.filter(u => u.user_id.toLowerCase().includes(query))
            : users;
    }, [users, localSearchQuery]);


    // Обработка выбора пользователя
    const handleUserSelect = (user: User) => {
        setUserId(user.user_id);
        onSelectUserData(user);// Очищаем поле поиска после выбора
    };

    // Обработка нажатия Enter в поле поиска
    // показываем список для выбора
    const shouldShowUserList = localSearchQuery.trim() || filteredUsers.length > 0;

    return (
        <Stack spacing={2}>
            <Box>
                <Typography variant="subtitle2" fontWeight={600} mb={1}>
                    Пошук користувача
                </Typography>
                <Stack direction="row" spacing={1}>
                    <TextField
                        fullWidth
                        placeholder="Введіть ID користувача"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        onKeyDown={(e) => {
                            if (e.key === 'Enter' && onSearchUsers) {
                                e.preventDefault();
                                onSearchUsers();
                            }
                        }}
                        size="medium"
                        sx={{'& .MuiOutlinedInput-root': {borderRadius: 2}}}
                        InputProps={{
                            startAdornment: (
                                <InputAdornment position="start">
                                    <PersonSearchIcon color="action"/>
                                </InputAdornment>
                            ),
                        }}
                    />

                    {onSearchUsers && (
                        <Button
                            variant="contained"
                            onClick={onSearchUsers}
                            disabled={usersLoading || !searchQuery.trim()}
                            sx={{
                                borderRadius: 2,
                                minWidth: 100,
                                px: 3,
                            }}
                        >
                            {usersLoading ? <CircularProgress size={24}/> : "Пошук"}
                        </Button>
                    )}
                </Stack>
            </Box>

            {/* Показываем индикатор загрузки только если есть внешний поиск */}
            {usersLoading ? (
                <Box sx={{p: 3, textAlign: 'center'}}>
                    <CircularProgress/>
                    <Typography variant="body2" color="text.secondary" mt={1}>
                        Пошук користувачів...
                    </Typography>
                </Box>
            ) : (
                /* Локальный список пользователей */
                shouldShowUserList && (
                    <Paper
                        variant="outlined"
                        sx={{
                            maxHeight: 300,
                            overflow: 'auto',
                            borderRadius: 2,
                            mt: 1,
                        }}
                    >
                        <Stack spacing={1} sx={{p: 1}}>
                            {filteredUsers.length > 0 ? (
                                filteredUsers.map(user => (
                                    <Paper
                                        key={user.user_id}
                                        variant="outlined"
                                        sx={{
                                            p: 1.5,
                                            cursor: 'pointer',
                                            borderRadius: 2,
                                            transition: 'all 0.2s',
                                            '&:hover': {
                                                bgcolor: 'action.hover',
                                                transform: 'translateY(-2px)',
                                                boxShadow: 2,
                                            }
                                        }}
                                        onClick={() => handleUserSelect(user)}
                                    >
                                        <Stack spacing={1}>
                                            <Stack direction="row" justifyContent="space-between" alignItems="center">
                                                <Typography variant="subtitle2" fontWeight={600}>
                                                    ID: {user.user_id}
                                                </Typography>
                                                <Chip label="Обрати" size="small" color="primary" variant="outlined"/>
                                            </Stack>
                                            {user.features && (
                                                <Box sx={{mt: 1}}>
                                                    {/* 1. ЖИВОСТЬ (Recency + Frequency) */}
                                                    <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap
                                                           sx={{mb: 1}}>
                                                        {/* Активность за неделю */}
                                                        <Box sx={{
                                                            bgcolor: (user.features.events_last_7d || 0) > 0 ? 'success.lighter' : 'action.hover',
                                                            px: 1, py: 0.5, borderRadius: 1,
                                                            border: '1px solid',
                                                            borderColor: (user.features.events_last_7d || 0) > 0 ? 'success.light' : 'divider'
                                                        }}>
                                                            <Typography variant="caption" fontWeight={700} color={
                                                                (user.features.events_last_7d || 0) > 0 ? 'success.main' : 'text.secondary'
                                                            }>
                                                                7d events: {user.features.events_last_7d ?? 0}
                                                            </Typography>
                                                        </Box>

                                                        {/* Давность последнего визита */}
                                                        <Box sx={{
                                                            bgcolor: (user.features.days_since_last ?? 999) < 3 ? 'success.lighter' :
                                                                (user.features.days_since_last ?? 999) > 30 ? 'error.lighter' : 'warning.lighter',
                                                            px: 1, py: 0.5, borderRadius: 1,
                                                            border: '1px solid',
                                                            borderColor: 'divider'
                                                        }}>
                                                            <Typography variant="caption" fontWeight={700} color={
                                                                (user.features.days_since_last ?? 999) < 3 ? 'success.main' :
                                                                    (user.features.days_since_last ?? 999) > 30 ? 'error.main' : 'warning.main'
                                                            }>
                                                                Recency: {user.features.days_since_last ?? '-'} дн.
                                                            </Typography>
                                                        </Box>
                                                    </Stack>

                                                    {/* 2. ДЕНЬГИ (Monetary) */}
                                                    <Stack direction="row" spacing={1.5} alignItems="center"
                                                           sx={{opacity: 0.9}}>
                                                        <Typography variant="caption" color="text.secondary" sx={{
                                                            display: 'flex',
                                                            alignItems: 'center',
                                                            gap: 0.5
                                                        }}>
                                                            🛒 <b>{user.features.total_purchases || 0}</b> (all)
                                                        </Typography>
                                                        <Divider orientation="vertical" flexItem
                                                                 sx={{height: 12, alignSelf: 'center'}}/>
                                                        <Typography variant="caption" color="text.secondary" sx={{
                                                            display: 'flex',
                                                            alignItems: 'center',
                                                            gap: 0.5
                                                        }}>
                                                            📅 <b>{user.features.purchases_last_30d || 0}</b> (30d)
                                                        </Typography>
                                                        <Divider orientation="vertical" flexItem
                                                                 sx={{height: 12, alignSelf: 'center'}}/>
                                                        <Typography variant="caption" color="text.secondary" sx={{
                                                            display: 'flex',
                                                            alignItems: 'center',
                                                            gap: 0.5
                                                        }}>
                                                            💰 <b>{user.features.total_spent?.toFixed(0) || 0}₴</b>
                                                        </Typography>
                                                    </Stack>

                                                    {/* 3. КОНТЕКСТ (Context) */}
                                                    <Stack direction="row" spacing={1.5} alignItems="center"
                                                           sx={{mt: 0.5, opacity: 0.7}}>
                                                        <Typography variant="caption" color="text.secondary">
                                                            Last: <b>{user.features.last_event_type || "-"}</b>
                                                        </Typography>
                                                        <Typography variant="caption" color="text.secondary">
                                                            Item: <b>{user.features.last_item || "-"}</b>
                                                        </Typography>
                                                    </Stack>
                                                </Box>
                                            )}
                                        </Stack>
                                    </Paper>
                                ))
                            ) : (
                                <Alert severity="info" sx={{borderRadius: 2}}>
                                    Користувачів не знайдено. Спробуйте інший запит.
                                </Alert>
                            )}
                        </Stack>
                    </Paper>

                )
            )}

            {/* Сообщение, если пользователи не найдены */}
            {searchQuery.trim() && !usersLoading && filteredUsers.length === 0 && (
                <Alert severity="info" sx={{borderRadius: 2}}>
                    Користувачів не знайдено. Спробуйте інший запит.
                </Alert>
            )}

            {/* Если выбран пользователь */}
            {userId && (
                <Fade in={!!userId}>
                    <Alert
                        severity="success"
                        sx={{borderRadius: 2}}
                        icon={<CheckCircleIcon/>}
                    >
                        Обрано користувача: {userId}
                    </Alert>
                </Fade>
            )}

            {/* Информация о количестве пользователей */}
            <Typography variant="caption" color="text.secondary">
                {users.length === 0 ? 'Немає користувачів' : `Знайдено користувачів: ${users.length}`}
                {searchQuery.trim() && filteredUsers.length > 0 && ` (відфільтровано: ${filteredUsers.length})`}
            </Typography>
        </Stack>
    );
};