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
    Fade,
    Divider,
} from '@mui/material';
import PersonSearchIcon from '@mui/icons-material/PersonSearch';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import type {User} from '../types';

interface UserSearchProps {
    users: User[];
    onSelectUserData: (user: User) => void;
    userId: string;
    setUserId: (id: string) => void;
    searchQuery?: string;
    setSearchQuery?: (query: string) => void;
    onSearchUsers?: () => void;
    usersLoading?: boolean;
}

type SegmentId = 'hot' | 'warm' | 'cold' | 'ignore';

interface SegmentInfo {
    id: SegmentId;
    label: string;
    color: 'success' | 'info' | 'warning' | 'error';
    badgeColor: string;
    bg: string;
    reason: string;
}

/**
 * Классификация пользователя по вероятности покупки + базовым фичам.
 */
function getSegment(features: Record<string, any>): SegmentInfo {
    const p = Number(features.purchase_proba ?? 0);
    const w = Number(features.will_purchase_pred ?? 0);
    const events7 = Number(features.events_last_7d ?? 0);
    const daysSinceLast = Number(features.days_since_last ?? 999);

    // Hot — высокая вероятность и недавняя активность
    if (p >= 0.7 && daysSinceLast <= 30) {
        return {
            id: 'hot',
            label: '🔥 Гаряча аудиторія',
            color: 'success',
            badgeColor: '#2e7d32',
            bg: '#e8f5e9',
            reason: `Модель дає ${(p * 100).toFixed(0)}% шанс покупки. Остання активність ${isNaN(daysSinceLast) ? '-' : daysSinceLast} дн. тому, подій за 7д: ${events7}.`,
        };
    }

    // Warm — средняя вероятность, но есть жизнь
    if (p >= 0.3 && (events7 >= 3 || daysSinceLast <= 14)) {
        return {
            id: 'warm',
            label: '⚡ Перспективний',
            color: 'info',
            badgeColor: '#0288d1',
            bg: '#e3f2fd',
            reason: `Шанс покупки ${(p * 100).toFixed(0)}%. Є недавня активність і/або історія покупок.`,
        };
    }

    // Cold — есть какой-то шанс, но слабые сигналы
    if (p >= 0.1) {
        return {
            id: 'cold',
            label: '🟠 Слабкий інтерес',
            color: 'warning',
            badgeColor: '#f57c00',
            bg: '#fff3e0',
            reason: `Невисокий шанс (${(p * 100).toFixed(0)}%). Можна включати в масові кампанії, але не в пріоритетні.`,
        };
    }

    // Ignore — очень низкая вероятность и/или нет активности
    if (p < 0.1 && w === 0) {
        return {
            id: 'ignore',
            label: '⛔ Нецільовий зараз',
            color: 'error',
            badgeColor: '#c62828',
            bg: '#ffebee',
            reason: `Модель бачить дуже низький шанс (${(p * 100).toFixed(1)}%). Або немає активності, або давно не заходив.`,
        };
    }

    // Дефолт — на всякий случай
    return {
        id: 'cold',
        label: '🟠 Слабкий інтерес',
        color: 'warning',
        badgeColor: '#f57c00',
        bg: '#fff3e0',
        reason: `Шанс покупки ${(p * 100).toFixed(0)}%.`,
    };
}

export const UserSearch: React.FC<UserSearchProps> = ({
    users,
    onSelectUserData,
    userId,
    setUserId,
    searchQuery: externalSearchQuery,
    setSearchQuery: externalSetSearchQuery,
    onSearchUsers,
    usersLoading = false,
}) => {
    const [localSearchQuery, setLocalSearchQuery] = useState('');

    const searchQuery = externalSearchQuery !== undefined ? externalSearchQuery : localSearchQuery;
    const setSearchQuery = externalSetSearchQuery || setLocalSearchQuery;

    const filteredUsers = useMemo(() => {
        const query = (searchQuery || '').toLowerCase().trim();
        const base = users.slice();

        // Если есть предикты, сортируем по убыванию вероятности
        base.sort((a, b) => {
            const fa = (a.features || {}) as Record<string, any>;
            const fb = (b.features || {}) as Record<string, any>;
            const pa = Number(fa.purchase_proba ?? 0);
            const pb = Number(fb.purchase_proba ?? 0);
            return pb - pa;
        });

        return query ? base.filter((u) => u.user_id.toLowerCase().includes(query)) : base;
    }, [users, searchQuery]);

    const handleUserSelect = (user: User) => {
        setUserId(user.user_id);
        onSelectUserData(user);
        setSearchQuery('');
    };

    const shouldShowUserList = searchQuery.trim().length > 0 || filteredUsers.length > 0;

    return (
        <Stack spacing={2}>
            {/* Поиск */}
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
                                    <PersonSearchIcon color="action" />
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
                            {usersLoading ? <CircularProgress size={24} /> : 'Пошук'}
                        </Button>
                    )}
                </Stack>
            </Box>

            {/* Список пользователей или лоадер */}
            {usersLoading ? (
                <Box sx={{p: 3, textAlign: 'center'}}>
                    <CircularProgress />
                    <Typography variant="body2" color="text.secondary" mt={1}>
                        Пошук користувачів...
                    </Typography>
                </Box>
            ) : (
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
                                filteredUsers.map((user) => {
                                    const f = (user.features || {}) as Record<string, any>;
                                    const segment = getSegment(f);

                                    const daysSinceLast = Number(f.days_since_last ?? NaN);
                                    const events7 = Number(f.events_last_7d ?? 0);
                                    const totalPurchases = Number(f.total_purchases ?? 0);
                                    const spentTotal = Number(f.total_spent ?? 0);

                                    const leftBorderColor = segment.badgeColor;
                                    const shortId = user.user_id.split('-')[0];

                                    return (
                                        <Paper
                                            key={user.user_id}
                                            variant="outlined"
                                            sx={{
                                                p: 2,
                                                cursor: 'pointer',
                                                borderRadius: 2,
                                                mb: 0.5,
                                                display: 'flex',
                                                flexDirection: 'column',
                                                gap: 1,
                                                borderLeft: `6px solid ${leftBorderColor}`,
                                                backgroundColor: segment.bg,
                                                transition: 'all 0.2s',
                                                '&:hover': {
                                                    boxShadow: 4,
                                                    transform: 'translateY(-2px)',
                                                },
                                            }}
                                            onClick={() => handleUserSelect(user)}
                                        >
                                            {/* HEADER: статус + вероятность + ID */}
                                            <Stack
                                                direction="row"
                                                justifyContent="space-between"
                                                alignItems="center"
                                            >
                                                <Stack direction="row" spacing={1} alignItems="center">
                                                    <Chip
                                                        size="small"
                                                        color={segment.color}
                                                        label={segment.label}
                                                        sx={{
                                                            fontWeight: 700,
                                                            borderRadius: 1.5,
                                                            fontSize: '0.75rem',
                                                        }}
                                                    />
                                                    <Typography
                                                        variant="caption"
                                                        sx={{fontFamily: 'monospace', opacity: 0.6}}
                                                    >
                                                        {shortId}
                                                    </Typography>
                                                </Stack>
                                            </Stack>

                                            {/* Краткое объяснение для маркетолога */}
                                            <Typography
                                                variant="body2"
                                                sx={{
                                                    fontSize: '0.8rem',
                                                    fontStyle: 'italic',
                                                    color: 'text.secondary',
                                                }}
                                            >
                                                {segment.reason}
                                            </Typography>

                                            <Divider sx={{borderStyle: 'dashed', opacity: 0.5}} />

                                            {/* Три ключевые цифры: давность, активность, деньги */}
                                            <Box
                                                sx={{
                                                    display: 'grid',
                                                    gridTemplateColumns: '1fr 1fr 1fr',
                                                    gap: 1.5,
                                                    alignItems: 'center',
                                                }}
                                            >
                                                {/* Давность */}
                                                <Box sx={{textAlign: 'center'}}>
                                                    <Typography variant="caption" color="text.secondary" display="block">
                                                        Остання активність
                                                    </Typography>
                                                    <Typography
                                                        variant="h6"
                                                        sx={{lineHeight: 1.2}}
                                                        color={
                                                            !isNaN(daysSinceLast) && daysSinceLast <= 7
                                                                ? 'success.main'
                                                                : !isNaN(daysSinceLast) && daysSinceLast > 30
                                                                ? 'error.main'
                                                                : 'text.primary'
                                                        }
                                                    >
                                                        {isNaN(daysSinceLast) ? '—' : `${daysSinceLast} дн. тому`}
                                                    </Typography>
                                                </Box>

                                                {/* Активность 7д / покупки */}
                                                <Box
                                                    sx={{
                                                        textAlign: 'center',
                                                        borderLeft: '1px solid',
                                                        borderRight: '1px solid',
                                                        borderColor: 'divider',
                                                    }}
                                                >
                                                    <Typography variant="caption" color="text.secondary" display="block">
                                                        Активність 7д
                                                    </Typography>
                                                    <Typography variant="h6" sx={{lineHeight: 1.2}}>
                                                        {events7}
                                                    </Typography>
                                                    <Typography
                                                        variant="caption"
                                                        color="text.secondary"
                                                        fontSize="0.7rem"
                                                    >
                                                        Покупок всього: {totalPurchases}
                                                    </Typography>
                                                </Box>

                                                {/* Деньги */}
                                                <Box sx={{textAlign: 'center'}}>
                                                    <Typography variant="caption" color="text.secondary" display="block">
                                                        Витратив
                                                    </Typography>
                                                    <Typography variant="h6" sx={{lineHeight: 1.2}}>
                                                        {(spentTotal / 1000).toFixed(1)}k₴
                                                    </Typography>
                                                    <Typography
                                                        variant="caption"
                                                        color="text.secondary"
                                                        fontSize="0.7rem"
                                                    >
                                                        Lifetime value
                                                    </Typography>
                                                </Box>
                                            </Box>
                                        </Paper>
                                    );
                                })
                            ) : (
                                <Alert severity="info" sx={{borderRadius: 2}}>
                                    Користувачів не знайдено. Спробуйте інший запит.
                                </Alert>
                            )}
                        </Stack>
                    </Paper>
                )
            )}

            {/* Инфо по поиску */}
            {searchQuery.trim() && !usersLoading && filteredUsers.length === 0 && (
                <Alert severity="info" sx={{borderRadius: 2}}>
                    Користувачів не знайдено. Спробуйте інший запит.
                </Alert>
            )}

            {userId && (
                <Fade in={!!userId}>
                    <Alert severity="success" sx={{borderRadius: 2}} icon={<CheckCircleIcon />}>
                        Обрано користувача: {userId}
                    </Alert>
                </Fade>
            )}

            <Typography variant="caption" color="text.secondary">
                {users.length === 0
                    ? 'Немає користувачів'
                    : `Знайдено користувачів: ${users.length}`}
                {searchQuery.trim() &&
                    filteredUsers.length > 0 &&
                    ` (відфільтровано: ${filteredUsers.length})`}
            </Typography>
        </Stack>
    );
};