import React from 'react';
import {Box, Chip, Grid, Paper, Stack, Tooltip, Typography} from '@mui/material';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import LocalAtmIcon from '@mui/icons-material/LocalAtm';
import TimelineIcon from '@mui/icons-material/Timeline';
import BoltIcon from '@mui/icons-material/Bolt';
import PercentIcon from '@mui/icons-material/Percent';
import ShoppingCartCheckoutIcon from '@mui/icons-material/ShoppingCartCheckout';
import type {User} from '../types';

type MetricKey =
    | 'avg_spend_per_event'
    | 'events_last_7d'
    | 'events_per_day'
    | 'conversion_rate_30d'
    | 'purchase_frequency';

interface UserInsightsProps {
    user?: User;
}

function formatNumber(val: number | undefined, digits = 0): string {
    if (val === null || val === undefined || Number.isNaN(val)) return '—';
    return val.toLocaleString(undefined, {
        maximumFractionDigits: digits,
        minimumFractionDigits: digits,
    });
}

function rateValue(key: MetricKey, value?: number): {
    label: string;
    color: 'success' | 'warning' | 'default' | 'error'
} {
    if (value === undefined || value === null || Number.isNaN(value)) {
        return {label: 'немає даних', color: 'default'};
    }

    switch (key) {
        case 'avg_spend_per_event':
            if (value >= 300) return {label: 'високо', color: 'success'};
            if (value >= 100) return {label: 'середньо', color: 'warning'};
            return {label: 'низько', color: 'error'};
        case 'events_last_7d':
            if (value >= 100) return {label: 'активний', color: 'success'};
            if (value >= 20) return {label: 'середня активність', color: 'warning'};
            return {label: 'низька активність', color: 'error'};
        case 'events_per_day':
            if (value >= 30) return {label: 'часто', color: 'success'};
            if (value >= 5) return {label: 'інколи', color: 'warning'};
            return {label: 'рідко', color: 'error'};
        case 'conversion_rate_30d':
            if (value >= 0.10) return {label: 'висока', color: 'success'};
            if (value >= 0.03) return {label: 'середня', color: 'warning'};
            return {label: 'низька', color: 'error'};
        case 'purchase_frequency':
            if (value >= 0.30) return {label: 'часто', color: 'success'};
            if (value >= 0.10) return {label: 'середньо', color: 'warning'};
            return {label: 'рідко', color: 'error'};
        default:
            return {label: '—', color: 'default'};
    }
}

export const UserInsights: React.FC<UserInsightsProps> = ({user}) => {
    const f = user?.features || {} as Record<string, any>;

    const metrics: Array<{
        key: MetricKey;
        label: string;
        hint: string;
        value: number | undefined;
        format: (v?: number) => string;
        icon: React.ReactNode;
        suffix?: string;
    }> = [
        {
            key: 'avg_spend_per_event',
            label: 'Середній дохід на подію',
            hint: 'Скільки в середньому приносить кожна дія користувача. Чим вище — тим «гарячіший» клієнт.',
            value: typeof f.avg_spend_per_event === 'number' ? f.avg_spend_per_event : undefined,
            format: (v) => `${formatNumber(v, 2)} ₴`,
            icon: <LocalAtmIcon color="success" fontSize="small"/>,
        },
        {
            key: 'events_last_7d',
            label: 'Активність за 7 днів',
            hint: 'Скільки дій користувач зробив за останній тиждень. Більше — краще.',
            value: typeof f.events_last_7d === 'number' ? f.events_last_7d : undefined,
            format: (v) => formatNumber(v, 0),
            icon: <TimelineIcon color="primary" fontSize="small"/>,
        },
        {
            key: 'events_per_day',
            label: 'Середня інтенсивність',
            hint: 'Скільки дій у середньому на день за весь час. Показує звичку повертатися.',
            value: typeof f.events_per_day === 'number' ? f.events_per_day : undefined,
            format: (v) => formatNumber(v, 1),
            icon: <BoltIcon color="warning" fontSize="small"/>,
        },
        {
            key: 'conversion_rate_30d',
            label: 'Конверсія (30д)',
            hint: 'Частка дій, що завершилися покупкою за 30 днів. 0.126 = 12.6%.',
            value: typeof f.conversion_rate_30d === 'number' ? f.conversion_rate_30d : undefined,
            format: (v) => `${formatNumber((v ?? 0) * 100, 1)}%`,
            icon: <PercentIcon color="secondary" fontSize="small"/>,
        },
        {
            key: 'purchase_frequency',
            label: 'Частота покупок',
            hint: 'Як часто користувач оформлює покупку у ті дні, коли активний.',
            value: typeof f.purchase_frequency === 'number' ? f.purchase_frequency : undefined,
            format: (v) => formatNumber(v, 2),
            icon: <ShoppingCartCheckoutIcon color="info" fontSize="small"/>,
        },
    ];

    if (!user) {
        return null;
    }

    return (
        <Paper variant="outlined" sx={{p: 2, borderRadius: 2}}>
            <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{mb: 1.5}}>
                <Typography variant="subtitle1" fontWeight={700}>
                    Ключові фактори для предикта
                </Typography>
                <Stack direction="row" alignItems="center" spacing={1}>
                    <Typography variant="caption" color="text.secondary">
                        Користувач: {user.user_id}
                    </Typography>
                </Stack>
            </Stack>

            <Grid container spacing={1.5}>
                {metrics.map((m) => {
                    const rating = rateValue(m.key, m.value);


                    return (
                        <>
                            {/* @ts-ignore */}
                            <Grid item xs={12} sm={6} md={4} key={m.key}>
                                <Box sx={{
                                    p: 1.25,
                                    borderRadius: 1.5,
                                    border: '1px solid',
                                    borderColor: 'divider',
                                }}>
                                    <Stack direction="row" alignItems="center" justifyContent="space-between"
                                           sx={{mb: 0.5}}>
                                        <Stack direction="row" alignItems="center" spacing={1}>
                                            {m.icon}
                                            <Typography variant="body2" fontWeight={700}>{m.label}</Typography>
                                            <Tooltip title={m.hint} arrow>
                                                <InfoOutlinedIcon fontSize="small" color="disabled"/>
                                            </Tooltip>
                                        </Stack>
                                        <Chip size="small" color={rating.color} label={rating.label}
                                              variant="outlined"/>
                                    </Stack>

                                    <Typography variant="h6" fontWeight={800}>
                                        {m.format(m.value)}
                                    </Typography>
                                </Box>
                            </Grid>
                        </>
                    );
                })}
            </Grid>

            {/* Пояснення для маркетолога */}
            <Typography variant="caption" color="text.secondary" sx={{mt: 1.5, display: 'block'}}>
                Підказка: зелені показники підвищують ймовірність покупки. Якщо частина метрик порожня — дані поки що не
                зібрані.
            </Typography>
        </Paper>
    );
};
