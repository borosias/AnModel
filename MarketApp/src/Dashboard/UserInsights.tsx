import React from 'react';
import {
    Box,
    Chip,
    Divider,
    LinearProgress,
    Paper,
    Stack,
    Tooltip,
    Typography,
    useTheme,
} from '@mui/material';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import AttachMoneyIcon from '@mui/icons-material/AttachMoney'; // Заменил LocalAtmIcon на более современный
import TimelineIcon from '@mui/icons-material/Timeline';
import AccessTimeIcon from '@mui/icons-material/AccessTime'; // Для временных показателей
import PercentIcon from '@mui/icons-material/Percent';
import ShoppingCartIcon from '@mui/icons-material/ShoppingCart';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
// Для конверсии
import CalculateIcon from '@mui/icons-material/Calculate'; // Для средней частоты
import type {User} from '../types';

interface UserInsightsProps {
    user?: User;
}

// Улучшенная функция форматирования для гривен
function formatCurrency(val: number | undefined, digits = 0): string {
    if (val === null || val === undefined || Number.isNaN(val)) return '—';
    return val.toLocaleString('uk-UA', {
        style: 'currency',
        currency: 'UAH',
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
    });
}

function formatNumber(val: number | undefined, digits = 0): string {
    if (val === null || val === undefined || Number.isNaN(val)) return '—';
    return val.toLocaleString(undefined, {
        maximumFractionDigits: digits,
        minimumFractionDigits: digits,
    });
}

type SegmentId = 'hot' | 'warm' | 'cold' | 'ignore';

interface SegmentInfo {
    id: SegmentId;
    label: string;
    color: 'success' | 'info' | 'warning' | 'error' | 'secondary';
    description: string;
}

/**
 * Статус пользователя для карточки (логика сохранена)
 */
function getSegment(features: Record<string, any>): SegmentInfo {
    const p = Number(features.purchase_proba ?? 0);
    const w = Number(features.will_purchase_pred ?? 0);
    const events7 = Number(features.events_last_7d ?? 0);
    const daysSinceLast = Number(features.days_since_last ?? 999);

    if (p >= 0.7 && daysSinceLast <= 30) {
        return {
            id: 'hot',
            label: '🔥 Гаряча аудиторія',
            color: 'success',
            description: `Модель дає ${(p * 100).toFixed(0)}% шанс покупки. Активний останнім часом.`,
        };
    }

    if (p >= 0.3 && (events7 >= 3 || daysSinceLast <= 14)) {
        return {
            id: 'warm',
            label: '⚡ Перспективний',
            color: 'info',
            description: `Є відчутний шанс покупки (${(p * 100).toFixed(0)}%), є недавня активність.`,
        };
    }

    if (p >= 0.1) {
        return {
            id: 'cold',
            label: '🟠 Слабкий інтерес',
            color: 'warning',
            description: `Шанс покупки помірний (${(p * 100).toFixed(0)}%). Можна включати в масові кампанії.`,
        };
    }

    if (p < 0.1 && w === 0) {
        return {
            id: 'ignore',
            label: '⛔ Нецільовий зараз',
            color: 'error',
            description: `Модель бачить дуже низький шанс покупки (${(p * 100).toFixed(1)}%).`,
        };
    }

    return {
        id: 'cold',
        label: '🟠 Слабкий інтерес',
        color: 'warning',
        description: `Шанс покупки ${(p * 100).toFixed(0)}%.`,
    };
}

// Компонент-обертка для каждой метрики
interface MetricBoxProps {
    icon: React.ReactNode;
    title: string;
    value: React.ReactNode;
    tooltip: string;
    secondaryValue?: React.ReactNode;
}

const MetricBox: React.FC<MetricBoxProps> = ({
                                                 icon,
                                                 title,
                                                 value,
                                                 tooltip,
                                                 secondaryValue,
                                             }) => (
    <Box
        sx={{
            p: 1.5, // Увеличенный padding для лучшей читаемости
            borderRadius: 2,
            border: '1px solid',
            borderColor: 'divider',
            height: '100%', // Для Grid/Stack
        }}
    >
        <Stack spacing={0.5}>
            <Stack direction="row" alignItems="center" spacing={1}>
                {icon}
                <Typography variant="body2" fontWeight={700} noWrap>
                    {title}
                </Typography>
                <Tooltip title={tooltip} arrow>
                    <InfoOutlinedIcon fontSize="small" color="disabled" sx={{ml: 'auto'}}/>
                </Tooltip>
            </Stack>
            {/* Основное значение */}
            <Typography variant="h6" fontWeight={800} color="text.primary">
                {value}
            </Typography>
            {/* Дополнительное значение */}
            {secondaryValue && (
                <Typography variant="caption" color="text.secondary">
                    {secondaryValue}
                </Typography>
            )}
        </Stack>
    </Box>
);

export const UserInsights: React.FC<UserInsightsProps> = ({user}) => {
    const theme = useTheme();

    if (!user || !user.features) {
        return null;
    }

    const f = user.features as Record<string, any>;

    const purchaseProba = Number(f.purchase_proba ?? 0);
    const willPurchasePred = Number(f.will_purchase_pred ?? 0);
    const daysToNextPred = Number(f.days_to_next_pred ?? NaN);
    const nextAmountPred = Number(f.next_purchase_amount_pred ?? NaN);

    const events7 = Number(f.events_last_7d ?? 0);
    const events30 = Number(f.events_last_30d ?? 0);
    const daysSinceLast = Number(f.days_since_last ?? NaN);
    const totalPurchases = Number(f.total_purchases ?? 0);
    const totalSpent = Number(f.total_spent ?? 0);
    const avgSpendPerEvent = Number(f.avg_spend_per_event ?? 0);
    const conversion30 = Number(f.conversion_rate_30d ?? 0);
    const purchaseFrequency = Number(f.purchase_frequency ?? 0);

    const segment = getSegment(f);

    // Для мини-графика активности: доля 7д в 30д
    const activityShare = events30 > 0 ? Math.min(100, (events7 / events30) * 100) : 0;
// Цвет рамки для VIP/Тревоги
    const borderColor = willPurchasePred === 1 ? theme.palette.success.main :
        segment.color === 'error' ? theme.palette.error.main :
            theme.palette.divider;

    // Мікро-тренди: розрахунки показників за останні 3 дні відносно 7 днів
    const microEventGrowth = Number(f.micro_event_growth ?? NaN);
    const microPurchaseGrowth = Number(f.micro_purchase_growth ?? NaN);
    const microPurchaseRatio = Number(f.micro_purchase_ratio ?? NaN);
    const microSpentGrowth = Number(f.micro_spent_growth ?? NaN);

    return (
        <Paper
            variant="outlined"
            sx={{
                p: 2.5,
                borderRadius: 3,
                mt: 2,
                border: '2px solid', // Утолщенная рамка для акцента
                borderColor: borderColor,
            }}
        >
            <Stack spacing={3}>
                {/* HEADER: имя + статус + вероятность */}
                <Stack direction="row" alignItems="center" justifyContent="space-between">
                    <Box>
                        <Typography variant="subtitle1" fontWeight={700}>
                            Профіль користувача
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                            ID: {user.user_id}
                        </Typography>
                    </Box>

                    <Stack direction="column" spacing={0.5} alignItems="flex-end">
                        <Chip
                            size="medium"
                            color={segment.color}
                            label={segment.label}
                            sx={{fontWeight: 700, borderRadius: 1.5, fontSize: '0.85rem'}}
                        />
                        <Chip
                            size="small"
                            variant="outlined"
                            label={`Шанс: ${(purchaseProba * 100).toFixed(0)}%`}
                            sx={{
                                fontWeight: 600,
                                borderRadius: 1.5,
                                borderColor: theme.palette[segment.color].main,
                                color: theme.palette[segment.color].main,
                            }}
                        />
                    </Stack>
                </Stack>

                {/* Описание от модели */}
                <Box
                    sx={{
                        p: 1.5,
                        bgcolor: `${theme.palette[segment.color].main}15`,
                        borderRadius: 2,
                        borderLeft: `5px solid ${theme.palette[segment.color].main}`,
                    }}
                >
                    <Typography variant="body2" fontWeight={600} color="text.primary">
                        Інсайт моделі:
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{fontStyle: 'italic'}}>
                        {segment.description}
                    </Typography>
                </Box>

                <Divider/>

                {/* БЛОК 1: Прогнозы Модели и Наследие */}
                <Typography variant="subtitle2" fontWeight={700}>
                    🔮 Прогнози та Наследие (LTV)
                </Typography>
                <Stack direction={{xs: 'column', sm: 'row'}} spacing={2}>
                    <MetricBox
                        icon={<AttachMoneyIcon color="primary" fontSize="small"/>}
                        title="Витрачено всього"
                        value={formatCurrency(totalSpent, 0)}
                        tooltip="Загальна сума, витрачена клієнтом за весь час. Важливий показник LTV."
                        secondaryValue={`Покупок: ${formatNumber(totalPurchases, 0)}`}
                    />
                    <MetricBox
                        icon={<AttachMoneyIcon color="success" fontSize="small"/>}
                        title="Прогноз наступної суми"
                        value={formatCurrency(nextAmountPred, 0)}
                        tooltip="Скільки модель очікує, що клієнт витратить під час наступної покупки."
                        secondaryValue={`Очікується через: ${isNaN(daysToNextPred) ? '—' : `${daysToNextPred.toFixed(0)} дн.`}`}
                    />
                    <MetricBox
                        icon={<AccessTimeIcon color="warning" fontSize="small"/>}
                        title="Середній дохід на дію (APV)"
                        value={`${formatNumber(avgSpendPerEvent, 2)} ₴`}
                        tooltip="Скільки в середньому приносить кожна дія користувача (перегляд, клік, покупка). Чим вища цифра — тим цінніший клієнт."
                        secondaryValue={`Конверсія (30 дн.): ${formatNumber(conversion30 * 100, 1)}%`}
                    />
                </Stack>

                <Divider/>

                {/* БЛОК 2: Динамика и Активность */}
                <Typography variant="subtitle2" fontWeight={700}>
                    📈 Поточна Динаміка та Поведінка
                </Typography>
                <Stack direction={{xs: 'column', sm: 'row'}} spacing={2}>
                    <Box sx={{flex: 1}}>
                        <MetricBox
                            icon={<TimelineIcon color="info" fontSize="small"/>}
                            title="Часова активність"
                            value={
                                <>
                                    7 дн: <b>{events7}</b>
                                    <span style={{marginLeft: '8px', opacity: 0.6}}>|</span>
                                    <span style={{marginLeft: '8px'}}>30 дн: <b>{events30}</b></span>
                                </>
                            }
                            tooltip="Скільки дій зробив користувач за останні 7 та 30 днів."
                            secondaryValue={
                                <>
                                    Останній візит: <b>{isNaN(daysSinceLast) ? '—' : `${daysSinceLast} дн.`}</b>
                                </>
                            }
                        />
                    </Box>

                    <Box sx={{flex: 1}}>
                        <MetricBox
                            icon={<CalculateIcon color="secondary" fontSize="small"/>}
                            title="Частота покупок"
                            value={formatNumber(purchaseFrequency, 2)}
                            tooltip="Скільки покупок у середньому припадає на активний день користувача. >1.0 означає мульти-замовлення."
                            secondaryValue={`Частота: ${formatNumber(purchaseFrequency, 2)} покупок/день`}
                        />
                    </Box>
                </Stack>

                {/* БЛОК 3: Мини-График Активности */}
                <Box
                    sx={{
                        p: 1.5,
                        borderRadius: 2,
                        border: '1px solid',
                        borderColor: 'divider',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 1.5,
                    }}
                >
                    <Stack direction="row" alignItems="center" spacing={1}>
                        <PercentIcon color="primary" fontSize="small"/>
                        <Typography variant="body2" fontWeight={700}>
                            Концентрація Активності (7д vs 30д)
                        </Typography>
                        <Tooltip
                            title="Скільки активності (подій) припадає на останній тиждень відносно всього місяця. Високе значення (близько 100%) може означати 'вибухову' активність з подальшою паузою."
                            arrow
                        >
                            <InfoOutlinedIcon fontSize="small" color="disabled" sx={{ml: 'auto'}}/>
                        </Tooltip>
                    </Stack>

                    <Box>
                        <LinearProgress
                            variant="determinate"
                            value={activityShare}
                            sx={{
                                mt: 0.5,
                                height: 12,
                                borderRadius: 5,
                                bgcolor: theme.palette.warning.light, // Фон - 30 дней
                                [`& .MuiLinearProgress-bar`]: {
                                    borderRadius: 5,
                                    bgcolor: theme.palette.info.main, // Цвет - 7 дней
                                },
                            }}
                        />
                        <Typography variant="caption" color="text.secondary" sx={{mt: 0.5}}>
                            {events7} з {events30} подій за 30д = **{activityShare.toFixed(0)}%**
                            (Нормальний діапазон: 25% - 40%)
                        </Typography>
                    </Box>
                </Box>

                {/* Мікро-тренди */}
                <Divider />
                <Typography variant="subtitle2" fontWeight={700}>
                    📊 Мікро-тренди (останні 3 дні)
                </Typography>
                <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
                    <MetricBox
                        icon={<TrendingUpIcon color="primary" fontSize="small" />}
                        title="Приріст подій"
                        value={isNaN(microEventGrowth) ? '—' : `${(microEventGrowth * 100).toFixed(0)}%`}
                        tooltip="Відношення кількості подій за останні 3 дні до середнього за останні 7 днів. >100% означає зростання активності."
                    />
                    <MetricBox
                        icon={<ShoppingCartIcon color="secondary" fontSize="small" />}
                        title="Приріст покупок"
                        value={isNaN(microPurchaseGrowth) ? '—' : `${(microPurchaseGrowth * 100).toFixed(0)}%`}
                        tooltip="Відношення кількості покупок за останні 3 дні до середнього за останні 7 днів. >100% означає зростання покупок."
                    />
                    <MetricBox
                        icon={<PercentIcon color="info" fontSize="small" />}
                        title="Конверсія 3д"
                        value={isNaN(microPurchaseRatio) ? '—' : `${(microPurchaseRatio * 100).toFixed(1)}%`}
                        tooltip="Частка покупок серед всіх дій за останні 3 дні."
                    />
                    <MetricBox
                        icon={<AttachMoneyIcon color="success" fontSize="small" />}
                        title="Приріст витрат"
                        value={isNaN(microSpentGrowth) ? '—' : `${(microSpentGrowth * 100).toFixed(0)}%`}
                        tooltip="Відношення суми витрат за останні 3 дні до середнього за останні 7 днів. >100% означає зростання витрат."
                    />
                </Stack>

                <Typography variant="caption" color="text.secondary" sx={{mt: 1, display: 'block'}}>
                    **Пояснення статусів:**
                    **{theme.palette.success.main} (Зелений):** Модель очікує покупку найближчим часом.
                    **{theme.palette.info.main} (Синій):** Висока ймовірність, але недостатньо впевнена для "гарячого" статусу.
                    **{theme.palette.warning.main} (Помаранчевий):** Потенційна аудиторія для прогріву/ретаргетингу.
                    **{theme.palette.error.main} (Червоний):** Низький шанс покупки; фокус на реактивацію, а не на конверсію.
                </Typography>
            </Stack>
        </Paper>
    );
};