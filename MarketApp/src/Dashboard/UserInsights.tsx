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
import AttachMoneyIcon from '@mui/icons-material/AttachMoney'; // Replaced LocalAtmIcon with a more modern one
import TimelineIcon from '@mui/icons-material/Timeline';
import AccessTimeIcon from '@mui/icons-material/AccessTime'; // For time indicators
import PercentIcon from '@mui/icons-material/Percent';
import ShoppingCartIcon from '@mui/icons-material/ShoppingCart';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
// For conversion
import CalculateIcon from '@mui/icons-material/Calculate'; // For average frequency
import type {User} from '../types';

interface UserInsightsProps {
    user?: User;
}

// Improved formatting function for UAH
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
 * User status for the card (logic preserved)
 */
function getSegment(features: Record<string, any>): SegmentInfo {
    const p = Number(features.purchase_proba ?? 0);
    const w = Number(features.will_purchase_pred ?? 0);
    const events7 = Number(features.events_last_7d ?? 0);
    const daysSinceLast = Number(features.days_since_last ?? 999);

    if (p >= 0.7 && daysSinceLast <= 30) {
        return {
            id: 'hot',
            label: '🔥 Hot Audience',
            color: 'success',
            description: `Model gives ${(p * 100).toFixed(0)}% chance of purchase. Recently active.`,
        };
    }

    if (p >= 0.3 && (events7 >= 3 || daysSinceLast <= 14)) {
        return {
            id: 'warm',
            label: '⚡ Promising',
            color: 'info',
            description: `Significant chance of purchase (${(p * 100).toFixed(0)}%), recent activity present.`,
        };
    }

    if (p >= 0.1) {
        return {
            id: 'cold',
            label: '🟠 Weak Interest',
            color: 'warning',
            description: `Purchase chance is moderate (${(p * 100).toFixed(0)}%). Can be included in mass campaigns.`,
        };
    }

    if (p < 0.1 && w === 0) {
        return {
            id: 'ignore',
            label: '⛔ Non-target now',
            color: 'error',
            description: `Model sees a very low purchase chance (${(p * 100).toFixed(1)}%).`,
        };
    }

    return {
        id: 'cold',
        label: '🟠 Weak Interest',
        color: 'warning',
        description: `Purchase chance ${(p * 100).toFixed(0)}%.`,
    };
}

// Wrapper component for each metric
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
            p: 1.5, // Increased padding for better readability
            borderRadius: 2,
            border: '1px solid',
            borderColor: 'divider',
            height: '100%', // For Grid/Stack
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
            {/* Main value */}
            <Typography variant="h6" fontWeight={800} color="text.primary">
                {value}
            </Typography>
            {/* Additional value */}
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

    // For activity mini-chart: 7d share in 30d
    const activityShare = events30 > 0 ? Math.min(100, (events7 / events30) * 100) : 0;
// Border color for VIP/Alert
    const borderColor = willPurchasePred === 1 ? theme.palette.success.main :
        segment.color === 'error' ? theme.palette.error.main :
            theme.palette.divider;

    // Micro-trends: indicator calculations for the last 3 days relative to 7 days
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
                border: '2px solid', // Thicker border for emphasis
                borderColor: borderColor,
            }}
        >
            <Stack spacing={3}>
                {/* HEADER: name + status + probability */}
                <Stack direction="row" alignItems="center" justifyContent="space-between">
                    <Box>
                        <Typography variant="subtitle1" fontWeight={700}>
                            User Profile
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
                            label={`Chance: ${(purchaseProba * 100).toFixed(0)}%`}
                            sx={{
                                fontWeight: 600,
                                borderRadius: 1.5,
                                borderColor: theme.palette[segment.color].main,
                                color: theme.palette[segment.color].main,
                            }}
                        />
                    </Stack>
                </Stack>

                {/* Model description */}
                <Box
                    sx={{
                        p: 1.5,
                        bgcolor: `${theme.palette[segment.color].main}15`,
                        borderRadius: 2,
                        borderLeft: `5px solid ${theme.palette[segment.color].main}`,
                    }}
                >
                    <Typography variant="body2" fontWeight={600} color="text.primary">
                        Model Insight:
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{fontStyle: 'italic'}}>
                        {segment.description}
                    </Typography>
                </Box>

                <Divider/>

                <Typography variant="subtitle2" fontWeight={700}>
                    🔮 Predictions and Legacy (LTV)
                </Typography>
                <Stack direction={{xs: 'column', sm: 'row'}} spacing={2}>
                    <MetricBox
                        icon={<AttachMoneyIcon color="primary" fontSize="small"/>}
                        title="Total Spent"
                        value={formatCurrency(totalSpent, 0)}
                        tooltip="Total amount spent by the customer over all time. Important LTV indicator."
                        secondaryValue={`Purchases: ${formatNumber(totalPurchases, 0)}`}
                    />
                    <MetricBox
                        icon={<AttachMoneyIcon color="success" fontSize="small"/>}
                        title="Predicted Next Amount"
                        value={formatCurrency(nextAmountPred, 0)}
                        tooltip="How much the model expects the customer to spend on their next purchase."
                        secondaryValue={`Expected in: ${isNaN(daysToNextPred) ? '—' : `${daysToNextPred.toFixed(0)} days`}`}
                    />
                    <MetricBox
                        icon={<AccessTimeIcon color="warning" fontSize="small"/>}
                        title="Average Revenue Per Action (APV)"
                        value={`${formatNumber(avgSpendPerEvent, 2)} ₴`}
                        tooltip="Average revenue from each user action (view, click, purchase). Higher value indicates a more valuable customer."
                        secondaryValue={`Conversion (30d): ${formatNumber(conversion30 * 100, 1)}%`}
                    />
                </Stack>

                <Divider/>

                {/* BLOCK 2: Dynamics and Activity */}
                <Typography variant="subtitle2" fontWeight={700}>
                    📈 Current Dynamics and Behavior
                </Typography>
                <Stack direction={{xs: 'column', sm: 'row'}} spacing={2}>
                    <Box sx={{flex: 1}}>
                        <MetricBox
                            icon={<TimelineIcon color="info" fontSize="small"/>}
                            title="Time Activity"
                            value={
                                <>
                                    7 days: <b>{events7}</b>
                                    <span style={{marginLeft: '8px', opacity: 0.6}}>|</span>
                                    <span style={{marginLeft: '8px'}}>30 days: <b>{events30}</b></span>
                                </>
                            }
                            tooltip="Number of actions performed by the user in the last 7 and 30 days."
                            secondaryValue={
                                <>
                                    Last visit: <b>{isNaN(daysSinceLast) ? '—' : `${daysSinceLast} days`}</b>
                                </>
                            }
                        />
                    </Box>

                    <Box sx={{flex: 1}}>
                        <MetricBox
                            icon={<CalculateIcon color="secondary" fontSize="small"/>}
                            title="Purchase Frequency"
                            value={formatNumber(purchaseFrequency, 2)}
                            tooltip="Average number of purchases per active user day. >1.0 means multi-orders."
                            secondaryValue={`Frequency: ${formatNumber(purchaseFrequency, 2)} purchases/day`}
                        />
                    </Box>
                </Stack>

                {/* BLOCK 3: Activity Mini-Chart */}
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
                            Activity Concentration (7d vs 30d)
                        </Typography>
                        <Tooltip
                            title="How much activity (events) occurs in the last week relative to the entire month. A high value (near 100%) may indicate 'burst' activity followed by a pause."
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
                                bgcolor: theme.palette.warning.light, // Background - 30 days
                                [`& .MuiLinearProgress-bar`]: {
                                    borderRadius: 5,
                                    bgcolor: theme.palette.info.main, // Color - 7 days
                                },
                            }}
                        />
                        <Typography variant="caption" color="text.secondary" sx={{mt: 0.5}}>
                            {events7} of {events30} events in 30d = **{activityShare.toFixed(0)}%**
                            (Normal range: 25% - 40%)
                        </Typography>
                    </Box>
                </Box>

                {/* Micro-trends */}
                <Divider />
                <Typography variant="subtitle2" fontWeight={700}>
                    📊 Micro-trends (last 3 days)
                </Typography>
                <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
                    <MetricBox
                        icon={<TrendingUpIcon color="primary" fontSize="small" />}
                        title="Event Growth"
                        value={isNaN(microEventGrowth) ? '—' : `${(microEventGrowth * 100).toFixed(0)}%`}
                        tooltip="Ratio of events in the last 3 days to the average over the last 7 days. >100% means activity growth."
                    />
                    <MetricBox
                        icon={<ShoppingCartIcon color="secondary" fontSize="small" />}
                        title="Purchase Growth"
                        value={isNaN(microPurchaseGrowth) ? '—' : `${(microPurchaseGrowth * 100).toFixed(0)}%`}
                        tooltip="Ratio of purchases in the last 3 days to the average over the last 7 days. >100% means purchase growth."
                    />
                    <MetricBox
                        icon={<PercentIcon color="info" fontSize="small" />}
                        title="3d Conversion"
                        value={isNaN(microPurchaseRatio) ? '—' : `${(microPurchaseRatio * 100).toFixed(1)}%`}
                        tooltip="Share of purchases among all actions in the last 3 days."
                    />
                    <MetricBox
                        icon={<AttachMoneyIcon color="success" fontSize="small" />}
                        title="Spent Growth"
                        value={isNaN(microSpentGrowth) ? '—' : `${(microSpentGrowth * 100).toFixed(0)}%`}
                        tooltip="Ratio of spending in the last 3 days to the average over the last 7 days. >100% means spending growth."
                    />
                </Stack>

                <Typography variant="caption" color="text.secondary" sx={{mt: 1, display: 'block'}}>
                    **Status Explanations:**
                    **{theme.palette.success.main} (Green):** Model expects a purchase soon.
                    **{theme.palette.info.main} (Blue):** High probability, but not confident enough for "hot" status.
                    **{theme.palette.warning.main} (Orange):** Potential audience for warming up/retargeting.
                    **{theme.palette.error.main} (Red):** Low purchase chance; focus on reactivation rather than conversion.
                </Typography>
            </Stack>
        </Paper>
    );
};