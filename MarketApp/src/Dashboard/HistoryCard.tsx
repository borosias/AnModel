import React from 'react';
import {
    Card,
    CardContent,
    Stack,
    Avatar,
    Box,
    Typography,
    Chip,
    Button,
    TableContainer,
    Paper,
    Table,
    TableHead,
    TableRow,
    TableCell,
    TableBody,
    alpha,
    useTheme,
} from '@mui/material';
import HistoryIcon from '@mui/icons-material/History';
import DeleteIcon from '@mui/icons-material/Delete';
import type {HistoryItem} from '../types';
import {formatFeatureValue} from '../constants';

interface HistoryCardProps {
    history: HistoryItem[];
    onClearHistory: () => void;
}

export const HistoryCard: React.FC<HistoryCardProps> = ({
                                                            history,
                                                            onClearHistory,
                                                        }) => {
    const theme = useTheme();
    return (
        <Card
            elevation={0}
            sx={{
                borderRadius: 3,
                border: `1px solid ${alpha(theme.palette.warning.main, 0.1)}`,
                background: "white",
            }}
        >
            <CardContent>
                <Stack spacing={2}>
                    <Stack direction="row" justifyContent="space-between" alignItems="center">
                        <Stack direction="row" alignItems="center" spacing={1.5}>
                            <Avatar
                                sx={{
                                    bgcolor: alpha(theme.palette.warning.main, 0.1),
                                    color: theme.palette.warning.main,
                                }}
                            >
                                <HistoryIcon/>
                            </Avatar>
                            <Box>
                                <Typography variant="h6" sx={{fontWeight: 700}}>
                                    Історія аналізу
                                </Typography>
                                <Typography variant="caption" color="text.secondary">
                                    Останні {Math.min(history.length, 20)} прогнозів
                                </Typography>
                            </Box>
                        </Stack>

                        <Stack direction="row" spacing={1} alignItems="center">
                            <Chip
                                label={`${history.length} записів`}
                                size="medium"
                                color="warning"
                                variant="outlined"
                                sx={{fontWeight: 600}}
                            />

                            {history.length > 0 && (
                                <Button
                                    variant="outlined"
                                    color="error"
                                    onClick={onClearHistory}
                                    startIcon={<DeleteIcon/>}
                                    size="small"
                                    sx={{
                                        borderRadius: 2,
                                        fontWeight: 600,
                                    }}
                                >
                                    Очистити
                                </Button>
                            )}
                        </Stack>
                    </Stack>

                    <TableContainer
                        component={Paper}
                        variant="outlined"
                        sx={{
                            borderRadius: 2,
                            '&::-webkit-scrollbar': {
                                width: 8,
                            },
                            '&::-webkit-scrollbar-track': {
                                background: alpha(theme.palette.grey[300], 0.5),
                                borderRadius: 4,
                            },
                            '&::-webkit-scrollbar-thumb': {
                                background: alpha(theme.palette.primary.main, 0.3),
                                borderRadius: 4,
                            },
                        }}
                    >
                        <Table stickyHeader >
                            <TableHead>
                                <TableRow sx={{
                                    '& th': {
                                        fontWeight: 700,
                                        bgcolor: alpha(theme.palette.primary.main, 1),
                                        color: theme.palette.primary.contrastText,
                                        borderBottom: `2px solid ${alpha(theme.palette.primary.main, 0.2)}`,
                                    }
                                }}>
                                    <TableCell>Дата</TableCell>
                                    <TableCell>Час</TableCell>
                                    <TableCell>Модель</TableCell>
                                    <TableCell>ID користувача</TableCell>
                                    <TableCell>Вірогідність покупки</TableCell>
                                    <TableCell>Чи буде покупка?</TableCell>
                                    <TableCell>Дні до наступної покупки</TableCell>
                                    <TableCell>Приблизна сума</TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {history.map((item, index) => (
                                    <TableRow
                                        key={item.id}
                                        sx={{
                                            '&:nth-of-type(even)': {bgcolor: alpha(theme.palette.action.hover, 0.1)},
                                            '&:hover': {
                                                bgcolor: alpha(theme.palette.primary.main, 0.05),
                                                cursor: 'pointer',
                                            },
                                            borderBottom: index < history.length - 1 ? `1px solid ${alpha(theme.palette.divider, 0.1)}` : 'none',
                                        }}
                                        onClick={() => {
                                            console.log('History item clicked:', item);
                                        }}
                                    >
                                        <TableCell>
                                            <Typography variant="body2" fontWeight={500}>
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
                                                color="primary"
                                                variant="outlined"
                                                sx={{fontWeight: 600}}
                                            />
                                        </TableCell>
                                        <TableCell>
                                            <Typography variant="body2" fontWeight={600} maxWidth={150} sx={{overflow: 'hidden', textOverflow: 'ellipsis', textWrap: 'nowrap'}} >
                                                {item.user_id ? `${item.user_id}` : "Ручний ввід"}
                                            </Typography>
                                        </TableCell>
                                        {item.output && Object.entries(item.output[0]).map(([key, value]) => (
                                            <TableCell key={key} align="center">
                                                <Typography variant="body2" fontWeight={500} p={2} sx={{fontWeight: (key === "next_purchase_amount_pred") ? 1000 : 400}}>
                                                   {formatFeatureValue(key,value as number)}
                                                </Typography>
                                            </TableCell>
                                        ))}

                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    </TableContainer>
                </Stack>
            </CardContent>
        </Card>
    );
};