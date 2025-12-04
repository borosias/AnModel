import React from 'react';
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
} from '@mui/material';
import PersonSearchIcon from '@mui/icons-material/PersonSearch';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import type {User} from '../types';

interface UserSearchProps {
  searchQuery: string;
  setSearchQuery: (query: string) => void;
  onSearchUsers: () => void;
  users: User[];
  isSearching: boolean;
  onLoadUserData: (user: User) => void;
  userId: string;
  setUserId: (id: string) => void;
}

export const UserSearch: React.FC<UserSearchProps> = ({
  searchQuery,
  setSearchQuery,
  onSearchUsers,
  users,
  isSearching,
  onLoadUserData,
  userId,
  setUserId,
}) => {
  const handleUserSelect = (user: User) => {
    setUserId(user.user_id);
    onLoadUserData(user);
  };

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
            onKeyPress={(e) => e.key === 'Enter' && onSearchUsers()}
            size="medium"
            sx={{
              '& .MuiOutlinedInput-root': { borderRadius: 2 }
            }}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <PersonSearchIcon color="action" />
                </InputAdornment>
              ),
            }}
          />
          <Button
            variant="contained"
            onClick={onSearchUsers}
            disabled={isSearching || !searchQuery.trim()}
            sx={{
              borderRadius: 2,
              minWidth: 100,
              px: 3,
            }}
          >
            {isSearching ? <CircularProgress size={24} /> : "Пошук"}
          </Button>
        </Stack>
      </Box>

      {isSearching ? (
        <Box sx={{ p: 3, textAlign: 'center' }}>
          <CircularProgress />
          <Typography variant="body2" color="text.secondary" mt={1}>
            Пошук користувачів...
          </Typography>
        </Box>
      ) : users.length > 0 ? (
        <Paper
          variant="outlined"
          sx={{
            maxHeight: 300,
            overflow: 'auto',
            borderRadius: 2,
          }}
        >
          <Stack spacing={1} sx={{ p: 1 }}>
            {users.map((user) => (
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
                    <Chip
                      label="Обрати"
                      size="small"
                      color="primary"
                      variant="outlined"
                    />
                  </Stack>
                  {user.name && (
                    <Typography variant="body2">
                      {user.name}
                    </Typography>
                  )}
                  <Stack direction="row" spacing={2}>
                    <Typography variant="caption" color="text.secondary">
                      Подій: {user.total_events || 0}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Покупок: {user.total_purchases || 0}
                    </Typography>
                  </Stack>
                </Stack>
              </Paper>
            ))}
          </Stack>
        </Paper>
      ) : searchQuery.trim() ? (
        <Alert severity="info" sx={{ borderRadius: 2 }}>
          Користувачів не знайдено. Спробуйте інший запит.
        </Alert>
      ) : null}

      {userId && (
        <Fade in={!!userId}>
          <Alert
            severity="success"
            sx={{ borderRadius: 2 }}
            icon={<CheckCircleIcon />}
          >
            Обрано користувача: {userId}
          </Alert>
        </Fade>
      )}
    </Stack>
  );
};