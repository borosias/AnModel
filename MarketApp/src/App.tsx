import { useMemo, useState } from 'react'
import './App.css'

/**
 * Базовые типы для метрик дашборда
 */
type Metric = {
  label: string
  value: number
  delta: number // percent vs. prev period
  suffix?: string
}

type SeriesPoint = { date: string; value: number }
type Trend = { name: string; change: number; series: number[] }
type User = { id: string; name: string }
type Recommendation = { title: string; description: string; impact: 'low' | 'medium' | 'high' }

/**
 * Типы для работы с моделями
 */
type ModelId = 'context_aware' | string

type ModelInfo = {
  id: ModelId
  name: string
  description: string
}

/**
 * Ответ предикта для любой модели.
 * Для конкретной модели можно сузить тип, но базово оставляем generic.
 */
type PredictionRecord = {
  [key: string]: unknown
}

/**
 * Ответ API /predict
 */
type PredictResponse = {
  predictions: PredictionRecord[]
}

/**
 * Хелперы форматирования
 */
const nf = new Intl.NumberFormat('ru-RU')
const pf = new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 1 })
const kfmt = (n: number) =>
  n >= 1_000_000 ? `${pf.format(n / 1_000_000)}M` : n >= 1_000 ? `${pf.format(n / 1_000)}K` : nf.format(n)

/**
 * Компоненты UI
 */
function DeltaBadge({ value }: { value: number }) {
  const up = value >= 0
  return (
    <span className={`delta ${up ? 'up' : 'down'}`}>
      {up ? '▲' : '▼'} {pf.format(Math.abs(value))}%
    </span>
  )
}

function Sparkline({
  data,
  width = 120,
  height = 36,
  color = '#7c3aed',
}: {
  data: number[]
  width?: number
  height?: number
  color?: string
}) {
  const max = Math.max(...data)
  const min = Math.min(...data)
  const range = max - min || 1
  const points = data
    .map((v, i) => {
      const x = (i / (data.length - 1)) * width
      const y = height - ((v - min) / range) * height
      return `${x},${y}`
    })
    .join(' ')
  return (
    <svg width={width} height={height} className="sparkline">
      <polyline fill="none" stroke={color} strokeWidth="2" points={points} />
    </svg>
  )
}

function LineChart({
  series,
  width = 820,
  height = 220,
  color = '#22c55e',
}: {
  series: SeriesPoint[]
  width?: number
  height?: number
  color?: string
}) {
  const values = series.map((p) => p.value)
  const max = Math.max(...values)
  const min = Math.min(...values)
  const range = max - min || 1
  const points = series
    .map((p, i) => {
      const x = (i / (series.length - 1)) * (width - 40) + 20 // padding
      const y = height - ((p.value - min) / range) * (height - 40) - 20
      return `${x},${y}`
    })
    .join(' ')
  return (
    <div className="chart-card">
      <div className="chart-header">
        <h3>Доход, ₽ — последние 12 месяцев</h3>
      </div>
      <svg width={width} height={height} className="line-chart">
        <defs>
          <linearGradient id="grad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.35" />
            <stop offset="100%" stopColor={color} stopOpacity="0" />
          </linearGradient>
        </defs>
        <polyline fill="none" stroke={color} strokeWidth="3" points={points} />
        <polygon fill="url(#grad)" points={`${points} ${width - 20},${height - 20} 20,${height - 20}`} />
        {series.map((p, i) => {
          const x = (i / (series.length - 1)) * (width - 40) + 20
          const y = height - ((p.value - min) / range) * (height - 40) - 20
          return <circle key={p.date} cx={x} cy={y} r={3} fill={color} />
        })}
      </svg>
      <div className="chart-legend">
        {series.map((p) => (
          <span key={p.date} className="legend-item">
            {p.date}
          </span>
        ))}
      </div>
    </div>
  )
}

/**
 * Клиент для API моделей
 * Внутри зашит префикс /api — в dev его проксирует Vite на Python‑сервер.
 */
async function callPredictApi(modelId: ModelId, record: Record<string, unknown>): Promise<PredictionRecord> {
  const resp = await fetch('/api/predict', {
    method: 'POST',
    headers: { ContentType: 'application/json',},
    body: JSON.stringify({
      model_id: modelId,
      records: [record],
    }),
  })

  if (!resp.ok) {
    const text = await resp.text().catch(() => '')
    throw new Error(`HTTP ${resp.status}${text ? `: ${text}` : ''}`)
  }

  const data = (await resp.json()) as PredictResponse
  if (!data.predictions || data.predictions.length === 0) {
    throw new Error('Сервер вернул пустой список предсказаний')
  }
  return data.predictions[0]
}

function App() {
  /**
   * Модели: сейчас одна, но структура сразу позволяет добавлять другие.
   */
  const models: ModelInfo[] = [
    {
      id: 'context_aware',
      name: 'ContextAware · Покупка в 7 дней',
      description: 'Вероятность покупки, дни до покупки и сумма следующей покупки.',
    },
    // Пример на будущее:
    // { id: 'another_model', name: 'Другая модель', description: 'Описание другой модели.' },
  ]

  // Mock users
  const users: User[] = [
    { id: 'u1', name: 'Алексей' },
    { id: 'u2', name: 'Мария' },
    { id: 'u3', name: 'Иван' },
    { id: 'u4', name: 'Наталья' },
  ]

  // Mock KPIs
  const metrics: Metric[] = [
    { label: 'Выручка', value: 12_450_300, delta: 8.4, suffix: '₽' },
    { label: 'Активные пользователи', value: 48_920, delta: 3.1 },
    { label: 'Конверсия', value: 4.8, delta: -0.6, suffix: '%' },
    { label: 'CAC', value: 720, delta: -5.2, suffix: '₽' },
    { label: 'LTV', value: 5_400, delta: 2.7, suffix: '₽' },
  ]

  // Revenue series for 12 months
  const revenueSeries: SeriesPoint[] = useMemo(
    () => [
      { date: 'Янв', value: 640_000 },
      { date: 'Фев', value: 680_000 },
      { date: 'Мар', value: 720_000 },
      { date: 'Апр', value: 690_000 },
      { date: 'Май', value: 750_000 },
      { date: 'Июн', value: 810_000 },
      { date: 'Июл', value: 870_000 },
      { date: 'Авг', value: 910_000 },
      { date: 'Сен', value: 960_000 },
      { date: 'Окт', value: 1_010_000 },
      { date: 'Ноя', value: 1_080_000 },
      { date: 'Дек', value: 1_220_000 },
    ],
    []
  )

  // Traffic/Channel trends
  const trends: Trend[] = [
    { name: 'Органика', change: 5.2, series: [12, 14, 13, 15, 16, 18, 21, 20, 22, 23, 24, 26] },
    { name: 'Платный трафик', change: -3.8, series: [20, 19, 21, 18, 17, 16, 18, 17, 16, 15, 14, 13] },
    { name: 'Рефералы', change: 7.9, series: [4, 4, 5, 5, 6, 7, 8, 8, 9, 10, 10, 11] },
    { name: 'Соцсети', change: 2.1, series: [6, 6, 7, 8, 8, 9, 10, 10, 11, 11, 12, 12] },
  ]

  // Mock recommendations per user
  const userRecommendations: Record<string, Recommendation[]> = {
    u1: [
      { title: 'Усилить SEO по топ-3 категориям', description: 'Рост органики на 10–15% за 6–8 недель.', impact: 'high' },
      { title: 'Запустить look-alike в VK Ads', description: 'Снижение CAC на 5–7% при бюджете 300k₽.', impact: 'medium' },
      { title: 'Email-реактивация', description: 'Вернуть до 2% «спящих» пользователей.', impact: 'low' },
    ],
    u2: [
      { title: 'Реферальная программа', description: 'Увеличение LTV на 3–5%.', impact: 'medium' },
      { title: 'AB-тест первого экрана', description: 'Рост конверсии в регистрацию на 1–2 п.п.', impact: 'medium' },
    ],
    u3: [
      { title: 'Сократить неэффективные ключевые слова', description: 'Экономия бюджета до 8%.', impact: 'high' },
      { title: 'Push-цепочки', description: 'Рост ретеншена D7 на 0.5–1 п.п.', impact: 'low' },
    ],
    u4: [
      { title: 'Сегментация по RFM', description: 'Точечные офферы — +2–3% к повторным покупкам.', impact: 'high' },
    ],
  }

  /**
   * State
   */
  const [activeTab, setActiveTab] = useState<'dashboard' | 'reco' | 'models'>('dashboard')
  const [selectedUser, setSelectedUser] = useState<User>(users[0])

  // Выбор модели
  const [selectedModelId, setSelectedModelId] = useState<ModelId>(models[0]?.id ?? 'context_aware')

  // Поля для скоринга context_aware (можно расширять)
  const [totalEvents, setTotalEvents] = useState('100')
  const [totalPurchases, setTotalPurchases] = useState('3')
  const [daysSinceLast, setDaysSinceLast] = useState('5')
  const [avgSpend30d, setAvgSpend30d] = useState('1200')

  const [prediction, setPrediction] = useState<PredictionRecord | null>(null)
  const [loadingPredict, setLoadingPredict] = useState(false)
  const [predictError, setPredictError] = useState<string | null>(null)

  const selectedModel = models.find((m) => m.id === selectedModelId) ?? models[0]

  /**
   * Обработчик сабмита формы скоринга.
   * Внутри формируем запись с фичами для соответствующей модели.
   */
  const handleScore = async () => {
    setLoadingPredict(true)
    setPredictError(null)
    setPrediction(null)

    try {
      let record: Record<string, unknown>

      // На будущее: можно развести разные форматы рекорда по modelId
      if (selectedModelId === 'context_aware') {
        record = {
          // имена полей подогнаны под фичи snapshot-модели
          total_events: Number(totalEvents),
          total_purchases: Number(totalPurchases),
          days_since_last_event: Number(daysSinceLast),
          avg_spend_per_purchase_30d: Number(avgSpend30d),
        }
      } else {
        // По умолчанию — пустой объект (можно заменить на универсальную JSON-форму)
        record = {}
      }

      const pred = await callPredictApi(selectedModelId, record)
      setPrediction(pred)
    } catch (err) {
      setPredictError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoadingPredict(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <span className="brand-dot" /> AnModel — Маркетинговая панель
        </div>
        <nav className="tabs">
          <button className={`tab ${activeTab === 'dashboard' ? 'active' : ''}`} onClick={() => setActiveTab('dashboard')}>
            Дашборд
          </button>
          <button className={`tab ${activeTab === 'reco' ? 'active' : ''}`} onClick={() => setActiveTab('reco')}>
            Рекомендации
          </button>
          <button className={`tab ${activeTab === 'models' ? 'active' : ''}`} onClick={() => setActiveTab('models')}>
            Модели
          </button>
        </nav>
      </header>

      {activeTab === 'dashboard' && (
        <main className="content">
          <section className="kpi-grid">
            {metrics.map((m) => (
              <div key={m.label} className="card kpi">
                <div className="kpi-top">
                  <h3>{m.label}</h3>
                  <DeltaBadge value={m.delta} />
                </div>
                <div className="kpi-value">
                  {m.suffix === '%'
                    ? `${pf.format(m.value)}%`
                    : m.suffix === '₽'
                    ? `${kfmt(m.value)} ₽`
                    : kfmt(m.value)}
                </div>
                <Sparkline data={[...Array(10)].map((_, i) => m.value * (0.8 + i * 0.02))} />
              </div>
            ))}
          </section>

          <section className="grid two">
            <div className="card">
              <LineChart series={revenueSeries} />
            </div>
            <div className="card">
              <h3>Тренды каналов</h3>
              <ul className="trend-list">
                {trends.map((t) => (
                  <li key={t.name} className="trend-item">
                    <div className="trend-info">
                      <strong>{t.name}</strong>
                      <DeltaBadge value={t.change} />
                    </div>
                    <Sparkline data={t.series} color={t.change >= 0 ? '#22c55e' : '#ef4444'} />
                  </li>
                ))}
              </ul>
            </div>
          </section>

          <section className="grid two">
            <div className="card">
              <h3>Аномалии за период</h3>
              <ul className="bulleted">
                <li>Резкий рост возвратов в «Электроника» (+18% неделя к неделе).</li>
                <li>Падение кликабельности в поисковых кампаниях (-0.7 п.п.).</li>
                <li>Обновление приложения повысило CR на iOS (+1.2 п.п.).</li>
              </ul>
            </div>
            <div className="card">
              <h3>Прогноз на следующий месяц</h3>
              <p className="lead">Выручка: 1.25–1.32 млн ₽ при текущих настройках.</p>
              <p>
                Основные драйверы: органический трафик и ретеншен существующих клиентов. Рекомендуем фокус на SEO и
                email-реактивацию.
              </p>
            </div>
          </section>
        </main>
      )}

      {activeTab === 'reco' && (
        <main className="content">
          <section className="card">
            <div className="row">
              <h3 className="mr">Рекомендации по пользователю</h3>
              <select
                value={selectedUser.id}
                onChange={(e) => setSelectedUser(users.find((u) => u.id === e.target.value) || users[0])}
              >
                {users.map((u) => (
                  <option key={u.id} value={u.id}>
                    {u.name}
                  </option>
                ))}
              </select>
            </div>
            <div className="user-summary">
              <div className="summary-kpis">
                <div className="pill">ARPU: 275 ₽</div>
                <div className="pill">CR: 4.8%</div>
                <div className="pill">Retention D30: 21%</div>
              </div>
              <Sparkline data={[18, 17, 19, 20, 22, 23, 24, 25, 26, 27, 29, 30]} width={220} height={54} color="#0ea5e9" />
            </div>
            <ul className="reco-list">
              {(userRecommendations[selectedUser.id] || []).map((r, idx) => (
                <li key={idx} className={`reco-item ${r.impact}`}>
                  <div className="reco-title">
                    {r.title}
                    <span className="impact">
                      {r.impact === 'high' ? 'высокий' : r.impact === 'medium' ? 'средний' : 'низкий'} эффект
                    </span>
                  </div>
                  <div className="reco-desc">{r.description}</div>
                </li>
              ))}
            </ul>
            <div className="footnote">Данные и прогнозы — мок для демонстрации интерфейса.</div>
          </section>
        </main>
      )}

      {activeTab === 'models' && (
        <main className="content">
          <section className="card">
            <div className="row">
              <h3 className="mr">Онлайн-скоринг моделей</h3>
              <select value={selectedModelId} onChange={(e) => setSelectedModelId(e.target.value)}>
                {models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </select>
            </div>
            <p style={{ marginTop: 8, opacity: 0.8 }}>{selectedModel?.description}</p>

            {/* Форма ввода фич для выбранной модели.
                Пока реализован специализированный блок под context_aware. */}
            {selectedModelId === 'context_aware' && (
              <>
                <div className="row" style={{ marginTop: 12, gap: 16, flexWrap: 'wrap' }}>
                  <label>
                    Событий всего
                    <input
                      type="number"
                      value={totalEvents}
                      onChange={(e) => setTotalEvents(e.target.value)}
                      style={{ display: 'block', width: 120, marginTop: 4 }}
                    />
                  </label>
                  <label>
                    Покупок всего
                    <input
                      type="number"
                      value={totalPurchases}
                      onChange={(e) => setTotalPurchases(e.target.value)}
                      style={{ display: 'block', width: 120, marginTop: 4 }}
                    />
                  </label>
                  <label>
                    Дней с последнего события
                    <input
                      type="number"
                      value={daysSinceLast}
                      onChange={(e) => setDaysSinceLast(e.target.value)}
                      style={{ display: 'block', width: 160, marginTop: 4 }}
                    />
                  </label>
                  <label>
                    Средний чек за 30 дн, ₽
                    <input
                      type="number"
                      value={avgSpend30d}
                      onChange={(e) => setAvgSpend30d(e.target.value)}
                      style={{ display: 'block', width: 160, marginTop: 4 }}
                    />
                  </label>
                </div>
              </>
            )}

            <div style={{ marginTop: 16 }}>
              <button disabled={loadingPredict} onClick={handleScore}>
                {loadingPredict ? 'Считаем…' : 'Посчитать предсказание'}
              </button>
            </div>

            {predictError && (
              <div style={{ marginTop: 12, color: '#ef4444' }}>
                <strong>Ошибка:</strong> {predictError}
              </div>
            )}

            {prediction && (
              <div style={{ marginTop: 16 }}>
                <h4>Результат предсказания</h4>
                {/* Если это context_aware, красиво выводим основные поля.
                    Для других моделей просто покажем JSON. */}
                {selectedModelId === 'context_aware' ? (
                  <ul className="bulleted">
                    <li>
                      Вероятность покупки в 7 дней:{' '}
                      <strong>
                        {typeof prediction.purchase_proba === 'number'
                          ? (prediction.purchase_proba * 100).toFixed(1)
                          : '-'}
                        %
                      </strong>
                    </li>
                    <li>
                      Флаг покупки:{' '}
                      <strong>
                        {typeof prediction.will_purchase_pred === 'number' || typeof prediction.will_purchase_pred === 'boolean'
                          ? Number(prediction.will_purchase_pred) === 1
                            ? 'Да'
                            : 'Нет'
                          : '-'}
                      </strong>
                    </li>
                    <li>
                      Дней до следующей покупки:{' '}
                      <strong>
                        {typeof prediction.days_to_next_pred === 'number'
                          ? prediction.days_to_next_pred.toFixed(1)
                          : '-'}
                      </strong>
                    </li>
                    <li>
                      Сумма следующей покупки:{' '}
                      <strong>
                        {typeof prediction.next_purchase_amount_pred === 'number'
                          ? `${prediction.next_purchase_amount_pred.toFixed(2)} ₽`
                          : '-'}
                      </strong>
                    </li>
                  </ul>
                ) : (
                  <pre style={{ marginTop: 8, fontSize: 12, whiteSpace: 'pre-wrap' }}>
                    {JSON.stringify(prediction, null, 2)}
                  </pre>
                )}
              </div>
            )}
          </section>
        </main>
      )}

      <footer className="footer">© {new Date().getFullYear()} AnModel · Демонстрационный интерфейс</footer>
    </div>
  )
}

export default App