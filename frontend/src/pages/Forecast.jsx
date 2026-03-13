import { useEffect, useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ReferenceLine,
} from 'recharts'
import { fetchProphetForecast, fetchArimaForecast, fetchMetrics } from '../api/client'
import { TrendingUp } from 'lucide-react'

const REGIONS = ['England', 'Wales', 'Scotland']
const MODELS = ['prophet', 'arima']

const MODEL_LABELS = { prophet: 'Prophet', arima: 'ARIMA' }

// Merge historical + forecast into a single recharts dataset
function buildChartData(series) {
  const map = {}
  for (const point of series) {
    const key = point.year
    if (!map[key]) map[key] = { year: key }
    if (point.type === 'historical') map[key].historical = point.value
    else map[key].forecast = point.value
  }
  return Object.values(map).sort((a, b) => a.year - b.year)
}

function MetricsTable({ metrics }) {
  if (!metrics.length) return null
  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6 overflow-x-auto">
      <h2 className="font-semibold text-gray-700 mb-4">Model Evaluation Metrics</h2>
      <table className="min-w-full text-sm">
        <thead>
          <tr className="border-b text-gray-500 uppercase text-xs">
            <th className="text-left py-2 pr-4">Region</th>
            <th className="text-right py-2 px-3">Prophet MAE</th>
            <th className="text-right py-2 px-3">Prophet RMSE</th>
            <th className="text-right py-2 px-3">Prophet MAPE</th>
            <th className="text-right py-2 px-3">ARIMA MAE</th>
            <th className="text-right py-2 px-3">ARIMA RMSE</th>
            <th className="text-right py-2 px-3">ARIMA MAPE</th>
          </tr>
        </thead>
        <tbody>
          {metrics.map(m => (
            <tr key={m.region} className="border-b last:border-0 hover:bg-gray-50">
              <td className="py-2 pr-4 font-medium">{m.region}</td>
              <td className="text-right px-3">{m.prophet_mae}</td>
              <td className="text-right px-3">{m.prophet_rmse}</td>
              <td className="text-right px-3">{m.prophet_mape?.toFixed ? m.prophet_mape.toFixed(2) + '%' : m.prophet_mape}</td>
              <td className="text-right px-3">{m.arima_mae}</td>
              <td className="text-right px-3">{m.arima_rmse}</td>
              <td className="text-right px-3">{m.arima_mape?.toFixed ? m.arima_mape.toFixed(2) + '%' : m.arima_mape}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function Forecast() {
  const [region, setRegion] = useState('England')
  const [model, setModel] = useState('prophet')
  const [chartData, setChartData] = useState([])
  const [metrics, setMetrics] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Determine the split year (first forecast year)
  const splitYear = chartData.find(d => d.forecast != null)?.year ?? null

  useEffect(() => {
    setLoading(true)
    setError(null)
    const fetcher = model === 'prophet' ? fetchProphetForecast : fetchArimaForecast
    fetcher(region)
      .then(series => setChartData(buildChartData(series)))
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }, [region, model])

  useEffect(() => {
    fetchMetrics()
      .then(setMetrics)
      .catch(() => {})
  }, [])

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
          <TrendingUp size={22} /> Forecast
        </h1>
        <p className="text-gray-500 mt-1">
          Long-term 65+ population share forecast using Prophet and ARIMA models (2020–2150)
        </p>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-4">
        <div className="flex flex-col gap-1">
          <label className="text-xs font-semibold text-gray-500 uppercase">Region</label>
          <div className="flex gap-2">
            {REGIONS.map(r => (
              <button
                key={r}
                onClick={() => setRegion(r)}
                className={`px-4 py-1.5 rounded-full text-sm font-medium border transition-colors ${
                  region === r
                    ? 'bg-brand-600 text-white border-brand-600'
                    : 'bg-white text-gray-600 border-gray-200 hover:border-brand-400'
                }`}
              >
                {r}
              </button>
            ))}
          </div>
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-xs font-semibold text-gray-500 uppercase">Model</label>
          <div className="flex gap-2">
            {MODELS.map(m => (
              <button
                key={m}
                onClick={() => setModel(m)}
                className={`px-4 py-1.5 rounded-full text-sm font-medium border transition-colors ${
                  model === m
                    ? 'bg-brand-600 text-white border-brand-600'
                    : 'bg-white text-gray-600 border-gray-200 hover:border-brand-400'
                }`}
              >
                {MODEL_LABELS[m]}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Chart */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
        <h2 className="font-semibold text-gray-700 mb-4">
          {region} — {MODEL_LABELS[model]} Forecast
        </h2>

        {loading && <p className="text-center text-gray-400 py-20">Loading forecast…</p>}
        {error && <p className="text-center text-red-400 py-10">Error: {error}</p>}

        {!loading && !error && (
          <ResponsiveContainer width="100%" height={380}>
            <LineChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="year" tick={{ fontSize: 11 }} />
              <YAxis tickFormatter={v => `${v}%`} domain={['auto', 'auto']} tick={{ fontSize: 11 }} />
              <Tooltip formatter={v => [`${Number(v).toFixed(2)}%`]} />
              <Legend />
              {splitYear && (
                <ReferenceLine
                  x={splitYear}
                  stroke="#9ca3af"
                  strokeDasharray="5 5"
                  label={{ value: 'Forecast start', position: 'insideTopRight', fontSize: 11, fill: '#6b7280' }}
                />
              )}
              <Line
                type="monotone"
                dataKey="historical"
                name="Historical"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4 }}
              />
              <Line
                type="monotone"
                dataKey="forecast"
                name={`${MODEL_LABELS[model]} Forecast`}
                stroke="#f59e0b"
                strokeWidth={2}
                strokeDasharray="6 3"
                dot={false}
                activeDot={{ r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Metrics */}
      <MetricsTable metrics={metrics} />
    </div>
  )
}
