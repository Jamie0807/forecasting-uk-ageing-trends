import { useEffect, useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import { fetchAgeingRatio, fetchOverview } from '../api/client'
import { TrendingUp, Users, Calendar } from 'lucide-react'

const REGION_COLORS = {
  England: '#3b82f6',
  Scotland: '#10b981',
  Wales: '#f59e0b',
}

function StatCard({ region, latest_percent, latest_year, total_change }) {
  const color = REGION_COLORS[region] ?? '#6b7280'
  const sign = total_change >= 0 ? '+' : ''
  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-5 flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="font-semibold text-gray-700">{region}</span>
        <span
          className="text-xs px-2 py-0.5 rounded-full font-medium text-white"
          style={{ background: color }}
        >
          {latest_year}
        </span>
      </div>
      <div className="text-3xl font-bold" style={{ color }}>
        {latest_percent}%
      </div>
      <div className="text-sm text-gray-500 flex items-center gap-1">
        <TrendingUp size={13} />
        {sign}{total_change}% since start
      </div>
    </div>
  )
}

// Transform flat records [{Year, region, percent65plus}] into recharts format
function pivotForChart(records) {
  const map = {}
  for (const r of records) {
    if (!map[r.Year]) map[r.Year] = { year: r.Year }
    map[r.Year][r.region] = r.percent65plus
  }
  return Object.values(map).sort((a, b) => a.year - b.year)
}

export default function Dashboard() {
  const [chartData, setChartData] = useState([])
  const [stats, setStats] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([fetchAgeingRatio(), fetchOverview()])
      .then(([ratio, overview]) => {
        setChartData(pivotForChart(ratio))
        setStats(overview)
      })
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading)
    return <p className="text-center text-gray-500 py-20">Loading data…</p>
  if (error)
    return (
      <p className="text-center text-red-500 py-20">
        Failed to load data: {error}. Make sure the backend is running on port 8000.
      </p>
    )

  const regions = Object.keys(REGION_COLORS)

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-500 mt-1">
          Historical 65+ population share across UK regions (ONS data)
        </p>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {stats.map(s => (
          <StatCard key={s.region} {...s} />
        ))}
      </div>

      {/* Trend chart */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
        <h2 className="font-semibold text-gray-700 mb-4 flex items-center gap-2">
          <Users size={16} /> 65+ Population Share by Region
        </h2>
        <ResponsiveContainer width="100%" height={360}>
          <LineChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis
              dataKey="year"
              tick={{ fontSize: 12 }}
              tickFormatter={v => String(v)}
            />
            <YAxis
              tickFormatter={v => `${v}%`}
              domain={['auto', 'auto']}
              tick={{ fontSize: 12 }}
            />
            <Tooltip formatter={v => [`${Number(v).toFixed(2)}%`]} />
            <Legend />
            {regions.map(region => (
              <Line
                key={region}
                type="monotone"
                dataKey={region}
                stroke={REGION_COLORS[region]}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4 }}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Info footer */}
      <p className="text-xs text-gray-400 flex items-center gap-1">
        <Calendar size={12} />
        Data source: UK Office for National Statistics (ONS)
      </p>
    </div>
  )
}
