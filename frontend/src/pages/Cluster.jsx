import { useEffect, useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import { fetchCluster } from '../api/client'
import { GitBranch } from 'lucide-react'

// Build recharts data: [{year, RegionA, RegionB, ...}]
function buildChartData(clusterResults) {
  const map = {}
  for (const item of clusterResults) {
    for (const point of item.trend) {
      if (!map[point.year]) map[point.year] = { year: point.year }
      map[point.year][item.region] = point.value
    }
  }
  return Object.values(map).sort((a, b) => a.year - b.year)
}

const CLUSTER_NAMES = ['Cluster A', 'Cluster B', 'Cluster C', 'Cluster D', 'Cluster E']

export default function Cluster() {
  const [results, setResults] = useState([])
  const [chartData, setChartData] = useState([])
  const [nClusters, setNClusters] = useState(3)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    fetchCluster(nClusters)
      .then(data => {
        setResults(data)
        setChartData(buildChartData(data))
      })
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }, [nClusters])

  // Group regions by cluster for the legend / summary
  const clusters = {}
  for (const item of results) {
    if (!clusters[item.cluster]) clusters[item.cluster] = []
    clusters[item.cluster].push(item)
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
          <GitBranch size={22} /> Cluster Analysis
        </h1>
        <p className="text-gray-500 mt-1">
          KMeans clustering of UK regions by their 65+ ageing trend trajectories
        </p>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-4">
        <label className="text-sm font-semibold text-gray-600">Number of clusters:</label>
        {[2, 3, 4].map(n => (
          <button
            key={n}
            onClick={() => setNClusters(n)}
            className={`w-9 h-9 rounded-full text-sm font-semibold border transition-colors ${
              nClusters === n
                ? 'bg-brand-600 text-white border-brand-600'
                : 'bg-white text-gray-600 border-gray-200 hover:border-brand-400'
            }`}
          >
            {n}
          </button>
        ))}
      </div>

      {loading && <p className="text-center text-gray-400 py-20">Running cluster analysis…</p>}
      {error && <p className="text-center text-red-400 py-10">Error: {error}</p>}

      {!loading && !error && (
        <>
          {/* Trend chart coloured by cluster */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
            <h2 className="font-semibold text-gray-700 mb-4">
              Ageing Trend per Region (coloured by cluster)
            </h2>
            <ResponsiveContainer width="100%" height={360}>
              <LineChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="year" tick={{ fontSize: 11 }} />
                <YAxis tickFormatter={v => `${v}%`} domain={['auto', 'auto']} tick={{ fontSize: 11 }} />
                <Tooltip formatter={v => [`${Number(v).toFixed(2)}%`]} />
                <Legend />
                {results.map(item => (
                  <Line
                    key={item.region}
                    type="monotone"
                    dataKey={item.region}
                    stroke={item.color}
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 4 }}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Cluster summary cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(clusters).map(([clusterId, items]) => (
              <div
                key={clusterId}
                className="bg-white rounded-xl shadow-sm border border-gray-100 p-5"
              >
                <div className="flex items-center gap-2 mb-3">
                  <span
                    className="w-3 h-3 rounded-full"
                    style={{ background: items[0].color }}
                  />
                  <span className="font-semibold text-gray-700">
                    {CLUSTER_NAMES[Number(clusterId)] ?? `Cluster ${clusterId}`}
                  </span>
                </div>
                <ul className="space-y-1">
                  {items.map(item => {
                    const lastPoint = item.trend[item.trend.length - 1]
                    const firstPoint = item.trend[0]
                    return (
                      <li key={item.region} className="text-sm text-gray-600 flex justify-between">
                        <span>{item.region}</span>
                        <span className="text-gray-400 text-xs">
                          {firstPoint?.value?.toFixed(1)}% → {lastPoint?.value?.toFixed(1)}%
                        </span>
                      </li>
                    )
                  })}
                </ul>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
