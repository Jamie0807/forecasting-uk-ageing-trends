import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
})

export const fetchAgeingRatio = () => api.get('/ageing-ratio').then(r => r.data)
export const fetchRegions = () => api.get('/regions').then(r => r.data)
export const fetchOverview = () => api.get('/overview').then(r => r.data)
export const fetchProphetForecast = (region) =>
  api.get('/forecast/prophet', { params: { region } }).then(r => r.data)
export const fetchArimaForecast = (region) =>
  api.get('/forecast/arima', { params: { region } }).then(r => r.data)
export const fetchMetrics = () => api.get('/metrics').then(r => r.data)
export const fetchCluster = (nClusters = 3) =>
  api.get('/cluster', { params: { n_clusters: nClusters } }).then(r => r.data)
