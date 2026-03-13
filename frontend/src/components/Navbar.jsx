import { NavLink } from 'react-router-dom'
import { BarChart2, TrendingUp, GitBranch } from 'lucide-react'

const links = [
  { to: '/dashboard', label: 'Dashboard', Icon: BarChart2 },
  { to: '/forecast', label: 'Forecast', Icon: TrendingUp },
  { to: '/cluster', label: 'Cluster Analysis', Icon: GitBranch },
]

export default function Navbar() {
  return (
    <header className="bg-brand-900 text-white shadow-md">
      <div className="container mx-auto px-4 max-w-7xl flex items-center gap-6 h-14">
        <span className="font-bold text-lg tracking-tight whitespace-nowrap">
          🇬🇧 UK Ageing Trends
        </span>
        <nav className="flex gap-1">
          {links.map(({ to, label, Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `flex items-center gap-1.5 px-3 py-1.5 rounded text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-white/20 text-white'
                    : 'text-blue-200 hover:text-white hover:bg-white/10'
                }`
              }
            >
              <Icon size={15} />
              {label}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  )
}
