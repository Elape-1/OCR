/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: {
          50: '#f8fafc',
          100: '#e2e8f0',
          200: '#cbd5e1',
          300: '#94a3b8',
          400: '#64748b',
          500: '#475569',
          600: '#334155',
          700: '#1e293b',
          800: '#0f172a',
          900: '#020617',
        },
        accent: {
          500: '#f59e0b',
          600: '#d97706',
        },
      },
      boxShadow: {
        panel: '0 24px 80px rgba(15, 23, 42, 0.45)',
      },
    },
  },
  plugins: [],
}
