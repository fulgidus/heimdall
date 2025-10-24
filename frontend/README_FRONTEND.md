# Frontend - Heimdall Analytics Platform

Modern, responsive dashboard built with React, TypeScript, and TailwindCSS.

## 🎨 Design System

### Color Palette
- **Oxford Blue**: `#0b132bff` - Primary dark background
- **Sea Green**: `#09814aff` - Secondary accent
- **French Gray**: `#c7cedbff` - Neutral text
- **Light Green**: `#70ee9cff` - Success/positive
- **Neon Blue**: `#446df6ff` - Primary interactive

## 🛠️ Tech Stack

- **Frontend Framework**: React 19 + TypeScript
- **Build Tool**: Vite
- **Styling**: SCSS
- **State Management**: Zustand
- **HTTP Client**: Axios
- **Routing**: React Router v6
- **Icons**: Lucide React
- **Query Management**: TanStack Query (React Query)

## 📁 Project Structure

```
frontend/
├── src/
│   ├── components/        # Reusable UI components
│   │   ├── Button.tsx
│   │   ├── Card.tsx
│   │   ├── Input.tsx
│   │   ├── Badge.tsx
│   │   ├── Header.tsx
│   │   ├── Sidebar.tsx
│   │   └── MainLayout.tsx
│   ├── pages/            # Route pages
│   │   ├── Dashboard.tsx
│   │   ├── Analytics.tsx
│   │   ├── Settings.tsx
│   │   ├── Projects.tsx
│   │   ├── Profile.tsx
│   │   └── Login.tsx
│   ├── store/            # Zustand stores
│   │   ├── authStore.ts
│   │   └── dashboardStore.ts
│   ├── lib/              # Utilities and helpers
│   │   └── api.ts
│   ├── types/            # TypeScript definitions
│   ├── styles/           # Global styles
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
├── public/
├── tailwind.config.js
├── postcss.config.js
├── vite.config.ts
└── package.json
```

## 🚀 Getting Started

### Prerequisites
- Node.js 18+
- npm or yarn

### Installation

```bash
cd frontend
npm install
```

### Development

```bash
npm run dev
```

The app will be available at `http://localhost:3000`

### Build

```bash
npm run build
```

### Production Preview

```bash
npm run preview
```

## 📦 Dependencies

### Core
- `react@^19.1.1` - React library
- `react-dom@^19.1.1` - React DOM
- `react-router-dom@^6.24.1` - Client-side routing
- `typescript@~5.6.3` - Type safety

### UI & Styling
- `tailwindcss@^3.4.1` - Utility-first CSS
- `postcss@^8.4.31` - CSS processor
- `autoprefixer@^10.4.16` - Vendor prefixes
- `lucide-react@^0.263.1` - Icon library
- `classnames@^2.3.2` - Class name utility

### State Management & Data
- `zustand@^4.4.1` - Light state management
- `axios@^1.6.0` - HTTP client
- `@tanstack/react-query@^5.28.0` - Server state management

## 🎯 Features

### Implemented
- ✅ Modern, responsive UI with dark theme
- ✅ Authentication & protected routes
- ✅ Dashboard with metrics and charts
- ✅ Analytics page with data visualization
- ✅ Project management page
- ✅ User profile & settings
- ✅ Sidebar navigation (mobile responsive)
- ✅ Header with notifications
- ✅ Zustand store for state management
- ✅ API client with interceptors

### Planned
- 📊 Chart.js / Recharts integration for live charts
- 📱 Mobile app (React Native)
- 🔔 Real-time notifications
- 🎨 Theme customization
- 🌐 Multi-language support
- ♿ Enhanced accessibility

## 🔐 Authentication

The app uses JWT-based authentication stored in Zustand. Protected routes redirect to login when not authenticated.

```typescript
const { isAuthenticated, login, logout } = useAuthStore();
```

## 🎨 Component Usage

### Button
```tsx
<Button variant="accent" size="md">
  Click me
</Button>
```

Variants: `primary`, `secondary`, `accent`, `success`, `danger`
Sizes: `xs`, `sm`, `md`, `lg`, `xl`

### Card
```tsx
<Card variant="elevated">
  Content here
</Card>
```

Variants: `default`, `bordered`, `elevated`

### Input
```tsx
<Input 
  label="Email" 
  type="email" 
  error={error} 
  helperText="Helper text"
/>
```

## 🚀 Deployment

### Docker

```dockerfile
# Build stage
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# Runtime stage
FROM node:18-alpine
WORKDIR /app
RUN npm install -g serve
COPY --from=builder /app/dist ./dist
EXPOSE 3000
CMD ["serve", "-s", "dist", "-l", "3000"]
```

### Environment Variables

Create `.env` file:

```env
VITE_API_URL=http://api.heimdall.com/api
VITE_ENV=production
```

## 📝 Development Guidelines

1. **Components**: Create reusable, composable components in `src/components`
2. **Pages**: One component per page in `src/pages`
3. **State**: Use Zustand for global state
4. **Styling**: Use TailwindCSS utility classes
5. **Types**: Define interfaces in `src/types`

## 🤝 Contributing

1. Create a feature branch
2. Make your changes
3. Test locally with `npm run dev`
4. Build for production `npm run build`
5. Submit a pull request

## 📄 License

MIT

## 📞 Support

For issues and questions, please create an issue in the repository.
