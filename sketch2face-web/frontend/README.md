# Face Mind AI - Frontend

Advanced AI-powered facial recognition and matching system built with React, Vite, and Tailwind CSS.

## Features

- 🔐 User Authentication (Login/Register)
- 📸 Face Image Upload with Drag & Drop
- 🎯 AI-Powered Face Matching
- 📊 Match History Tracking
- 👥 Admin Dashboard
- 🎨 Modern, Responsive UI
- 📈 Top-K Similar Faces Gallery
- 💬 User Feedback System

## Tech Stack

- **React 18** - UI Library
- **Vite** - Build Tool
- **React Router** - Navigation
- **Axios** - HTTP Client
- **Tailwind CSS** - Styling
- **Lucide React** - Icons

## Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd frontend
```

2. **Install dependencies**
```bash
npm install
```

3. **Set up environment variables**
```bash
cp .env.example .env
```

Edit `.env` and set your API URL:
```
VITE_API_URL=http://localhost:5000/api
```

4. **Run the development server**
```bash
npm run dev
```

The app will open at `http://localhost:3000`

## Project Structure

```
frontend/
├── src/
│   ├── api/                 # Backend API communication
│   │   ├── axios.js         # Axios instance with interceptors
│   │   ├── auth.api.js      # Authentication endpoints
│   │   ├── match.api.js     # Face matching endpoints
│   │   └── admin.api.js     # Admin endpoints
│   │
│   ├── auth/                # Authentication utilities
│   │   └── requireAuth.jsx  # Protected route wrapper
│   │
│   ├── components/          # Reusable UI components
│   │   ├── Navbar.jsx       # Navigation bar
│   │   ├── UploadBox.jsx    # Image upload component
│   │   ├── MatchResult.jsx  # Match result display
│   │   ├── TopKGallery.jsx  # Top matches gallery
│   │   ├── DecisionPanel.jsx # User feedback component
│   │   └── Loader.jsx       # Loading spinner
│   │
│   ├── pages/               # Application pages
│   │   ├── Login.jsx        # Login page
│   │   ├── Register.jsx     # Registration page
│   │   ├── Match.jsx        # Face matching page
│   │   ├── Admin.jsx        # Admin dashboard
│   │   ├── History.jsx      # Match history
│   │   └── HowItWorks.jsx   # Information page
│   │
│   ├── utils/
│   │   └── token.js         # Token management utilities
│   │
│   ├── App.jsx              # Main app component
│   ├── main.jsx             # Entry point
│   └── index.css            # Global styles
│
├── index.html
├── package.json
├── vite.config.js
├── tailwind.config.js
└── postcss.config.js
```

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build

## API Integration

The frontend expects the following backend endpoints:

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - User login
- `GET /api/auth/profile` - Get user profile

### Face Matching
- `POST /api/match/upload` - Upload and match face
- `GET /api/match/history` - Get match history
- `GET /api/match/:id` - Get match details
- `POST /api/match/:id/feedback` - Provide feedback

### Admin
- `GET /api/admin/users` - Get all users
- `GET /api/admin/matches` - Get all matches
- `GET /api/admin/stats` - Get statistics
- `POST /api/admin/faces/upload` - Upload face to database
- `GET /api/admin/faces` - Get all faces
- `DELETE /api/admin/users/:id` - Delete user

## Key Features Explained

### Authentication Flow
- JWT token-based authentication
- Protected routes using `RequireAuth` wrapper
- Automatic token refresh on API calls
- Redirect to login on token expiration

### Face Matching
- Drag & drop or click to upload images
- Real-time preview of uploaded images
- Confidence score display
- Top-K similar faces gallery
- Match result visualization

### User Feedback
- Thumbs up/down for match accuracy
- Optional text comments
- Feedback submission to improve AI model

### Admin Dashboard
- User management
- Face database management
- Statistics and analytics
- Batch face uploads

## Environment Variables

Create a `.env` file in the root directory:

```env
VITE_API_URL=http://localhost:5000/api
```

## Deployment

### Build for Production

```bash
npm run build
```

This creates a `dist` folder with optimized production files.

### Deploy to Vercel/Netlify

1. Connect your repository
2. Set build command: `npm run build`
3. Set output directory: `dist`
4. Add environment variable: `VITE_API_URL`

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Support

For support, email support@facemindai.com or open an issue in the repository.