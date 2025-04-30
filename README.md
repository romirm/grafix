# Grafix 
A web app that uses Google's Gemini API to parse through student information and generate an interactive web graph. Made for KTP by Romir Mohan.

## 🚧 Built With

* [![React][React.js]][React-url]
* [![Firebase][Firebase]][Firebase-url]
* [![Gemini][Gemini]][Gemini-url]

<!-- Badge References -->

<!-- Vite -->
[Vite]: https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white
[Vite-url]: https://vitejs.dev/

<!-- React -->
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/

<!-- Firebase -->
[Firebase]: https://img.shields.io/badge/Firebase-FFCA28?style=for-the-badge&logo=firebase&logoColor=white
[Firebase-url]: https://firebase.google.com/

<!-- Gemini -->
[Gemini]: https://img.shields.io/badge/Gemini%20API-4285F4?style=for-the-badge&logo=google&logoColor=white
[Gemini-url]: https://ai.google.dev/

---

## 🛠️ Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/your-username/aptly.git
cd grafix
```

### 2. Install dependencies

```bash
npm install
```

### 3. Set up environment variables

Create a `.env` file at the root of your project and add any required keys (e.g. Gemini API credentials, if applicable).

```env
# Example
VITE_GEMINI_API_KEY=your-key-here
```

### 4. Run locally

```bash
npm run dev
```

The app will be available at `http://localhost:5173`.

---

## 🌐 Deployment

This project uses **Firebase Hosting**.

### Build for production:

```bash
npm run build
```

### Deploy:

```bash
firebase deploy
```

Make sure `dist/` is your public directory and that Firebase is configured for single-page apps.

---

## 📸 Features
- See who you connect with
- Filter through student interests
- Get insight quickly

---

## 🤝 Contributions
Feel free to fork!
