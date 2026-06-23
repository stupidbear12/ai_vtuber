/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        sion: {
          primary: "#7C3AED",
          secondary: "#A78BFA",
          bg: "#0F0F1A",
          card: "#1A1A2E",
          input: "#252540",
          text: "#E2E8F0",
          muted: "#94A3B8",
        },
      },
    },
  },
  plugins: [],
};
