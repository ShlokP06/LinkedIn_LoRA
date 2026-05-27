/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        display: ["Inter", "system-ui", "sans-serif"],
      },
      colors: {
        lavender: {
          50: "#f5f3ff",
          100: "#ede9fe",
          200: "#ddd6fe",
          300: "#c4b5fd",
          400: "#a78bfa",
          500: "#8b5cf6",
        },
        aurora: {
          sky: "#bae6fd",
          lavender: "#e9d5ff",
          peach: "#fed7aa",
          mint: "#bbf7d0",
          rose: "#fecdd3",
        },
      },
      animation: {
        aurora: "aurora 12s ease-in-out infinite",
        "aurora-slow": "aurora 20s ease-in-out infinite reverse",
        shimmer: "shimmer 2s linear infinite",
        "float-1": "float1 4s ease-in-out infinite",
        "float-2": "float2 5s ease-in-out infinite",
        "float-3": "float3 6s ease-in-out infinite",
        "spin-slow": "spin 8s linear infinite",
        "border-rotate": "borderRotate 3s linear infinite",
        grain: "grain 8s steps(10) infinite",
        "pulse-glow": "pulseGlow 2s ease-in-out infinite",
      },
      keyframes: {
        aurora: {
          "0%, 100%": { backgroundPosition: "0% 50%" },
          "50%": { backgroundPosition: "100% 50%" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        float1: {
          "0%, 100%": { transform: "translateY(0px) rotate(-2deg)" },
          "50%": { transform: "translateY(-16px) rotate(-2deg)" },
        },
        float2: {
          "0%, 100%": { transform: "translateY(0px) rotate(2deg)" },
          "50%": { transform: "translateY(-12px) rotate(2deg)" },
        },
        float3: {
          "0%, 100%": { transform: "translateY(0px) rotate(-1deg)" },
          "50%": { transform: "translateY(-20px) rotate(-1deg)" },
        },
        borderRotate: {
          "0%": { "--angle": "0deg" },
          "100%": { "--angle": "360deg" },
        },
        grain: {
          "0%, 100%": { transform: "translate(0, 0)" },
          "10%": { transform: "translate(-2%, -3%)" },
          "20%": { transform: "translate(3%, 1%)" },
          "30%": { transform: "translate(-1%, 4%)" },
          "40%": { transform: "translate(2%, -2%)" },
          "50%": { transform: "translate(-3%, 2%)" },
          "60%": { transform: "translate(1%, -4%)" },
          "70%": { transform: "translate(-2%, 3%)" },
          "80%": { transform: "translate(3%, -1%)" },
          "90%": { transform: "translate(-1%, 2%)" },
        },
        pulseGlow: {
          "0%, 100%": { opacity: "0.6" },
          "50%": { opacity: "1" },
        },
      },
      backgroundSize: {
        "300%": "300%",
        "400%": "400%",
      },
      backdropBlur: {
        xs: "2px",
      },
      boxShadow: {
        glass: "0 8px 32px rgba(31, 38, 135, 0.08)",
        "glass-lg":
          "0 24px 64px rgba(31, 38, 135, 0.12), 0 8px 24px rgba(31, 38, 135, 0.06)",
        "glass-xl":
          "0 32px 80px rgba(31, 38, 135, 0.15), 0 12px 32px rgba(31, 38, 135, 0.08)",
        glow: "0 0 24px rgba(139, 92, 246, 0.3)",
        "glow-blue": "0 0 24px rgba(96, 165, 250, 0.4)",
        "glow-lg": "0 0 48px rgba(139, 92, 246, 0.25)",
      },
    },
  },
  plugins: [],
};
