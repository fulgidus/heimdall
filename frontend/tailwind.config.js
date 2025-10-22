/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                // Custom palette
                "oxford-blue": "#0b132bff",
                "sea-green": "#09814aff",
                "french-gray": "#c7cedbff",
                "light-green": "#70ee9cff",
                "neon-blue": "#446df6ff",

                // Semantic colors
                "primary": "#0b132bff",
                "secondary": "#09814aff",
                "accent": "#446df6ff",
                "success": "#70ee9cff",
                "neutral": "#c7cedbff",
            },
            backgroundColor: {
                "oxford-blue": "#0b132bff",
                "sea-green": "#09814aff",
                "french-gray": "#c7cedbff",
                "light-green": "#70ee9cff",
                "neon-blue": "#446df6ff",
            },
            textColor: {
                "oxford-blue": "#0b132bff",
                "sea-green": "#09814aff",
                "french-gray": "#c7cedbff",
                "light-green": "#70ee9cff",
                "neon-blue": "#446df6ff",
            },
            borderColor: {
                "oxford-blue": "#0b132bff",
                "sea-green": "#09814aff",
                "french-gray": "#c7cedbff",
                "light-green": "#70ee9cff",
                "neon-blue": "#446df6ff",
            },
            backgroundImage: {
                "gradient-top": "linear-gradient(0deg, #0b132bff, #09814aff, #c7cedbff, #70ee9cff, #446df6ff)",
                "gradient-right": "linear-gradient(90deg, #0b132bff, #09814aff, #c7cedbff, #70ee9cff, #446df6ff)",
                "gradient-bottom": "linear-gradient(180deg, #0b132bff, #09814aff, #c7cedbff, #70ee9cff, #446df6ff)",
                "gradient-left": "linear-gradient(270deg, #0b132bff, #09814aff, #c7cedbff, #70ee9cff, #446df6ff)",
                "gradient-top-right": "linear-gradient(45deg, #0b132bff, #09814aff, #c7cedbff, #70ee9cff, #446df6ff)",
                "gradient-bottom-right": "linear-gradient(135deg, #0b132bff, #09814aff, #c7cedbff, #70ee9cff, #446df6ff)",
                "gradient-top-left": "linear-gradient(225deg, #0b132bff, #09814aff, #c7cedbff, #70ee9cff, #446df6ff)",
                "gradient-bottom-left": "linear-gradient(315deg, #0b132bff, #09814aff, #c7cedbff, #70ee9cff, #446df6ff)",
                "gradient-radial": "radial-gradient(#0b132bff, #09814aff, #c7cedbff, #70ee9cff, #446df6ff)",
            },
            spacing: {
                xs: "0.5rem",
                sm: "1rem",
                md: "1.5rem",
                lg: "2rem",
                xl: "3rem",
                "2xl": "4rem",
            },
            borderRadius: {
                xs: "0.25rem",
                sm: "0.5rem",
                md: "0.75rem",
                lg: "1rem",
                xl: "1.5rem",
                full: "9999px",
            },
            fontSize: {
                xs: ["0.75rem", { lineHeight: "1rem" }],
                sm: ["0.875rem", { lineHeight: "1.25rem" }],
                base: ["1rem", { lineHeight: "1.5rem" }],
                lg: ["1.125rem", { lineHeight: "1.75rem" }],
                xl: ["1.25rem", { lineHeight: "1.75rem" }],
                "2xl": ["1.5rem", { lineHeight: "2rem" }],
                "3xl": ["1.875rem", { lineHeight: "2.25rem" }],
                "4xl": ["2.25rem", { lineHeight: "2.5rem" }],
            },
            fontFamily: {
                sans: ["Inter", "system-ui", "sans-serif"],
                mono: ["Fira Code", "monospace"],
            },
            boxShadow: {
                xs: "0 1px 2px 0 rgba(0, 0, 0, 0.05)",
                sm: "0 1px 2px 0 rgba(0, 0, 0, 0.1)",
                md: "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
                lg: "0 10px 15px -3px rgba(0, 0, 0, 0.1)",
                xl: "0 20px 25px -5px rgba(0, 0, 0, 0.1)",
                "2xl": "0 25px 50px -12px rgba(0, 0, 0, 0.25)",
            },
        },
    },
    plugins: [],
}
