import { useRef } from "react";
import { motion, useInView } from "framer-motion";
import { Zap, Layers, UserCheck } from "lucide-react";

interface Feature {
  icon: React.ReactNode;
  title: string;
  description: string;
  gradient: string;
  iconBg: string;
  accent: string;
}

const FEATURES: Feature[] = [
  {
    icon: <Zap className="w-5 h-5" />,
    title: "Groq-Powered Prompts",
    description:
      "Your casual description is instantly transformed into a precise, professional portrait prompt using Groq's ultra-fast inference — in under 2 seconds.",
    gradient:
      "linear-gradient(135deg, rgba(196,181,253,0.25), rgba(129,140,248,0.15))",
    iconBg: "linear-gradient(135deg, #6366f1, #8b5cf6)",
    accent: "#6366f1",
  },
  {
    icon: <Layers className="w-5 h-5" />,
    title: "FLUX.1-dev Quality",
    description:
      "State-of-the-art image generation with FLUX.1-dev — delivering photorealistic portraits with exceptional detail, natural lighting, and professional composition.",
    gradient:
      "linear-gradient(135deg, rgba(186,230,253,0.25), rgba(125,211,252,0.15))",
    iconBg: "linear-gradient(135deg, #0ea5e9, #38bdf8)",
    accent: "#0ea5e9",
  },
  {
    icon: <UserCheck className="w-5 h-5" />,
    title: "Custom LoRA — Your Face",
    description:
      "Fine-tuned on real data, our custom LoRA checkpoint ensures every portrait authentically captures the subject's identity with consistent, recognizable results.",
    gradient:
      "linear-gradient(135deg, rgba(254,205,170,0.25), rgba(251,146,60,0.12))",
    iconBg: "linear-gradient(135deg, #f97316, #fb923c)",
    accent: "#f97316",
  },
];

const containerVariants = {
  hidden: {},
  show: {
    transition: { staggerChildren: 0.12 },
  },
};

const cardVariants = {
  hidden: { opacity: 0, y: 32 },
  show: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.55, ease: [0.16, 1, 0.3, 1] },
  },
};

export function Features() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: "-80px" });

  return (
    <section id="features" className="pt-24 pb-12 px-4 relative">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.65, ease: [0.16, 1, 0.3, 1] }}
          className="text-center mb-16"
        >
          <p className="text-sm font-semibold text-violet-500 uppercase tracking-widest mb-4">
            Why it works
          </p>
          <h2 className="text-4xl lg:text-5xl font-black text-slate-900 mb-4 tracking-tight">
            Built on world-class{" "}
            <span className="gradient-text">AI infrastructure</span>
          </h2>
          <p className="text-lg text-slate-500 max-w-2xl mx-auto leading-relaxed whitespace-nowrap">
            Every component is best-in-class — from inference speed to image quality.
          </p>
        </motion.div>

        {/* Cards */}
        <motion.div
          ref={ref}
          variants={containerVariants}
          initial="hidden"
          animate={inView ? "show" : "hidden"}
          className="grid md:grid-cols-3 gap-6"
        >
          {FEATURES.map((feature) => (
            <motion.div
              key={feature.title}
              variants={cardVariants}
              whileHover={{
                y: -6,
                boxShadow:
                  "0 32px 64px rgba(31,38,135,0.14), 0 8px 24px rgba(31,38,135,0.08)",
              }}
              transition={{ type: "spring", stiffness: 200, damping: 30 }}
              className="relative rounded-3xl p-7 overflow-hidden group cursor-default"
              style={{
                background: "rgba(255,255,255,0.65)",
                backdropFilter: "blur(24px)",
                WebkitBackdropFilter: "blur(24px)",
                border: "1px solid rgba(255,255,255,0.55)",
                boxShadow:
                  "0 8px 32px rgba(31,38,135,0.07), inset 0 1px 0 rgba(255,255,255,0.8)",
              }}
            >
              {/* Card gradient background */}
              <div
                className="absolute inset-0 rounded-3xl opacity-60 group-hover:opacity-100 transition-opacity duration-300"
                style={{ background: feature.gradient }}
              />

              <div className="relative z-10 space-y-5">
                {/* Icon */}
                <div
                  className="w-11 h-11 rounded-2xl flex items-center justify-center text-white"
                  style={{
                    background: feature.iconBg,
                    boxShadow: `0 6px 18px ${feature.accent}40`,
                  }}
                >
                  {feature.icon}
                </div>

                {/* Text */}
                <div className="space-y-2">
                  <h3 className="text-lg font-bold text-slate-800">
                    {feature.title}
                  </h3>
                  <p className="text-sm text-slate-500 leading-relaxed text-justify">
                    {feature.description}
                  </p>
                </div>

                {/* Decorative accent line */}
                <div
                  className="h-0.5 w-12 rounded-full opacity-40 group-hover:w-20 group-hover:opacity-70 transition-all duration-400"
                  style={{ background: feature.iconBg }}
                />
              </div>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
