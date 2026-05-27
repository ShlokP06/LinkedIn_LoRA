import { useRef, useState } from "react";
import { motion, useInView } from "framer-motion";
import { Eye } from "lucide-react";
import { EXAMPLES } from "../data/examples";

const containerVariants = {
  hidden: {},
  show: {
    transition: { staggerChildren: 0.1 },
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

function ExampleCard({ image, prompt }: { image: string; prompt: string }) {
  const [imgError, setImgError] = useState(false);

  return (
    <motion.div
      variants={cardVariants}
      whileHover={{
        y: -6,
        boxShadow:
          "0 32px 64px rgba(31,38,135,0.14), 0 8px 24px rgba(31,38,135,0.08)",
      }}
      transition={{ type: "spring", stiffness: 200, damping: 30 }}
      className="relative rounded-3xl overflow-hidden group cursor-default"
      style={{
        background: "rgba(255,255,255,0.65)",
        backdropFilter: "blur(24px)",
        WebkitBackdropFilter: "blur(24px)",
        border: "1px solid rgba(255,255,255,0.55)",
        boxShadow:
          "0 8px 32px rgba(31,38,135,0.07), inset 0 1px 0 rgba(255,255,255,0.8)",
      }}
    >
      {/* Image */}
      <div className="aspect-[3/4] w-full overflow-hidden bg-gradient-to-br from-slate-100 to-slate-200/60">
        {!imgError ? (
          <img
            src={image}
            alt="Generated portrait example"
            className="w-full h-full object-cover object-top transition-transform duration-500 group-hover:scale-105"
            onError={() => setImgError(true)}
          />
        ) : (
          <div className="w-full h-full flex flex-col items-center justify-center gap-2">
            <div
              className="w-12 h-12 rounded-2xl flex items-center justify-center text-white text-xl"
              style={{ background: "linear-gradient(135deg, #6366f1, #8b5cf6)" }}
            >
              ✦
            </div>
            <span className="text-slate-400 text-xs font-medium">Image coming soon</span>
          </div>
        )}
      </div>

      {/* Prompt text */}
      <div className="p-5">
        <p className="text-xs font-semibold text-violet-500 uppercase tracking-widest mb-2">
          Prompt
        </p>
        <p className="text-slate-500 text-sm leading-relaxed">
          {prompt}
        </p>
      </div>

      {/* Subtle gradient overlay on hover */}
      <div
        className="absolute inset-0 rounded-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"
        style={{
          background:
            "linear-gradient(135deg, rgba(196,181,253,0.08), rgba(129,140,248,0.05))",
        }}
      />
    </motion.div>
  );
}

export function Examples() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: "-80px" });

  return (
    <section className="py-10 lg:py-14 px-4 relative">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.65, ease: [0.16, 1, 0.3, 1] }}
          className="text-center mb-4"
        >
          <motion.div
            className="inline-flex items-center gap-3 px-6 py-2.5 rounded-full rotating-border mb-6"
            whileHover={{ scale: 1.04 }}
            transition={{ type: "spring", stiffness: 300, damping: 25 }}
            style={{
              background: "rgba(255,255,255,0.75)",
              backdropFilter: "blur(20px)",
              WebkitBackdropFilter: "blur(20px)",
            }}
          >
            <Eye className="w-4 h-4 text-violet-500 flex-shrink-0" />
            <span className="text-sm font-bold uppercase tracking-widest gradient-text">
              See it in action
            </span>
          </motion.div>
        </motion.div>

        {/* Cards grid */}
        <motion.div
          id="examples"
          ref={ref}
          variants={containerVariants}
          initial="hidden"
          animate={inView ? "show" : "hidden"}
          className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6"
        >
          {EXAMPLES.map((ex, i) => (
            <ExampleCard key={i} image={ex.image} prompt={ex.prompt} />
          ))}
        </motion.div>
      </div>
    </section>
  );
}
