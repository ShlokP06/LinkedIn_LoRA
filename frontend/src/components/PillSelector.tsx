import { motion } from "framer-motion";

interface PillOption {
  value: number;
  label: string;
  badge?: string | undefined;
}

interface PillSelectorProps {
  options: PillOption[];
  value: number;
  onChange: (value: number) => void;
  layoutId: string;
}

export function PillSelector({
  options,
  value,
  onChange,
  layoutId,
}: PillSelectorProps) {
  return (
    <div className="flex flex-wrap gap-2">
      {options.map((opt) => {
        const isSelected = opt.value === value;
        return (
          <motion.button
            key={opt.value}
            onClick={() => onChange(opt.value)}
            className="relative flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-sm font-medium transition-colors duration-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-violet-400/60"
            style={{
              color: isSelected ? "#6d28d9" : "#6b7280",
            }}
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            transition={{ type: "spring", stiffness: 400, damping: 25 }}
          >
            {isSelected && (
              <motion.div
                layoutId={layoutId}
                className="absolute inset-0 rounded-xl"
                style={{
                  background:
                    "linear-gradient(135deg, rgba(196,181,253,0.5) 0%, rgba(186,230,253,0.4) 100%)",
                  border: "1px solid rgba(139,92,246,0.25)",
                  boxShadow: "0 2px 8px rgba(139,92,246,0.15)",
                }}
                transition={{ type: "spring", stiffness: 350, damping: 28 }}
              />
            )}
            <span className="relative z-10">{opt.label}</span>
            {opt.badge && (
              <span className="relative z-10 text-[10px] font-semibold px-1.5 py-0.5 rounded-md bg-gradient-to-r from-violet-500 to-indigo-500 text-white leading-none">
                {opt.badge}
              </span>
            )}
          </motion.button>
        );
      })}
    </div>
  );
}
