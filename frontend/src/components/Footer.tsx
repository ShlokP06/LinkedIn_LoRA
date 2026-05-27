import { motion } from "framer-motion";
import { Mail, Github, Linkedin } from "lucide-react";

const CONTACT_LINKS = [
  {
    label: "LinkedIn",
    href: "https://www.linkedin.com/in/shlok-parikh-370773335/",
    icon: <Linkedin className="w-4 h-4" />,
    color: "#0a66c2",
  },
  {
    label: "parikh.shlokp@gmail.com",
    href: "mailto:parikh.shlokp@gmail.com",
    icon: <Mail className="w-4 h-4" />,
    color: "#6366f1",
  },
  {
    label: "ShlokP06",
    href: "https://github.com/ShlokP06",
    icon: <Github className="w-4 h-4" />,
    color: "#1a1523",
  },
];

export function Footer() {
  return (
    <motion.footer
      initial={{ opacity: 0 }}
      whileInView={{ opacity: 1 }}
      viewport={{ once: true }}
      transition={{ duration: 0.6 }}
      className="relative py-8 px-6"
    >
      <div className="max-w-7xl mx-auto">
        <div className="h-px bg-gradient-to-r from-transparent via-slate-200 to-transparent mb-10" />

        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          {/* Contact links */}
          <div className="flex items-center gap-3">
            {CONTACT_LINKS.map(({ label, href, icon, color }) => (
              <motion.a
                key={label}
                href={href}
                target={href.startsWith("mailto") ? undefined : "_blank"}
                rel="noopener noreferrer"
                whileHover={{ scale: 1.08, y: -2 }}
                whileTap={{ scale: 0.95 }}
                transition={{ type: "spring", stiffness: 350, damping: 25 }}
                className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium text-slate-600 hover:text-slate-900 transition-colors"
                style={{
                  background: "rgba(255,255,255,0.6)",
                  backdropFilter: "blur(12px)",
                  WebkitBackdropFilter: "blur(12px)",
                  border: "1px solid rgba(255,255,255,0.7)",
                  boxShadow: "0 2px 8px rgba(31,38,135,0.06)",
                  color,
                }}
              >
                {icon}
                <span className="text-slate-600">{label}</span>
              </motion.a>
            ))}
          </div>

          {/* Status dot */}
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <motion.span
              animate={{ opacity: [1, 0.4, 1] }}
              transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
              className="w-1.5 h-1.5 rounded-full bg-emerald-400"
            />
            Systems operational
          </div>
        </div>
      </div>
    </motion.footer>
  );
}
