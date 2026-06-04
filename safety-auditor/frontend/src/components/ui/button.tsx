import { cn } from "@/lib/utils";
import { ButtonHTMLAttributes, forwardRef } from "react";

export const Button = forwardRef<
  HTMLButtonElement,
  ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "default" | "outline" | "ghost" }
>(({ className, variant = "default", ...props }, ref) => (
  <button
    ref={ref}
    className={cn(
      "inline-flex items-center justify-center rounded-lg px-4 py-2 text-sm font-medium transition",
      variant === "default" && "bg-accent text-white hover:bg-violet-700 disabled:opacity-50",
      variant === "outline" && "border border-border bg-white hover:bg-surface",
      variant === "ghost" && "hover:bg-accent-muted text-slate-700",
      className
    )}
    {...props}
  />
));
Button.displayName = "Button";
