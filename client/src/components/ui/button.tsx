import type { ButtonHTMLAttributes } from "react";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  size?: "sm" | "default";
};

export function Button({
  className = "",
  size = "default",
  ...props
}: ButtonProps) {
  const sizeClass = size === "sm" ? "btn-sm" : "";
  return (
    <button className={`btn ${sizeClass} ${className}`.trim()} {...props} />
  );
}
