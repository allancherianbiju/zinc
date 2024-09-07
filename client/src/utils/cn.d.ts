declare module "../../utils/cn" {
  export function cn(...classes: string[]): string;
}

import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs) {
  return twMerge(clsx(inputs));
}
