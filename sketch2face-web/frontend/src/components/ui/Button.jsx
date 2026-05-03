import { cn } from '../../lib/utils';

const buttonVariants = {
  default: "bg-primary text-primary-foreground hover:bg-primary/90 hover:shadow-[0_0_20px_hsl(190_95%_50%/0.4)]",
  destructive: "bg-destructive text-destructive-foreground hover:bg-destructive/90",
  outline: "border border-primary/50 bg-transparent text-primary hover:bg-primary/10 hover:border-primary hover:shadow-[0_0_15px_hsl(190_95%_50%/0.3)]",
  secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/80",
  ghost: "hover:bg-accent/10 hover:text-accent",
  link: "text-primary underline-offset-4 hover:underline",
  cyber: "bg-gradient-to-r from-primary to-accent text-primary-foreground font-semibold hover:shadow-[0_0_30px_hsl(190_95%_50%/0.5)] hover:scale-105",
  hero: "bg-primary text-primary-foreground font-semibold text-base px-8 py-6 hover:shadow-[0_0_40px_hsl(190_95%_50%/0.6)] hover:scale-105",
  heroOutline: "border-2 border-primary/60 bg-transparent text-primary font-semibold text-base px-8 py-6 hover:bg-primary/10 hover:border-primary hover:shadow-[0_0_30px_hsl(190_95%_50%/0.4)]",
  success: "bg-success text-success-foreground hover:bg-success/90 hover:shadow-[0_0_20px_hsl(150_80%_45%/0.4)]",
};

const buttonSizes = {
  default: "h-10 px-4 py-2",
  sm: "h-9 rounded-md px-3",
  lg: "h-11 rounded-md px-8",
  xl: "h-14 rounded-lg px-10 text-base",
  icon: "h-10 w-10",
};

export function Button({ 
  className, 
  variant = "default", 
  size = "default", 
  children, 
  ...props 
}) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-all duration-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0",
        buttonVariants[variant] || buttonVariants.default,
        buttonSizes[size] || buttonSizes.default,
        className
      )}
      {...props}
    >
      {children}
    </button>
  );
}
