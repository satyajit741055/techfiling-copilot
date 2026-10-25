type IconProps = { className?: string };

const base = {
  xmlns: 'http://www.w3.org/2000/svg',
  fill: 'none',
  viewBox: '0 0 24 24',
  strokeWidth: 1.8,
  stroke: 'currentColor',
  'aria-hidden': true,
} as const;

export const LogoMark = ({ className = 'h-8 w-8' }: IconProps) => (
  <svg viewBox="0 0 32 32" className={className} aria-hidden>
    <rect width="32" height="32" rx="8" className="fill-brand-600" />
    <path d="M8 22V14M14 22V9M20 22v-6M26 22V12" stroke="white" strokeWidth="2.4" strokeLinecap="round" transform="translate(-2 0)" />
  </svg>
);

export const ArrowUp = ({ className = 'h-5 w-5' }: IconProps) => (
  <svg {...base} className={className}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 19V5m0 0l-6 6m6-6l6 6" />
  </svg>
);

export const ClockIcon = ({ className = 'h-4 w-4' }: IconProps) => (
  <svg {...base} className={className}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6l4 2m6-2a10 10 0 11-20 0 10 10 0 0120 0z" />
  </svg>
);

export const TrashIcon = ({ className = 'h-4 w-4' }: IconProps) => (
  <svg {...base} className={className}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M14.7 6l-.4 12m-4.6 0L9.3 6m10.9 0a48 48 0 00-4.7-.5M4 5.5a48 48 0 013.8-.5m0 0V4a2 2 0 012-2h4.4a2 2 0 012 2v1m-8.4 0a48 48 0 018.4 0" />
  </svg>
);

export const AlertIcon = ({ className = 'h-5 w-5' }: IconProps) => (
  <svg {...base} className={className}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v4m0 4h.01M10.3 3.9L2.4 17.6A2 2 0 004.1 20.6h15.8a2 2 0 001.7-3L13.7 3.9a2 2 0 00-3.4 0z" />
  </svg>
);

export const CloseIcon = ({ className = 'h-4 w-4' }: IconProps) => (
  <svg {...base} className={className}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M6 6l12 12M18 6L6 18" />
  </svg>
);

export const ChevronDown = ({ className = 'h-4 w-4' }: IconProps) => (
  <svg {...base} className={className}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M6 9l6 6 6-6" />
  </svg>
);

export const DocumentIcon = ({ className = 'h-5 w-5' }: IconProps) => (
  <svg {...base} className={className}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.3V6.8a2 2 0 00-.6-1.4l-3.3-3.3a2 2 0 00-1.4-.6H7a2 2 0 00-2 2v16a2 2 0 002 2h10a2 2 0 002-2v-1.5M14 2.5V7h4.5M8.5 12h7M8.5 16h4" />
  </svg>
);

export const ShieldCheck = ({ className = 'h-5 w-5' }: IconProps) => (
  <svg {...base} className={className}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.8l2 2 4-4.3M12 3l7.5 3v5.5c0 4.3-3 8-7.5 9.5-4.5-1.5-7.5-5.2-7.5-9.5V6L12 3z" />
  </svg>
);

export const TableIcon = ({ className = 'h-5 w-5' }: IconProps) => (
  <svg {...base} className={className}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M3.5 6.5a2 2 0 012-2h13a2 2 0 012 2v11a2 2 0 01-2 2h-13a2 2 0 01-2-2v-11zM3.5 10h17M3.5 14.5h17M9.5 10v9.5" />
  </svg>
);

export const MenuIcon = ({ className = 'h-5 w-5' }: IconProps) => (
  <svg {...base} className={className}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M4 7h16M4 12h16M4 17h16" />
  </svg>
);

export const Spinner = ({ className = 'h-4 w-4' }: IconProps) => (
  <svg className={`animate-spin ${className}`} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" aria-hidden>
    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
  </svg>
);
