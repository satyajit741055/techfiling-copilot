/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable React strict mode for better development experience
  reactStrictMode: true,
  
  // Allow images from any domain (for future use)
  images: {
    remotePatterns: [],
  },
  
  // Allow dev server access from browser preview
  allowedDevOrigins: ['127.0.0.1', 'localhost'],
};

module.exports = nextConfig;
