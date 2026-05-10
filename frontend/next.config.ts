import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  allowedDevOrigins: ["192.168.0.138", "192.168.0.163"],
  // We need to configure webpack to ignore the "canvas" module. 
  // react-pdf optionally uses it for Node.js rendering, but we only use it in the browser.
  webpack: (config) => {
    config.resolve.alias.canvas = false;
    return config;
  },
  // Silence the Turbopack warning about the webpack config above.
  // Turbopack often handles missing optional peer dependencies like 'canvas' gracefully on its own.
  turbopack: {},
};

export default nextConfig;
