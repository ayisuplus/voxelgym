import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "jsdom",
    include: ["tests/**/*.test.js"],
    environmentOptions: {
      jsdom: { url: "http://voxel.test/" },
    },
  },
  coverage: {
    provider: "v8",
    include: ["static/app.js"],
    reporter: ["text", "json", "html"],
    thresholds: {
      lines: 80,
    },
  },
});
