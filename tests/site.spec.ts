import { defineConfig, devices } from "npm:@playwright/test";

export default defineConfig({
  testDir: ".",
  fullyParallel: true,
  forbidOnly: !!Deno.env.get("CI"),
  retries: Deno.env.get("CI") ? 2 : 0,
  workers: Deno.env.get("CI") ? 1 : undefined,
  reporter: "html",
  use: {
    baseURL: "http://localhost:8000",
    trace: "on-first-retry",
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
  ],
  webServer: {
    command: "deno run --allow-net --allow-read jsr:@std/http@1/file-server _site",
    port: 8000,
    reuseExistingServer: !Deno.env.get("CI"),
  },
});
