import { chromium, type Browser, type Page } from "npm:playwright";

const BASE = "http://localhost:8000";
const SITE_DIR = "_site";

async function serve(dir: string, port: number): Promise<Deno.ChildProcess> {
  const cmd = new Deno.Command("deno", {
    args: [
      "run",
      "--allow-net",
      "--allow-read",
      "jsr:@std/http@1/file-server",
      "--port", String(port),
      dir,
    ],
    stdout: "null",
    stderr: "null",
  });
  const proc = cmd.spawn();
  for (let i = 0; i < 10; i++) {
    await new Promise((r) => setTimeout(r, 500));
    try {
      const res = await fetch(`http://localhost:${port}/`);
      if (res.ok) return proc;
    } catch {
      // not ready yet
    }
  }
  throw new Error("server did not start in time");
}

async function withBrowser<T>(fn: (b: Browser) => Promise<T>): Promise<T> {
  const browser = await chromium.launch({ headless: true });
  try {
    return await fn(browser);
  } finally {
    await browser.close();
  }
}

async function withPage<T>(b: Browser, url: string, fn: (p: Page) => Promise<T>): Promise<T> {
  const page = await b.newPage();
  await page.goto(url, { waitUntil: "load" });
  return await fn(page);
}

let proc: Deno.ChildProcess | null = null;

Deno.test({
  name: "start server",
  sanitizeResources: false,
  sanitizeOps: false,
  fn: async () => {
    proc = await serve(SITE_DIR, 8000);
  },
});

Deno.test({
  name: "homepage loads and has correct title",
  sanitizeResources: false,
  sanitizeOps: false,
  fn: async () => {
    await withBrowser(async (browser) => {
      await withPage(browser, BASE, async (page) => {
        const title = await page.title();
        console.assert(title === "Home", `expected "Home", got "${title}"`);
        const content = await page.textContent("main") ?? "";
        console.assert(content.includes("Welcome to my blog"), "expected welcome text in main");
      });
    });
  },
});

Deno.test({
  name: "navigation links are present",
  sanitizeResources: false,
  sanitizeOps: false,
  fn: async () => {
    await withBrowser(async (browser) => {
      await withPage(browser, BASE, async (page) => {
        const links = await page.locator("nav.site-nav a").allTextContents();
        const expected = [
          "Home",
          "Blog",
          "Experiments",
          "Looking Forward",
          "My Setup",
          "Bookmarks",
          "Tutorials",
        ];
        for (const name of expected) {
          console.assert(links.includes(name), `missing nav link: "${name}"`);
        }
      });
    });
  },
});

Deno.test({
  name: "blog page lists posts",
  sanitizeResources: false,
  sanitizeOps: false,
  fn: async () => {
    await withBrowser(async (browser) => {
      await withPage(browser, `${BASE}/blog.html`, async (page) => {
        const h1 = await page.textContent("h1");
        console.assert(h1 === "Blog Posts", `expected "Blog Posts", got "${h1}"`);
        const posts = await page.locator("article").count();
        console.assert(posts > 0, "expected at least one blog post");
      });
    });
  },
});

Deno.test({
  name: "site has valid HTML structure",
  sanitizeResources: false,
  sanitizeOps: false,
  fn: async () => {
    await withBrowser(async (browser) => {
      await withPage(browser, BASE, async (page) => {
        const header = page.locator("header.site-header");
        await header.waitFor({ state: "visible", timeout: 5000 });
        console.assert(await header.isVisible(), "site header should be visible");

        const footer = page.locator("footer");
        console.assert(await footer.isVisible(), "footer should be visible");
        const footerText = await footer.textContent();
        console.assert(footerText?.includes("Raleigh"), 'footer should mention "Raleigh"');
      });
    });
  },
});

Deno.test({
  name: "stop server",
  sanitizeResources: false,
  sanitizeOps: false,
  fn: () => {
    if (proc) {
      proc.kill("SIGTERM");
      proc = null;
    }
  },
});
