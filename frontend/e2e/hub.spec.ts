import { test, expect } from "@playwright/test";

test.describe("Hub Page", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
  });

  test("renders the hub page with title and description", async ({ page }) => {
    await expect(page.locator("h1")).toContainText(
      "Pricing Intelligence Lab"
    );
    await expect(page.locator("text=A guided journey through")).toBeVisible();
  });

  test("shows all 19 module cards", async ({ page }) => {
    const moduleCards = page.locator('a[href^="/modules/m"]');
    await expect(moduleCards).toHaveCount(19);
  });

  test("shows progress indicator", async ({ page }) => {
    await expect(page.locator("text=/\\d+\\/19 chapters/")).toBeVisible();
  });

  test("renders section headers for all 8 parts", async ({ page }) => {
    for (const label of [
      "Foundation",
      "Supervised Learning",
      "Unsupervised Learning",
      "Time Series",
      "Optimization & RL",
      "Neural & Transformers",
      "LLM & RAG",
      "Synthesis",
    ]) {
      await expect(page.locator(`text=${label}`).first()).toBeVisible();
    }
  });

  test("module cards have business questions", async ({ page }) => {
    await expect(
      page.locator("text=Why does adding more complexity")
    ).toBeVisible();
    await expect(
      page.locator("text=When does cutting your price")
    ).toBeVisible();
  });

  test("uses light theme (no dark backgrounds)", async ({ page }) => {
    const body = page.locator("body");
    await expect(body).toHaveCSS("background-color", "rgb(255, 255, 255)");
  });
});
