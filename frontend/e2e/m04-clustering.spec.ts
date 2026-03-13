import { test, expect } from "@playwright/test";

test.describe("M04 - Clustering (Market Segmentation)", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/modules/m04-clustering");
  });

  test("page loads with module title", async ({ page }) => {
    await expect(
      page.getByRole("heading", { name: /Segmentation/ })
    ).toBeVisible();
  });

  test("train button triggers API and renders chart", async ({ page }) => {
    const trainBtn = page.locator('button:has-text("Train")').first();
    await expect(trainBtn).toBeVisible();

    const responsePromise = page.waitForResponse(
      (resp) => resp.url().includes("/api/m04/train") && resp.status() === 200
    );

    await trainBtn.click();
    const response = await responsePromise;
    const data = await response.json();

    expect(data.figures).toBeDefined();
    expect(Object.keys(data.figures).length).toBeGreaterThan(0);
    expect(data.metrics).toBeDefined();

    await page.waitForSelector(".js-plotly-plot", { timeout: 15_000 });
    const chart = page.locator(".js-plotly-plot").first();
    await expect(chart).toBeVisible();
  });
});
