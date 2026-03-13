import { test, expect } from "@playwright/test";

test.describe("M00 - Foundations (Bias-Variance)", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/modules/m00-foundations");
  });

  test("page loads with module title", async ({ page }) => {
    await expect(page.locator("text=Bias-Variance")).toBeVisible();
  });

  test("has polynomial degree control", async ({ page }) => {
    const degreeControl = page.locator("text=/[Dd]egree/").first();
    await expect(degreeControl).toBeVisible();
  });

  test("train button triggers API and renders chart", async ({ page }) => {
    const trainBtn = page.locator('button:has-text("Train")').first();
    await expect(trainBtn).toBeVisible();

    const responsePromise = page.waitForResponse(
      (resp) => resp.url().includes("/api/m00/train") && resp.status() === 200
    );

    await trainBtn.click();
    const response = await responsePromise;
    const data = await response.json();

    expect(data.figures).toBeDefined();
    expect(data.figures.primary).toBeDefined();
    expect(data.metrics).toBeDefined();
    expect(data.metrics.train_rmse).toBeDefined();
    expect(data.metrics.test_rmse).toBeDefined();

    await page.waitForSelector(".js-plotly-plot", { timeout: 10_000 });
    const chart = page.locator(".js-plotly-plot").first();
    await expect(chart).toBeVisible();
  });
});
