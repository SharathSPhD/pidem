import { test, expect } from "@playwright/test";

test.describe("M01 - Regression (Price Elasticity)", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/modules/m01-regression");
  });

  test("page loads with module title", async ({ page }) => {
    await expect(
      page.getByRole("heading", { name: "Price Elasticity" })
    ).toBeVisible();
  });

  test("train button triggers API and returns coefficients", async ({
    page,
  }) => {
    const trainBtn = page
      .locator('button:has-text("Train"), button:has-text("Build")')
      .first();
    await expect(trainBtn).toBeVisible();

    const responsePromise = page.waitForResponse(
      (resp) => resp.url().includes("/api/m01/train") && resp.status() === 200
    );

    await trainBtn.click();
    const response = await responsePromise;
    const data = await response.json();

    expect(data.figures).toBeDefined();
    expect(data.metrics).toBeDefined();
    expect(data.data.coefficients).toBeDefined();
    expect(data.data.coefficients.length).toBeGreaterThan(0);

    await page.waitForSelector(".js-plotly-plot", { timeout: 10_000 });
    const chart = page.locator(".js-plotly-plot").first();
    await expect(chart).toBeVisible();
  });
});
