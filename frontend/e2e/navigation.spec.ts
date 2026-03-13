import { test, expect } from "@playwright/test";

test.describe("Navigation", () => {
  test("navbar is visible on hub page", async ({ page }) => {
    await page.goto("/");
    const nav = page.locator("nav");
    await expect(nav).toBeVisible();
    await expect(
      nav.locator("text=Pricing Intelligence Lab")
    ).toBeVisible();
  });

  test("navbar has chapter group dropdowns", async ({ page }) => {
    await page.goto("/");
    for (const label of [
      "I · Foundation",
      "II · Supervised",
      "III · Unsupervised",
    ]) {
      await expect(page.locator(`button:has-text("${label}")`)).toBeVisible();
    }
  });

  test("navigating to M00 works from hub", async ({ page }) => {
    await page.goto("/");
    await page.locator('a[href="/modules/m00-foundations"]').click();
    await page.waitForURL("**/modules/m00-foundations");
    await expect(
      page.getByRole("heading", { name: /Bias.Variance/ })
    ).toBeVisible();
  });

  test("navigating to M01 works from hub", async ({ page }) => {
    await page.goto("/");
    await page.locator('a[href="/modules/m01-regression"]').click();
    await page.waitForURL("**/modules/m01-regression");
    await expect(
      page.getByRole("heading", { name: "Price Elasticity" })
    ).toBeVisible();
  });

  test("clicking logo returns to hub", async ({ page }) => {
    await page.goto("/modules/m00-foundations");
    await page.locator('a[href="/"]').first().click();
    await page.waitForURL("/");
    await expect(
      page.locator("h1:has-text('Pricing Intelligence Lab')")
    ).toBeVisible();
  });
});
