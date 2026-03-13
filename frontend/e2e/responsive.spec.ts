import { test, expect } from "@playwright/test";

test.describe("Responsive Layout", () => {
  test("hub renders at desktop width (1280px)", async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto("/");
    await expect(
      page.locator("h1:has-text('Pricing Intelligence Lab')")
    ).toBeVisible();
    const moduleCards = page.locator('a[href^="/modules/m"]');
    await expect(moduleCards).toHaveCount(19);
  });

  test("hub renders at tablet width (768px)", async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.goto("/");
    await expect(
      page.locator("h1:has-text('Pricing Intelligence Lab')")
    ).toBeVisible();
    const moduleCards = page.locator('a[href^="/modules/m"]');
    await expect(moduleCards).toHaveCount(19);
  });

  test("navbar shows mobile menu button at tablet width", async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.goto("/");
    const mobileBtn = page.locator('button[aria-label="Open menu"]');
    await expect(mobileBtn).toBeVisible();
  });
});
