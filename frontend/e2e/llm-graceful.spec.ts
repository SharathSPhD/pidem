import { test, expect } from "@playwright/test";

test.describe("LLM Graceful Failure", () => {
  test("M16 LLM page loads without crash", async ({ page }) => {
    await page.goto("/modules/m16-llm");
    await expect(
      page.getByRole("heading", { name: "LLM Capabilities" })
    ).toBeVisible();
  });

  test("M17 RAG page loads without crash", async ({ page }) => {
    await page.goto("/modules/m17-rag");
    await expect(
      page.getByRole("heading", { name: "RAG Pipeline", exact: true })
    ).toBeVisible();
  });

  test("LLM chat endpoint returns graceful offline message", async ({
    page,
  }) => {
    const response = await page.request.post(
      "http://127.0.0.1:8000/api/v1/llm/chat",
      {
        data: {
          query: "What affects pricing?",
          use_finetuned: false,
        },
      }
    );

    expect(response.ok()).toBeTruthy();
    const data = await response.json();
    expect(data.answer).toBeDefined();
  });

  test("LLM fine-tuned endpoint returns graceful fallback", async ({
    page,
  }) => {
    const response = await page.request.post(
      "http://127.0.0.1:8000/api/v1/llm/chat",
      {
        data: {
          query: "Explain price elasticity",
          use_finetuned: true,
        },
      }
    );

    expect(response.ok()).toBeTruthy();
    const data = await response.json();
    expect(data.answer).toBeDefined();
    expect(data.answer).toMatch(/not available|offline/i);
  });

  test("RAG query endpoint returns graceful fallback", async ({ page }) => {
    const response = await page.request.post(
      "http://127.0.0.1:8000/api/v1/llm/rag/query",
      {
        data: { query: "What is the optimal pricing strategy?" },
      }
    );

    expect(response.ok()).toBeTruthy();
    const data = await response.json();
    expect(data.answer).toBeDefined();
    expect(data.answer).toMatch(/not available|offline/i);
  });
});
