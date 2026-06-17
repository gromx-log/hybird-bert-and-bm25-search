import asyncio
import pandas as pd
from playwright.async_api import async_playwright
import os

# ─────────────────────────────────────────────
# KONFIGURASI — edit di sini
# ─────────────────────────────────────────────
CITIES = [
    {
        "name": "Jakarta Barat",
        "url": "https://www.rumah123.com/jual/jakarta-barat/rumah/",
        "output": "JAKBAR_rumah123_nlp_data.csv",
        "max_pages": 25
    },
    {
        "name": "Jakarta Pusat",
        "url": "https://www.rumah123.com/jual/jakarta-pusat/rumah/",
        "output": "JAKPUS_rumah123_nlp_data.csv",
        "max_pages": 25
    },
    {
        "name": "Tangerang",
        "url": "https://www.rumah123.com/jual/tangerang/rumah/",
        "output": "TANGERANG_rumah123_nlp_data.csv",
        "max_pages": 25
    },
]

# ─────────────────────────────────────────────
# SCRAPER PER KOTA
# ─────────────────────────────────────────────
async def scrape_city(city: dict):
    name       = city["name"]
    base_url   = city["url"]
    output_csv = city["output"]
    max_pages  = city["max_pages"]

    print(f"\n{'='*55}")
    print(f"🏙️  MULAI SCRAPING: {name}")
    print(f"{'='*55}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800}
        )
        page = await context.new_page()
        await page.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )

        print(f"📡 Membuka: {base_url}")
        await page.goto(base_url, wait_until="domcontentloaded", timeout=60000)

        print(f"\n👋 Cek browser — kalau ada CAPTCHA, solve dulu.")
        print(f"   Kalau listing sudah keliatan, tekan ENTER.")
        await asyncio.to_thread(input, f"👉 [{name}] Tekan ENTER untuk mulai scraping...")

        # Resume jika file sudah ada
        all_scraped_data = []
        if os.path.exists(output_csv):
            existing = pd.read_csv(output_csv)
            all_scraped_data = existing.to_dict("records")
            existing_titles = set(existing["title"].tolist())
            print(f"⏭️  Resume: {len(all_scraped_data)} listing sudah ada, akan skip duplikat.")
        else:
            existing_titles = set()

        for current_page in range(1, max_pages + 1):
            try:
                page_url = base_url if current_page == 1 else f"{base_url}?page={current_page}"
                print(f"\n🔄 Halaman {current_page}/{max_pages}: {page_url}")

                if current_page > 1:
                    await page.goto(page_url, wait_until="domcontentloaded", timeout=60000)

                await page.wait_for_timeout(3000)

                all_h2s = await page.query_selector_all("h2")
                page_data = []

                for h2 in all_h2s:
                    title = await h2.inner_text()
                    if title.strip() in existing_titles:
                        continue  # Skip duplikat

                    parent = await h2.query_selector(
                        "xpath=./ancestor::div[(contains(@class, 'rounded-2xl') or contains(@class, 'shadow-md')) and contains(@class, 'bg-white')][1]"
                    )
                    if parent:
                        full_card_text = await parent.inner_text()
                        a_tag = await h2.query_selector("xpath=./ancestor::a[1]")
                        url_suffix = await a_tag.get_attribute("href") if a_tag else ""
                        full_url = f"https://www.rumah123.com{url_suffix}" if url_suffix else "N/A"

                        if "Rp" in full_card_text:
                            page_data.append({
                                "kota": name,
                                "page_num": current_page,
                                "title": title.strip(),
                                "text_blob": full_card_text.replace("\n", " ").strip(),
                                "url": full_url
                            })
                            existing_titles.add(title.strip())

                all_scraped_data.extend(page_data)
                print(f"✅ {len(page_data)} listing baru | Total: {len(all_scraped_data)}")

                # Auto-save tiap halaman
                if all_scraped_data:
                    pd.DataFrame(all_scraped_data).to_csv(output_csv, index=False, encoding="utf-8")

                if len(page_data) == 0:
                    print(f"⚠️ Tidak ada listing di halaman {current_page}. Berhenti.")
                    break

            except Exception as e:
                if "closed" in str(e).lower():
                    print(f"🛑 Browser ditutup manual. Menyimpan {len(all_scraped_data)} listing.")
                else:
                    print(f"⚠️ Error halaman {current_page}: {e}")
                break

        print(f"\n🎉 [{name}] Selesai! {len(all_scraped_data)} listing disimpan ke '{output_csv}'")

        try:
            await browser.close()
        except:
            pass

    return output_csv


# ─────────────────────────────────────────────
# MAIN — jalankan semua kota satu per satu
# ─────────────────────────────────────────────
async def main():
    print("🚀 SCRAPER MULTI-KOTA RUMAH123")
    print(f"   Kota yang akan di-scrape: {', '.join(c['name'] for c in CITIES)}")
    print(f"   Max halaman per kota: {CITIES[0]['max_pages']}")
    print()

    results = []
    for city in CITIES:
        output = await scrape_city(city)
        results.append(output)
        print(f"\n✅ {city['name']} selesai → {output}")

        if city != CITIES[-1]:
            print("\n⏳ Jeda 5 detik sebelum kota berikutnya...")
            await asyncio.sleep(5)

    # Gabungkan semua hasil
    print("\n" + "="*55)
    print("📦 MENGGABUNGKAN SEMUA HASIL...")
    all_dfs = []
    for f in results:
        if os.path.exists(f):
            df = pd.read_csv(f)
            all_dfs.append(df)
            print(f"  {f}: {len(df)} baris")

    if all_dfs:
        df_combined = pd.concat(all_dfs, ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=["title"]).reset_index(drop=True)
        df_combined.to_csv("BARU_rumah123_multicity_nlp_data.csv", index=False, encoding="utf-8")
        print(f"\n✅ Gabungan disimpan ke: rumah123_multicity_nlp_data.csv")
        print(f"   Total: {len(df_combined)} listing unik")
        print(f"\n   Distribusi per kota:")
        print(df_combined["kota"].value_counts().to_string())


if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(main())