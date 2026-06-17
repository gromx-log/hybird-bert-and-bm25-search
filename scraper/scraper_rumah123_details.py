import asyncio
import pandas as pd
from playwright.async_api import async_playwright
import os
import random

# Create a lock for safely writing to the CSV from multiple tasks
csv_lock = asyncio.Lock()
# Prevent multiple tasks from requesting input at exactly the same time
captcha_lock = asyncio.Lock()

# Global event to freeze all tabs if one tab hits a Captcha
captcha_cleared_event = asyncio.Event()
captcha_cleared_event.set() # Initially clear to run (True)

global_df = None

async def scrape_single_url(sem, context, url, output_csv):
    global global_df
    async with sem:
        # Before doing anything, make sure we aren't paused globally by a Captcha
        await captcha_cleared_event.wait()
        
        print(f"🚀 Started: {url}")
        
        page = await context.new_page()
        await page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        try:
            # Stagger the requests randomly to avoid sending 4 simultaneous SYN packets to Cloudflare
            await asyncio.sleep(random.uniform(3.0, 7.0)) 
            
            # Check again just before making the network request
            await captcha_cleared_event.wait() 
            
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            await page.wait_for_timeout(4000) 


            # Look for Captcha 
            page_title = await page.title()
            if any(kw in page_title for kw in ["Captcha", "Security", "Just a moment", "Attention Required"]):
                
                # If the event is SET, it means we are the first tab to notice the block
                if captcha_cleared_event.is_set():
                    # Freeze all other tabs
                    captcha_cleared_event.clear()
                    
                    async with captcha_lock:
                        print(f"\n🛑 Cloudflare/Captcha triggered on: {url}!")
                        print("⚠️ All scraper tabs have been paused to prevent looping.")
                        print("👉 Please solve the security check in the browser window.")
                        await asyncio.to_thread(input, "👉 Press ENTER here in the terminal ONLY AFTER you have solved it...")
                        
                        # Unfreeze all tabs
                        captcha_cleared_event.set()
                else:
                    # Another tab already triggered the freeze. We just wait respectfully.
                    print(f"⏸️ Tab paused waiting for Captcha clearance: {url}")
                    await captcha_cleared_event.wait()
                    
                # Reload page after solving (all blocked tabs need to reload their target)
                print(f"🔄 Reloading {url} after captcha clearance...")
                await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                await page.wait_for_timeout(3000)

            # Extract the description using the h3 text "Deskripsi" and its container
            desc_element = await page.query_selector("xpath=//h3[contains(text(), 'Deskripsi')]/following-sibling::p")
            
            # If not found via exact sibling, look inside parent
            if not desc_element:
                desc_element = await page.query_selector("xpath=//h3[contains(text(), 'Deskripsi')]/parent::div//p")
                
            if desc_element:
                full_desc = await desc_element.inner_text()
                full_desc = full_desc.replace("\n", " ").strip()
            else:
                full_desc = "NOT_FOUND"

            print(f"✅ Extracted ({len(full_desc)} chars) from: {url}")
            
            async with csv_lock:
                global_df.loc[global_df['url'] == url, 'full_description'] = full_desc
                global_df.to_csv(output_csv, index=False, encoding='utf-8')

        except Exception as e:
            print(f"⚠️ Error {url}: {str(e)[:100]}...")
            async with csv_lock:
                global_df.loc[global_df['url'] == url, 'full_description'] = "ERROR"
                global_df.to_csv(output_csv, index=False, encoding='utf-8')
        finally:
            await page.close()


async def run_details_scraper(input_csv="SURABAYA_rumah123_nlp_data.csv", output_csv="SURABAYA_rumah123_full_details.csv", max_concurrent=4):
    global global_df
    
    if not os.path.exists(input_csv):
        print(f"❌ Input file {input_csv} not found. Please run the base scraper (scraper_rumah123.py) first.")
        return

    global_df = pd.read_csv(input_csv)
    if 'url' not in global_df.columns:
        print("❌ No 'url' column found in the CSV. Please run the updated base scraper (scraper_rumah123.py) first.")
        return

    # Check for existing progress to resume
    existing_urls = set()
    if os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            if 'full_description' in existing_df.columns:
                # Find URLs that already have a successful description
                scraped = existing_df.dropna(subset=['full_description'])
                scraped = scraped[scraped['full_description'] != 'ERROR']
                existing_urls = set(scraped['url'].unique())
                print(f"⏭️ Resuming progress... found {len(existing_urls)} properties already scraped in '{output_csv}'. Skipping them.")
                
                # Merge existing dataframe data back into the main `global_df`
                global_df = global_df.merge(existing_df[['url', 'full_description']], on='url', how='left')
        except Exception as e:
            print(f"⚠️ Could not load '{output_csv}' for resuming: {e}")

    urls = global_df['url'].dropna().unique().tolist()
    # Filter out invalid URLs and already scraped URLs
    urls = [u for u in urls if str(u).startswith("http") and str(u) != "N/A" and u not in existing_urls]

    if 'full_description' not in global_df.columns:
        global_df['full_description'] = None

    if not urls:
        print("❌ No valid URLs found to scrape.")
        return

    print(f"📊 Found {len(urls)} unique properties to scrape for full details using {max_concurrent} concurrent tabs.")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            viewport={'width': 1280, 'height': 800}
        )
        
        sem = asyncio.Semaphore(max_concurrent)
        
        # Launch all tasks asynchronously
        tasks = [scrape_single_url(sem, context, url, output_csv) for url in urls]
        await asyncio.gather(*tasks)

        print("\n🔌 Closing browser...")
        try:
            await browser.close()
        except:
            pass
        
    print(f"\n🎉 Success! Auto-saved all detailed info to '{output_csv}'.")
    print("Preview:")
    print(global_df.head(2))

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    
    # You can change the number of simultaneous browsers/tabs here
    asyncio.run(run_details_scraper(
    input_csv="BARU_rumah123_multicity_nlp_data.csv",
    output_csv="BARU_rumah123_multicity_full_details.csv",
    max_concurrent=1
))
