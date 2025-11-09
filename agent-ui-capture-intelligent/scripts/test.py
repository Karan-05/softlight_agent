from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    browser = p.firefox.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto("https://www.notion.so/login")
    input("Log in, then press Enter here…")
    context.storage_state(path="/Users/karanallagh/Desktop/light_2/agent-ui-capture-intelligent/secrets/notion-storage.json")
    browser.close()