import openpyxl
import gspread
import os
import sys

EXCEL_FILE = "/Users/lehoangan/Documents/GitHub/ROOM/echoscene/debug_bbox/layout_ranking_report.xlsx"
SPREADSHEET_ID = "1GVVTYxuZbnfK6Ypv0Il6yVOiH-BJRaiWT5P3w-LCpbY"
TAB_NAME = "IDX3"

def main():
    if not os.path.exists(EXCEL_FILE):
        print(f"Error: Excel file not found at {EXCEL_FILE}")
        sys.exit(1)

    print(f"Reading Excel file: {EXCEL_FILE}...")
    wb = openpyxl.load_workbook(EXCEL_FILE, data_only=True)
    ws = wb.active

    # Extract all rows from Excel
    data = []
    for row in ws.iter_rows(values_only=True):
        # Convert None to empty string
        cleaned_row = [str(val) if val is not None else "" for val in row]
        data.append(cleaned_row)

    print(f"Extracted {len(data)} rows from Excel.")

    # Authenticate with gspread
    print("Authenticating with Google Sheets...")
    # Will look for credentials.json or service_account.json
    try:
        gc = gspread.oauth()
    except Exception as e:
        print("\n[!] OAuth credentials missing. Please place your Google OAuth `credentials.json` at `~/.config/gspread/credentials.json` or use service account credentials.")
        print(f"Details: {e}")
        return

    print(f"Opening spreadsheet: {SPREADSHEET_ID}...")
    sh = gc.open_by_key(SPREADSHEET_ID)

    try:
        worksheet = sh.worksheet(TAB_NAME)
    except gspread.exceptions.WorksheetNotFound:
        print(f"Worksheet '{TAB_NAME}' not found. Creating it...")
        worksheet = sh.add_worksheet(title=TAB_NAME, rows=len(data)+10, cols=len(data[0])+5)

    print(f"Clearing existing content in '{TAB_NAME}' sheet...")
    worksheet.clear()

    print(f"Updating '{TAB_NAME}' with new data...")
    worksheet.update(values=data, range_name="A1")

    print(f"Successfully uploaded layout ranking report to '{TAB_NAME}' tab!")

if __name__ == "__main__":
    main()
