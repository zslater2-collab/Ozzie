"""
One-time cleanup for the KProp Google Sheet tab (June 28, 2026).

Why: the KProp tab was created June 25, 2026, then its column layout was rewritten three
times in three days as the K-prop signal itself was redefined (K-composite -> Kalshi-implied
-> the current over-favorite price-bias signal, June 27). New columns were inserted mid-list,
and _get_or_create_ws overwrote the header row in place without moving old data cells -- so the
June 25-26 rows now sit misaligned under the current header AND are from superseded (one leaked)
signal definitions. This script separates them out.

What it does:
  * Reads the live KProp tab (its header row is already the current over-favorite schema).
  * Splits rows into CURRENT (k_prop_tier is 'base' or 'sharp' -- only the June-27 schema has a
    tier) vs LEGACY (everything else).
  * --apply: copies LEGACY rows to a 'KProp_legacy_archive' tab (nothing is destroyed -- the
    archive keeps a full copy), then rewrites the KProp tab to header + CURRENT rows only.
  * Default (no --apply): DRY RUN. Prints the split and sample rows; changes nothing.

Run it where the creds exist:
    GOOGLE_SHEETS_CREDENTIALS env var set (same JSON the app uses).
    python clean_kprop_tab.py            # dry run -- preview
    python clean_kprop_tab.py --apply    # actually clean the tab
"""
import os, sys, json

SHEETS_ID    = '1AKalzsMqSDmLe5j26de3R6_YVrpptJVZY9tWrZNCbHs'
KPROP_TAB    = 'KProp'
ARCHIVE_TAB  = 'KProp_legacy_archive'
TIER_COL_NAME = 'k_prop_tier'
CURRENT_TIERS = ('base', 'sharp')

def main():
    apply = '--apply' in sys.argv
    creds_raw = os.environ.get('GOOGLE_SHEETS_CREDENTIALS', '')
    if not creds_raw:
        print('ERROR: GOOGLE_SHEETS_CREDENTIALS env var not set. Run this where the app creds live '
              '(Render shell, or set the var locally to the service-account JSON).')
        sys.exit(1)

    import gspread
    from google.oauth2.service_account import Credentials
    scopes = ['https://www.googleapis.com/auth/spreadsheets',
              'https://www.googleapis.com/auth/drive']
    gc = gspread.authorize(Credentials.from_service_account_info(json.loads(creds_raw), scopes=scopes))
    sh = gc.open_by_key(SHEETS_ID)
    ws = sh.worksheet(KPROP_TAB)

    values = ws.get_all_values()
    if not values:
        print('KProp tab is empty -- nothing to do.'); return
    header, rows = values[0], values[1:]
    if TIER_COL_NAME not in header:
        print(f"ERROR: '{TIER_COL_NAME}' not in header -- live header is not the current schema. "
              f"Header found: {header}. Aborting so nothing is mis-sorted.")
        sys.exit(1)
    tci = header.index(TIER_COL_NAME)

    def is_current(r):
        return len(r) > tci and r[tci].strip().lower() in CURRENT_TIERS

    current = [r for r in rows if is_current(r)]
    legacy  = [r for r in rows if not is_current(r)]

    def date_span(rs):
        ds = sorted({r[0] for r in rs if r and r[0]})
        return f'{ds[0]} .. {ds[-1]}' if ds else '(none)'

    print(f'KProp tab: {len(rows)} data rows.')
    print(f'  CURRENT (tier base/sharp): {len(current):4d}   dates {date_span(current)}')
    print(f'  LEGACY  (to archive)     : {len(legacy):4d}   dates {date_span(legacy)}')
    if legacy:
        print('\n  Sample legacy rows (first 3, first 6 cols):')
        for r in legacy[:3]:
            print('   ', r[:6])

    if not legacy:
        print('\nNo legacy rows found -- tab is already clean. Nothing to do.'); return

    if not apply:
        print('\nDRY RUN -- no changes made. Re-run with --apply to archive the legacy rows and '
              'rewrite the KProp tab to current rows only.')
        return

    # 1) archive legacy rows (full copy preserved) BEFORE touching the main tab
    try:
        arch = sh.worksheet(ARCHIVE_TAB)
    except gspread.exceptions.WorksheetNotFound:
        arch = sh.add_worksheet(title=ARCHIVE_TAB, rows=max(100, len(legacy) + 10), cols=len(header) + 2)
        arch.append_row(['archived_from_KProp', f'header below was the live header at archive time'],
                        value_input_option='RAW')
        arch.append_row(header, value_input_option='RAW')
    arch.append_rows(legacy, value_input_option='RAW')
    print(f'\nArchived {len(legacy)} legacy row(s) to "{ARCHIVE_TAB}".')

    # 2) rewrite KProp = header + current rows only
    ws.clear()
    ws.update('A1', [header] + current, value_input_option='RAW')
    print(f'KProp tab rewritten: {len(current)} current row(s) under the current header. Done.')

if __name__ == '__main__':
    main()
